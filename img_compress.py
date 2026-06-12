#!/usr/bin/env python3
"""批量图片压缩脚本 - 适合项目构建流程，支持输出目录、增量压缩、多格式报告、Manifest"""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Optional


try:
    import yaml
except ImportError:
    sys.exit("需要 PyYAML: pip install PyYAML")

try:
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow: pip install Pillow")

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

    class FileSystemEventHandler:  # 占位，使模块级类定义不报错
        pass


EXIT_OK = 0
EXIT_ALL_FAILED = 1
EXIT_REPORT_WRITE_FAILED = 2
EXIT_INVALID_ARGS = 3
EXIT_BUDGET_EXCEEDED = 4
EXIT_REFERENCE_VALIDATION_FAILED = 5


CACHE_VERSION = 2
MIN_SUFFIX = ".min"


@dataclass
class EntryConfig:
    """多入口配置：每个入口对应一组源目录、输出目录、manifest"""
    name: str
    source_root: str
    output_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    report_prefix: Optional[str] = None


@dataclass
class BudgetRule:
    """体积预算规则：按目录前缀或策略名设置压缩后总体积上限"""
    name: str
    max_bytes: int
    scope_type: str = "directory"  # directory | strategy
    scope_value: str = ""  # 目录相对前缀 或 策略名


@dataclass
class ReferenceIssue:
    """引用校验发现的问题"""
    file_path: str
    issue_type: str  # stale_source | broken_link | external (external仅统计不报错)
    raw_path: str
    resolved_path: Optional[str] = None
    details: str = ""


@dataclass
class Strategy:
    name: str
    description: str = ""
    path_pattern: Optional[str] = None
    max_dimension: Optional[int] = None
    min_dimension: Optional[int] = None
    format: str = "original"
    quality: int = 85
    method: int = 4
    target_ratio: Optional[float] = None
    quality_start: Optional[int] = None
    quality_min: int = 50
    preserve_exif: bool = False
    _compiled_pattern: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        if self.path_pattern:
            self._compiled_pattern = re.compile(self.path_pattern, re.IGNORECASE)

    def matches_path(self, file_path: str) -> bool:
        if not self._compiled_pattern:
            return False
        return bool(self._compiled_pattern.search(file_path.replace("\\", "/")))

    def matches_dimension(self, width: int, height: int) -> bool:
        max_dim = max(width, height)
        if self.max_dimension is not None and max_dim <= self.max_dimension:
            return True
        if self.min_dimension is not None and max_dim >= self.min_dimension:
            return True
        return False

    def signature(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"CACHE_VERSION={CACHE_VERSION}".encode())
        hasher.update(f"name={self.name}".encode())
        hasher.update(f"format={self.format}".encode())
        hasher.update(f"quality={self.quality}".encode())
        hasher.update(f"method={self.method}".encode())
        hasher.update(f"target_ratio={self.target_ratio}".encode())
        hasher.update(f"quality_start={self.quality_start}".encode())
        hasher.update(f"quality_min={self.quality_min}".encode())
        hasher.update(f"preserve_exif={self.preserve_exif}".encode())
        hasher.update(f"max_dimension={self.max_dimension}".encode())
        hasher.update(f"min_dimension={self.min_dimension}".encode())
        return hasher.hexdigest()[:16]


@dataclass
class CompressionResult:
    file_path: str
    output_path: str
    strategy_name: str
    original_size: int
    compressed_size: int
    original_dimensions: tuple
    compressed_dimensions: tuple
    compression_ratio: float
    time_elapsed: float
    status: str
    original_mtime: float
    output_mtime: Optional[float] = None
    backup_path: Optional[str] = None
    file_hash: Optional[str] = None
    strategy_signature: Optional[str] = None
    recompress_reason: Optional[str] = None
    error: Optional[str] = None


def compute_file_hash(file_path: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_strategies(config: dict) -> list:
    strategies = []
    for s in config.get("strategies", []):
        strategies.append(Strategy(
            name=s.get("name", "unknown"),
            description=s.get("description", ""),
            path_pattern=s.get("path_pattern"),
            max_dimension=s.get("max_dimension"),
            min_dimension=s.get("min_dimension"),
            format=s.get("format", "original"),
            quality=s.get("quality", 85),
            method=s.get("method", 4),
            target_ratio=s.get("target_ratio"),
            quality_start=s.get("quality_start"),
            quality_min=s.get("quality_min", 50),
            preserve_exif=s.get("preserve_exif", False),
        ))
    return strategies


def resolve_strategy(file_path: str, width: int, height: int,
                     strategies: list) -> Strategy:
    for strategy in strategies:
        if strategy.path_pattern and strategy.matches_path(file_path):
            if strategy.matches_dimension(width, height):
                return strategy

    for strategy in strategies:
        if not strategy.path_pattern:
            if strategy.matches_dimension(width, height):
                return strategy

    max_dim = max(width, height)
    for strategy in strategies:
        if strategy.max_dimension is not None and max_dim <= strategy.max_dimension:
            return strategy
        if strategy.min_dimension is not None and max_dim >= strategy.min_dimension:
            return strategy

    fallback = Strategy(name="fallback", format="original", quality=85,
                        target_ratio=0.70, quality_start=85, quality_min=50,
                        preserve_exif=True)
    return fallback


def extract_exif(img: Image.Image) -> Optional[bytes]:
    try:
        return img.info.get("exif")
    except Exception:
        return None


def get_output_ext(strategy: Strategy, original_ext: str) -> str:
    if strategy.format == "webp":
        return ".webp"
    return original_ext


def compute_output_path(file_path: str, source_root: str, output_dir: Optional[str],
                         in_place: bool, no_in_place: bool,
                         strategy: Strategy, original_ext: str) -> str:
    output_ext = get_output_ext(strategy, original_ext)

    if output_dir:
        rel_path = os.path.relpath(file_path, source_root)
        base, _ = os.path.splitext(rel_path)
        target_path = os.path.join(output_dir, base + output_ext)
        return target_path

    if no_in_place:
        base, _ = os.path.splitext(file_path)
        return base + MIN_SUFFIX + output_ext

    if in_place:
        base, _ = os.path.splitext(file_path)
        return base + output_ext

    return file_path


def is_min_file(file_path: str, supported_ext: list) -> bool:
    name = os.path.basename(file_path)
    for ext in supported_ext:
        if name.endswith(MIN_SUFFIX + ext):
            return True
    return False


def is_own_output_file(file_path: str, source_root: str, output_dir: Optional[str],
                       no_in_place: bool, supported_ext: list) -> bool:
    if output_dir:
        abs_output = os.path.abspath(output_dir)
        abs_file = os.path.abspath(file_path)
        if abs_file.startswith(abs_output + os.sep):
            return True

    if no_in_place and is_min_file(file_path, supported_ext):
        return True

    return False


def compress_to_webp(img: Image.Image, output_path: str, strategy: Strategy,
                     exif_data: Optional[bytes] = None) -> int:
    save_kwargs = {
        "format": "WEBP",
        "quality": strategy.quality,
        "method": strategy.method,
        "lossless": False,
    }
    if exif_data and strategy.preserve_exif:
        try:
            from PIL.ExifTags import Base as ExifBase
            img.save(output_path, **save_kwargs)
            return os.path.getsize(output_path)
        except Exception:
            pass

    img.save(output_path, **save_kwargs)
    return os.path.getsize(output_path)


def compress_with_target_ratio(img: Image.Image, output_path: str,
                                original_size: int, strategy: Strategy,
                                exif_data: Optional[bytes] = None) -> int:
    target_size = int(original_size * strategy.target_ratio)
    quality = strategy.quality_start or strategy.quality
    output_ext = os.path.splitext(output_path)[1].lower()
    output_format = output_ext.lstrip(".")
    if output_format == "jpg":
        output_format = "jpeg"
    output_format = output_format.upper()

    temp_path = output_path + ".tmp"
    best_size = original_size
    best_quality = quality

    while quality >= strategy.quality_min:
        save_kwargs = {
            "format": output_format,
            "quality": quality,
            "optimize": True,
        }
        if output_format == "JPEG":
            save_kwargs["subsampling"] = "4:2:0"
        if exif_data and strategy.preserve_exif:
            save_kwargs["exif"] = exif_data

        save_img = img
        cleanup_rgb = None
        if (output_format == "JPEG") and img.mode in ("RGBA", "P"):
            rgb_img = img.convert("RGB")
            save_img = rgb_img
            cleanup_rgb = rgb_img

        save_img.save(temp_path, **save_kwargs)

        if cleanup_rgb:
            cleanup_rgb.close()

        current_size = os.path.getsize(temp_path)

        if current_size < best_size:
            best_size = current_size
            best_quality = quality
            if current_size <= target_size:
                break

        quality -= 5

    if os.path.exists(temp_path) and best_size < original_size:
        shutil.move(temp_path, output_path)
    else:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        save_kwargs = {
            "format": output_format,
            "quality": best_quality,
            "optimize": True,
        }
        if output_format == "JPEG":
            save_kwargs["subsampling"] = "4:2:0"
        if exif_data and strategy.preserve_exif:
            save_kwargs["exif"] = exif_data

        save_img = img
        cleanup_rgb = None
        if (output_format == "JPEG") and img.mode in ("RGBA", "P"):
            rgb_img = img.convert("RGB")
            save_img = rgb_img
            cleanup_rgb = rgb_img

        save_img.save(output_path, **save_kwargs)
        best_size = os.path.getsize(output_path)

        if cleanup_rgb:
            cleanup_rgb.close()

    return best_size


def backup_original(file_path: str, source_root: str, backup_dir: str) -> str:
    abs_backup = os.path.abspath(backup_dir)
    rel_path = os.path.relpath(file_path, source_root)
    backup_path = os.path.join(abs_backup, rel_path)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(file_path, backup_path)
    return backup_path


def compress_single(file_path: str, source_root: str, strategies: list,
                    output_dir: Optional[str], backup_dir: Optional[str],
                    in_place: bool, no_in_place: bool, keep_source: bool,
                    cache_entry: Optional[dict], file_hash: str,
                    original_mtime: float, original_size: int) -> CompressionResult:
    start_time = time.time()

    try:
        img = Image.open(file_path)
        img.load()
        width, height = img.size
        original_dimensions = (width, height)
    except Exception as e:
        return CompressionResult(
            file_path=file_path,
            output_path="",
            strategy_name="error",
            original_size=original_size,
            compressed_size=original_size,
            original_dimensions=(0, 0),
            compressed_dimensions=(0, 0),
            compression_ratio=1.0,
            time_elapsed=time.time() - start_time,
            status="error",
            original_mtime=original_mtime,
            error=f"无法打开图片: {e}",
        )

    strategy = resolve_strategy(file_path, width, height, strategies)
    strategy_sig = strategy.signature()
    exif_data = extract_exif(img)

    original_ext = os.path.splitext(file_path)[1].lower()
    output_path = compute_output_path(
        file_path, source_root, output_dir, in_place, no_in_place,
        strategy, original_ext
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    recompress_reason = None
    if cache_entry is not None:
        cached_size = cache_entry.get("compressed_size")
        cached_sig = cache_entry.get("strategy_signature")

        if cached_sig != strategy_sig:
            recompress_reason = "config_changed"
        elif not os.path.exists(output_path):
            recompress_reason = "output_missing"
        elif os.path.getsize(output_path) != cached_size:
            recompress_reason = "output_size_changed"
        elif cache_entry.get("file_hash") != file_hash:
            recompress_reason = "source_changed"
        elif cache_entry.get("strategy") != strategy.name:
            recompress_reason = "strategy_mismatch"
        else:
            img.close()
            return CompressionResult(
                file_path=file_path,
                output_path=output_path,
                strategy_name=strategy.name,
                original_size=original_size,
                compressed_size=cached_size,
                original_dimensions=original_dimensions,
                compressed_dimensions=original_dimensions,
                compression_ratio=cached_size / original_size,
                time_elapsed=0,
                status="skipped",
                original_mtime=original_mtime,
                output_mtime=os.path.getmtime(output_path),
                file_hash=file_hash,
                strategy_signature=strategy_sig,
            )

    backup_path = None
    if backup_dir:
        try:
            backup_path = backup_original(file_path, source_root, backup_dir)
        except Exception as e:
            img.close()
            return CompressionResult(
                file_path=file_path,
                output_path="",
                strategy_name="error",
                original_size=original_size,
                compressed_size=original_size,
                original_dimensions=original_dimensions,
                compressed_dimensions=original_dimensions,
                compression_ratio=1.0,
                time_elapsed=time.time() - start_time,
                status="error",
                original_mtime=original_mtime,
                error=f"备份失败: {e}",
            )

    try:
        if strategy.format == "webp":
            compressed_size = compress_to_webp(img, output_path, strategy, exif_data)
        else:
            compressed_size = compress_with_target_ratio(
                img, output_path, original_size, strategy, exif_data
            )
    except Exception as e:
        img.close()
        return CompressionResult(
            file_path=file_path,
            output_path=output_path,
            strategy_name=strategy.name,
            original_size=original_size,
            compressed_size=original_size,
            original_dimensions=original_dimensions,
            compressed_dimensions=original_dimensions,
            compression_ratio=1.0,
            time_elapsed=time.time() - start_time,
            status="error",
            original_mtime=original_mtime,
            error=f"压缩失败: {e}",
        )

    img.close()

    try:
        with Image.open(output_path) as new_img:
            compressed_dimensions = new_img.size
    except Exception:
        compressed_dimensions = original_dimensions

    if in_place and not no_in_place:
        if os.path.exists(file_path) and os.path.abspath(file_path) != os.path.abspath(output_path):
            if not keep_source:
                os.remove(file_path)

    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    time_elapsed = time.time() - start_time

    return CompressionResult(
        file_path=file_path,
        output_path=output_path,
        strategy_name=strategy.name,
        original_size=original_size,
        compressed_size=compressed_size,
        original_dimensions=original_dimensions,
        compressed_dimensions=compressed_dimensions,
        compression_ratio=compression_ratio,
        time_elapsed=time_elapsed,
        status="success",
        original_mtime=original_mtime,
        output_mtime=os.path.getmtime(output_path),
        backup_path=backup_path,
        file_hash=file_hash,
        strategy_signature=strategy_sig,
        recompress_reason=recompress_reason,
    )


def load_cache(cache_path: str) -> dict:
    if not os.path.isfile(cache_path):
        return {"version": CACHE_VERSION, "files": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("version") != CACHE_VERSION:
                return {"version": CACHE_VERSION, "files": {}}
            return data
    except (json.JSONDecodeError, IOError):
        return {"version": CACHE_VERSION, "files": {}}


def save_cache(cache_path: str, cache_data: dict):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def clean_stale_products(source_root: str, output_dir: Optional[str],
                         current_source_files: list, cache: dict,
                         supported_ext: list) -> dict:
    """清理过期产物：源文件已删除/改名时，删除对应的输出文件和缓存条目"""
    if not output_dir:
        return cache

    abs_source_root = os.path.abspath(source_root)
    abs_output_dir = os.path.abspath(output_dir)

    valid_rel_sources = set()
    for src in current_source_files:
        abs_src = os.path.abspath(src)
        if abs_src.startswith(abs_source_root + os.sep):
            rel = os.path.relpath(abs_src, abs_source_root)
            base, _ = os.path.splitext(rel)
            valid_rel_sources.add(base)

    stale_outputs = []
    for root, _, files in os.walk(abs_output_dir):
        for f in files:
            if f == ".compress_cache.json" or f == "manifest.json":
                continue
            full_path = os.path.join(root, f)
            abs_full = os.path.abspath(full_path)
            rel_path = os.path.relpath(abs_full, abs_output_dir)
            base, _ = os.path.splitext(rel_path)
            if base not in valid_rel_sources:
                stale_outputs.append(full_path)

    for stale in stale_outputs:
        try:
            os.remove(stale)
            print(f"  [清理] 过期产物: {stale}")
        except Exception as e:
            print(f"  [警告] 无法删除过期产物 {stale}: {e}")

    valid_abs_sources = {os.path.abspath(s) for s in current_source_files}
    stale_cache_keys = []
    for cache_key in cache.get("files", {}):
        abs_key = os.path.abspath(cache_key)
        if abs_key not in valid_abs_sources:
            stale_cache_keys.append(cache_key)

    for key in stale_cache_keys:
        del cache["files"][key]
        print(f"  [清理] 过期缓存: {key}")

    return cache


def process_file_wrapper(args):
    (file_path, source_root, strategies_data, output_dir, backup_dir,
     in_place, no_in_place, keep_source, cache_path, enable_cache) = args

    strategies = []
    for s in strategies_data:
        strategies.append(Strategy(
            name=s["name"],
            description=s.get("description", ""),
            path_pattern=s.get("path_pattern"),
            max_dimension=s.get("max_dimension"),
            min_dimension=s.get("min_dimension"),
            format=s.get("format", "original"),
            quality=s.get("quality", 85),
            method=s.get("method", 4),
            target_ratio=s.get("target_ratio"),
            quality_start=s.get("quality_start"),
            quality_min=s.get("quality_min", 50),
            preserve_exif=s.get("preserve_exif", False),
        ))

    original_mtime = os.path.getmtime(file_path)
    original_size = os.path.getsize(file_path)
    file_hash = compute_file_hash(file_path)

    cache_entry = None
    if enable_cache:
        cache = load_cache(cache_path)
        cache_entry = cache.get("files", {}).get(file_path)

    return compress_single(
        file_path, source_root, strategies, output_dir, backup_dir,
        in_place, no_in_place, keep_source, cache_entry,
        file_hash, original_mtime, original_size
    )


def strategy_to_dict(strategy: Strategy) -> dict:
    return {
        "name": strategy.name,
        "description": strategy.description,
        "path_pattern": strategy.path_pattern,
        "max_dimension": strategy.max_dimension,
        "min_dimension": strategy.min_dimension,
        "format": strategy.format,
        "quality": strategy.quality,
        "method": strategy.method,
        "target_ratio": strategy.target_ratio,
        "quality_start": strategy.quality_start,
        "quality_min": strategy.quality_min,
        "preserve_exif": strategy.preserve_exif,
        "signature": strategy.signature(),
    }


def scan_images(directory: str, extensions: list, skip_own_output: bool = False,
                source_root: Optional[str] = None, output_dir: Optional[str] = None,
                no_in_place: bool = False) -> list:
    images = []
    abs_source_root = os.path.abspath(directory) if source_root is None else os.path.abspath(source_root)

    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in extensions:
                continue
            full_path = os.path.join(root, f)

            if skip_own_output and is_own_output_file(
                full_path, abs_source_root, output_dir, no_in_place, extensions
            ):
                continue

            images.append(full_path)

    return sorted(images)


def format_size(size_bytes: int) -> str:
    if size_bytes < 0:
        return "0 B"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_report(results: list):
    print("\n" + "=" * 160)
    print("  图片压缩对比报告")
    print("=" * 160)

    header = (f"{'源文件':<36} {'输出文件':<36} {'策略':<12} "
              f"{'原始':>10} {'压缩后':>10} {'比例':>8} {'耗时':>8} {'原因':<16} {'状态':<10}")
    print(header)
    print("-" * 160)

    total_original = 0
    total_compressed = 0
    total_time = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    config_changed_count = 0
    source_changed_count = 0

    for r in results:
        short_src = r.file_path
        if len(short_src) > 34:
            short_src = "..." + short_src[-31:]
        short_out = r.output_path
        if len(short_out) > 34:
            short_out = "..." + short_out[-31:]
        ratio_str = f"{r.compression_ratio:.1%}" if r.original_size > 0 else "N/A"
        time_str = f"{r.time_elapsed:.2f}s" if r.time_elapsed > 0 else "-"
        status_display = {
            "success": "成功",
            "skipped": "跳过",
            "error": "失败",
        }.get(r.status, r.status)

        reason = ""
        if r.recompress_reason == "config_changed":
            reason = "配置变更"
            config_changed_count += 1
        elif r.recompress_reason == "source_changed":
            reason = "源图变更"
            source_changed_count += 1
        elif r.recompress_reason == "output_missing":
            reason = "产物缺失"
        elif r.recompress_reason == "output_size_changed":
            reason = "产物大小不符"
        elif r.recompress_reason == "strategy_mismatch":
            reason = "策略变更"

        print(
            f"{short_src:<36} {short_out:<36} {r.strategy_name:<12} "
            f"{format_size(r.original_size):>10} "
            f"{format_size(r.compressed_size):>10} "
            f"{ratio_str:>8} {time_str:>8} {reason:<16} {status_display:<10}"
        )

        if r.status == "success":
            total_original += r.original_size
            total_compressed += r.compressed_size
            total_time += r.time_elapsed
            success_count += 1
        elif r.status == "skipped":
            total_original += r.original_size
            total_compressed += r.compressed_size
            skipped_count += 1
        else:
            error_count += 1

    print("-" * 160)
    total_files = len(results)
    if total_files > 0:
        processed_original = sum(
            r.original_size for r in results if r.status != "error"
        )
        processed_compressed = sum(
            r.compressed_size for r in results if r.status != "error"
        )
        if processed_original > 0:
            overall_ratio = processed_compressed / processed_original
        else:
            overall_ratio = 0
        saved = processed_original - processed_compressed
        print(f"\n  总计: {total_files} 张 | 成功 {success_count} | 跳过 {skipped_count} | 失败 {error_count}")
        if config_changed_count > 0 or source_changed_count > 0:
            print(f"  重新压缩原因: 配置变更 {config_changed_count} | 源图变更 {source_changed_count}")
        print(f"  原始总体积: {format_size(processed_original)}")
        print(f"  压缩后总体积: {format_size(processed_compressed)}")
        print(f"  总压缩比: {overall_ratio:.1%}")
        print(f"  节省空间: {format_size(saved)}")
        print(f"  实际处理耗时: {total_time:.2f}s")
    print("=" * 160)


def aggregate_by_strategy(results: list) -> dict:
    agg = {}
    for r in results:
        s = r.strategy_name
        if s not in agg:
            agg[s] = {
                "count": 0, "success": 0, "skipped": 0, "error": 0,
                "original_size": 0, "compressed_size": 0,
                "max_time": 0, "slowest_file": None,
                "errors": [],
            }
        agg[s]["count"] += 1
        if r.status == "success":
            agg[s]["success"] += 1
            agg[s]["original_size"] += r.original_size
            agg[s]["compressed_size"] += r.compressed_size
            if r.time_elapsed > agg[s]["max_time"]:
                agg[s]["max_time"] = r.time_elapsed
                agg[s]["slowest_file"] = r.file_path
        elif r.status == "skipped":
            agg[s]["skipped"] += 1
            agg[s]["original_size"] += r.original_size
            agg[s]["compressed_size"] += r.compressed_size
        elif r.status == "error":
            agg[s]["error"] += 1
            if r.error:
                agg[s]["errors"].append({"file": r.file_path, "error": r.error})

    for s in agg:
        orig = agg[s]["original_size"]
        comp = agg[s]["compressed_size"]
        agg[s]["saved"] = orig - comp
        agg[s]["ratio"] = (comp / orig) if orig > 0 else 1.0

    return agg


def save_json_report(results: list, report_path: str, strategies: list):
    by_strategy = aggregate_by_strategy(results)
    processed = [r for r in results if r.status != "error"]
    errors = [r for r in results if r.status == "error"]

    slowest = sorted(
        processed, key=lambda r: r.time_elapsed, reverse=True
    )[:5] if processed else []

    total_original = sum(r.original_size for r in processed)
    total_compressed = sum(r.compressed_size for r in processed)
    total_saved = total_original - total_compressed
    total_time_elapsed = sum(r.time_elapsed for r in processed)

    overall_ratio = 0
    if total_original > 0:
        overall_ratio = round(total_compressed / total_original, 4)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_files": len(results),
            "success": sum(1 for r in results if r.status == "success"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "error": sum(1 for r in results if r.status == "error"),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "total_saved": total_saved,
            "total_time_elapsed": total_time_elapsed,
            "overall_ratio": overall_ratio,
            "recompress_reasons": {
                "config_changed": sum(1 for r in results if r.recompress_reason == "config_changed"),
                "source_changed": sum(1 for r in results if r.recompress_reason == "source_changed"),
                "output_missing": sum(1 for r in results if r.recompress_reason == "output_missing"),
            }
        },
        "by_strategy": by_strategy,
        "slowest_files": [
            {
                "file": r.file_path,
                "strategy": r.strategy_name,
                "time_elapsed": round(r.time_elapsed, 3),
            }
            for r in slowest
        ],
        "errors": [
            {"file": r.file_path, "error": r.error, "strategy": r.strategy_name}
            for r in errors
        ],
        "strategies_used": [strategy_to_dict(s) for s in strategies],
        "details": [],
    }

    for r in results:
        report_data["details"].append({
            "file": r.file_path,
            "output": r.output_path,
            "backup": r.backup_path,
            "strategy": r.strategy_name,
            "original_size": r.original_size,
            "compressed_size": r.compressed_size,
            "original_dimensions": list(r.original_dimensions),
            "compressed_dimensions": list(r.compressed_dimensions),
            "compression_ratio": round(r.compression_ratio, 4) if r.original_size > 0 else 0,
            "time_elapsed": round(r.time_elapsed, 3),
            "status": r.status,
            "original_mtime": r.original_mtime,
            "output_mtime": r.output_mtime,
            "file_hash": r.file_hash,
            "strategy_signature": r.strategy_signature,
            "recompress_reason": r.recompress_reason,
            "error": r.error,
        })

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)


def save_csv_report(results: list, report_path: str):
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "源文件", "输出文件", "备份路径", "策略", "原始大小(字节)", "压缩后大小(字节)",
            "原始尺寸", "压缩后尺寸", "压缩比", "耗时(秒)", "状态",
            "重新压缩原因", "源文件哈希", "错误",
        ])
        for r in results:
            writer.writerow([
                r.file_path,
                r.output_path,
                r.backup_path or "",
                r.strategy_name,
                r.original_size,
                r.compressed_size,
                f"{r.original_dimensions[0]}x{r.original_dimensions[1]}",
                f"{r.compressed_dimensions[0]}x{r.compressed_dimensions[1]}",
                f"{r.compression_ratio:.4f}" if r.original_size > 0 else "",
                f"{r.time_elapsed:.3f}",
                r.status,
                r.recompress_reason or "",
                r.file_hash or "",
                r.error or "",
            ])


def save_html_report(results: list, report_path: str, strategies: list):
    by_strategy = aggregate_by_strategy(results)
    processed = [r for r in results if r.status != "error"]
    errors = [r for r in results if r.status == "error"]
    slowest = sorted(processed, key=lambda r: r.time_elapsed, reverse=True)[:10]

    total_original = sum(r.original_size for r in processed)
    total_compressed = sum(r.compressed_size for r in processed)
    total_saved = total_original - total_compressed
    total_success = sum(1 for r in results if r.status == "success")
    total_skipped = sum(1 for r in results if r.status == "skipped")
    total_error = sum(1 for r in results if r.status == "error")

    overall_ratio = 0
    if total_original > 0:
        overall_ratio = total_compressed / total_original

    config_changed = sum(1 for r in results if r.recompress_reason == "config_changed")
    source_changed = sum(1 for r in results if r.recompress_reason == "source_changed")

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           margin: 20px; color: #333; background: #fafafa; }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    .summary { display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }
    .card { background: white; padding: 15px 25px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 140px; }
    .card .label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
    .card .value { font-size: 24px; font-weight: bold; color: #2c3e50; margin-top: 5px; }
    .card.success .value { color: #27ae60; }
    .card.skipped .value { color: #f39c12; }
    .card.error .value { color: #e74c3c; }
    .card.saved .value { color: #27ae60; }
    table { width: 100%; border-collapse: collapse; background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #ecf0f1; }
    th { background: #34495e; color: white; font-weight: 600; }
    tr:nth-child(even) { background: #f8f9fa; }
    tr:hover { background: #eaf2f8; }
    .status-success { color: #27ae60; font-weight: 600; }
    .status-skipped { color: #f39c12; font-weight: 600; }
    .status-error { color: #e74c3c; font-weight: 600; }
    .ratio-good { color: #27ae60; font-weight: 600; }
    .ratio-ok { color: #f39c12; font-weight: 600; }
    .ratio-bad { color: #e74c3c; font-weight: 600; }
    code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px;
           font-family: "SF Mono", Monaco, monospace; font-size: 12px; }
    .strategy-meta { background: white; padding: 15px; border-radius: 8px;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }
    .error-box { background: #fdf2f2; border-left: 4px solid #e74c3c;
                 padding: 12px 16px; margin: 8px 0; border-radius: 0 4px 4px 0; }
    .reason-badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
                    font-size: 11px; font-weight: 600; }
    .reason-config { background: #e8f4fd; color: #2980b9; }
    .reason-source { background: #fff4e5; color: #d35400; }
    """

    html_content = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head><meta charset='UTF-8'>",
        "<title>图片压缩报告</title>",
        f"<style>{css}</style></head><body>",
        f"<h1>📊 图片压缩对比报告</h1>",
        f"<p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",

        "<div class='summary'>",
        f"<div class='card'><div class='label'>总文件数</div>"
        f"<div class='value'>{len(results)}</div></div>",
        f"<div class='card success'><div class='label'>成功</div>"
        f"<div class='value'>{total_success}</div></div>",
        f"<div class='card skipped'><div class='label'>跳过</div>"
        f"<div class='value'>{total_skipped}</div></div>",
        f"<div class='card error'><div class='label'>失败</div>"
        f"<div class='value'>{total_error}</div></div>",
        f"<div class='card'><div class='label'>原始体积</div>"
        f"<div class='value'>{format_size(total_original)}</div></div>",
        f"<div class='card'><div class='label'>压缩后体积</div>"
        f"<div class='value'>{format_size(total_compressed)}</div></div>",
        f"<div class='card saved'><div class='label'>节省空间</div>"
        f"<div class='value'>{format_size(total_saved)}</div></div>",
        f"<div class='card'><div class='label'>压缩比</div>"
        f"<div class='value'>{overall_ratio * 100:.1f}%</div></div>",
        "</div>",
    ]

    if config_changed > 0 or source_changed > 0:
        html_content.append("<div class='summary'>")
        if config_changed > 0:
            html_content.append(
                f"<div class='card'><div class='label'>配置变更重压缩</div>"
                f"<div class='value reason-badge reason-config'>{config_changed}</div></div>"
            )
        if source_changed > 0:
            html_content.append(
                f"<div class='card'><div class='label'>源图变更重压缩</div>"
                f"<div class='value reason-badge reason-source'>{source_changed}</div></div>"
            )
        html_content.append("</div>")

    if errors:
        html_content.append("<h2>❌ 错误明细</h2>")
        for r in errors:
            html_content.append(
                f"<div class='error-box'>"
                f"<strong><code>{html.escape(r.file_path)}</code></strong> "
                f"[<code>{html.escape(r.strategy_name)}</code>]<br>"
                f"<span style='color:#c0392b;'>{html.escape(str(r.error))}</span>"
                f"</div>"
            )

    html_content.append("<h2>📈 按策略汇总</h2>")
    html_content.append(
        "<table><thead><tr>"
        "<th>策略</th><th>总数</th><th>成功</th><th>跳过</th><th>失败</th>"
        "<th>原始体积</th><th>压缩后</th><th>节省</th><th>压缩比</th>"
        "<th>最耗时文件</th><th>耗时</th>"
        "</tr></thead><tbody>"
    )

    for sname, sdata in sorted(by_strategy.items()):
        ratio_cls = "ratio-good" if sdata["ratio"] <= 0.7 else (
            "ratio-ok" if sdata["ratio"] <= 0.85 else "ratio-bad"
        )
        slowest_short = sdata["slowest_file"] or "-"
        if len(slowest_short) > 40:
            slowest_short = "..." + slowest_short[-37:]
        html_content.append(
            f"<tr><td><code>{sname}</code></td>"
            f"<td>{sdata['count']}</td>"
            f"<td>{sdata['success']}</td>"
            f"<td>{sdata['skipped']}</td>"
            f"<td>{sdata['error']}</td>"
            f"<td>{format_size(sdata['original_size'])}</td>"
            f"<td>{format_size(sdata['compressed_size'])}</td>"
            f"<td>{format_size(sdata['saved'])}</td>"
            f"<td class='{ratio_cls}'>{sdata['ratio']:.1%}</td>"
            f"<td><code>{html.escape(slowest_short)}</code></td>"
            f"<td>{sdata['max_time']:.2f}s</td></tr>"
        )

    html_content.append("</tbody></table>")

    if slowest:
        html_content.append("<h2>🐢 最耗时的 10 个文件</h2>")
        html_content.append(
            "<table><thead><tr>"
            "<th>排名</th><th>文件</th><th>策略</th><th>耗时</th>"
            "</tr></thead><tbody>"
        )
        for i, r in enumerate(slowest, 1):
            p = r.file_path
            if len(p) > 60:
                p = "..." + p[-57:]
            html_content.append(
                f"<tr><td>{i}</td><td><code>{html.escape(p)}</code></td>"
                f"<td><code>{r.strategy_name}</code></td>"
                f"<td>{r.time_elapsed:.3f}s</td></tr>"
            )
        html_content.append("</tbody></table>")

    html_content.append("<h2>📋 策略配置</h2>")
    for s in strategies:
        html_content.append(
            f"<div class='strategy-meta'><h3><code>{s.name}</code> "
            f"<small>{html.escape(s.description)}</small></h3>"
            f"<ul>"
        )
        if s.path_pattern:
            html_content.append(
                f"<li>路径正则: <code>{html.escape(s.path_pattern)}</code></li>"
            )
        if s.max_dimension is not None:
            html_content.append(f"<li>最大尺寸: {s.max_dimension}px</li>")
        if s.min_dimension is not None:
            html_content.append(f"<li>最小尺寸: {s.min_dimension}px</li>")
        html_content.append(f"<li>输出格式: {s.format}</li>")
        html_content.append(f"<li>质量: {s.quality}</li>")
        if s.target_ratio:
            html_content.append(f"<li>目标压缩比: {s.target_ratio:.0%}</li>")
        html_content.append(f"<li>保留 EXIF: {'是' if s.preserve_exif else '否'}</li>")
        html_content.append(f"<li>策略签名: <code>{s.signature()}</code></li>")
        html_content.append("</ul></div>")

    html_content.append("<h2>📝 详细列表</h2>")
    html_content.append(
        "<table><thead><tr>"
        "<th>源文件</th><th>输出文件</th><th>策略</th>"
        "<th>原始</th><th>压缩后</th><th>压缩比</th><th>耗时</th>"
        "<th>原因</th><th>状态</th>"
        "</tr></thead><tbody>"
    )

    for r in results:
        status_cls = f"status-{r.status}"
        status_text = {"success": "成功", "skipped": "跳过", "error": "失败"}.get(
            r.status, r.status
        )
        ratio_cls = "ratio-good" if r.compression_ratio <= 0.7 else (
            "ratio-ok" if r.compression_ratio <= 0.85 else "ratio-bad"
        ) if r.original_size > 0 else ""

        src = r.file_path
        if len(src) > 45:
            src = "..." + src[-42:]
        out = r.output_path or "-"
        if len(out) > 45:
            out = "..." + out[-42:]

        reason = ""
        if r.recompress_reason == "config_changed":
            reason = "<span class='reason-badge reason-config'>配置变更</span>"
        elif r.recompress_reason == "source_changed":
            reason = "<span class='reason-badge reason-source'>源图变更</span>"
        elif r.recompress_reason:
            reason = f"<code>{r.recompress_reason}</code>"

        ratio_str = f"{r.compression_ratio:.1%}" if r.original_size > 0 else "N/A"

        html_content.append(
            f"<tr><td><code>{html.escape(src)}</code></td>"
            f"<td><code>{html.escape(out)}</code></td>"
            f"<td><code>{r.strategy_name}</code></td>"
            f"<td>{format_size(r.original_size)}</td>"
            f"<td>{format_size(r.compressed_size)}</td>"
            f"<td class='{ratio_cls}'>{ratio_str}</td>"
            f"<td>{r.time_elapsed:.3f}s</td>"
            f"<td>{reason}</td>"
            f"<td class='{status_cls}'>{status_text}</td></tr>"
        )

    html_content.append("</tbody></table></body></html>")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))


def save_manifest(results: list, manifest_path: str, source_root: str,
                   output_dir: Optional[str]):
    source_root_abs = os.path.abspath(source_root)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0",
        "source_root": source_root_abs,
        "output_root": os.path.abspath(output_dir) if output_dir else None,
        "source_to_output": {},
        "output_to_source": {},
        "items": [],
    }

    for r in results:
        if r.status == "error":
            continue

        src_abs = os.path.abspath(r.file_path)
        out_abs = os.path.abspath(r.output_path)

        try:
            src_rel = os.path.relpath(src_abs, source_root_abs)
        except ValueError:
            src_rel = src_abs

        if output_dir:
            try:
                out_rel = os.path.relpath(out_abs, os.path.abspath(output_dir))
            except ValueError:
                out_rel = os.path.basename(out_abs)
        else:
            try:
                out_rel = os.path.relpath(out_abs, source_root_abs)
            except ValueError:
                out_rel = os.path.basename(out_abs)

        item = {
            "source": src_abs,
            "source_rel": src_rel.replace("\\", "/"),
            "output": out_abs,
            "output_rel": out_rel.replace("\\", "/"),
            "strategy": r.strategy_name,
            "file_hash": r.file_hash,
            "original_size": r.original_size,
            "compressed_size": r.compressed_size,
            "original_dimensions": list(r.original_dimensions),
            "compressed_dimensions": list(r.compressed_dimensions),
            "compression_ratio": round(r.compression_ratio, 4),
            "status": r.status,
            "skipped": r.status == "skipped",
        }

        manifest["items"].append(item)
        manifest["source_to_output"][src_rel.replace("\\", "/")] = out_rel.replace("\\", "/")
        manifest["output_to_source"][out_rel.replace("\\", "/")] = src_rel.replace("\\", "/")

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_path: str) -> dict:
    if not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def scan_text_files(paths: list, extensions: Optional[list] = None) -> list:
    """扫描要做引用替换的文件：支持目录/文件列表，默认 .html/.css/.js"""
    if extensions is None:
        extensions = [".html", ".htm", ".css", ".js"]
    exts_lower = {e.lower() for e in extensions}
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in exts_lower:
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(p):
            files.append(p)
    return sorted(files)


def replace_references_in_content(content: str, mapping: dict,
                                   manifest_source_root: str,
                                   manifest_output_root: Optional[str],
                                   text_file_path: str) -> tuple:
    """
    用 manifest 的 source_to_output 映射替换文本内容中的图片路径。
    匹配策略（按优先级）：
      1) 引号包裹的完整相对路径/绝对路径能匹配到 source_rel 或 source_abs
      2) 去掉 ../ 前缀后能匹配
    返回 (new_content, list_of_changes)
    """
    changes = []
    text_dir_abs = os.path.abspath(os.path.dirname(text_file_path))

    manifest_source_root_abs = os.path.abspath(manifest_source_root)

    def try_replace(match):
        quote = match.group(1)
        raw_path = match.group(2)
        if raw_path.startswith(("data:", "#", "http://", "https://")):
            return match.group(0)

        candidates = [raw_path]

        norm = raw_path.replace("\\", "/")
        stripped = re.sub(r"^(\.\./)+", "", norm)
        if stripped != norm:
            candidates.append(stripped)

        abs_candidates = []
        try:
            abs_p = os.path.abspath(os.path.join(text_dir_abs, raw_path))
            rel_from_src = os.path.relpath(abs_p, manifest_source_root_abs)
            rel_from_src_norm = rel_from_src.replace("\\", "/")
            if not rel_from_src_norm.startswith(".."):
                abs_candidates.append(rel_from_src_norm)
        except ValueError:
            pass

        all_keys = candidates + abs_candidates

        for key in all_keys:
            if key in mapping:
                new_rel = mapping[key]
                if manifest_output_root:
                    try:
                        out_abs = os.path.join(os.path.abspath(manifest_output_root), new_rel)
                        final = os.path.relpath(out_abs, text_dir_abs).replace("\\", "/")
                    except ValueError:
                        final = new_rel
                else:
                    try:
                        src_abs = os.path.join(manifest_source_root_abs, key)
                        out_abs = os.path.abspath(os.path.join(os.path.dirname(src_abs), new_rel))
                        final = os.path.relpath(out_abs, text_dir_abs).replace("\\", "/")
                    except Exception:
                        final = new_rel
                if final != raw_path:
                    changes.append({"from": raw_path, "to": final, "key": key})
                return f"{quote}{final}{quote}"

        return match.group(0)

    pattern = re.compile(r"""(['"])([^'"]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff?|svg))\1""", re.IGNORECASE)
    new_content = pattern.sub(try_replace, content)

    url_pattern = re.compile(r"""(url\(\s*)(['"]?)([^'")]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff?|svg))\2(\s*\))""", re.IGNORECASE)

    def try_replace_url(match):
        prefix = match.group(1)
        quote = match.group(2) or ""
        raw_path = match.group(3)
        suffix = match.group(4) or ""
        if raw_path.startswith(("data:", "#", "http://", "https://")):
            return match.group(0)

        class FakeMatch:
            def group(self, i):
                if i == 0:
                    return f"{quote}{raw_path}{quote}"
                return [quote, raw_path][i - 1]

        replacement = try_replace(FakeMatch())
        if quote:
            new_inner = replacement[len(quote):-len(quote)]
        else:
            new_inner = replacement
        return f"{prefix}{quote}{new_inner}{quote}{suffix}"

    new_content = url_pattern.sub(try_replace_url, new_content)
    return new_content, changes


def replace_references_in_files(manifest_path: str, target_paths: list,
                                 preview: bool = False) -> dict:
    """读取 manifest，替换目标文件中的图片引用。preview=True 只打印差异不写回。"""
    manifest = load_manifest(manifest_path)
    if not manifest:
        return {"error": f"manifest 文件不存在或无法读取: {manifest_path}"}

    mapping = manifest.get("source_to_output", {})
    src_root = manifest.get("source_root", "")
    out_root = manifest.get("output_root")

    if not mapping:
        return {"warning": "manifest 中没有路径映射（source_to_output 为空）", "files": []}

    text_files = scan_text_files(target_paths)
    summary = {"manifest": manifest_path, "total_files": len(text_files),
               "changed_files": 0, "total_replacements": 0, "files": []}

    for tf in text_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                original = f.read()
        except (IOError, UnicodeDecodeError) as e:
            summary["files"].append({"file": tf, "error": f"读取失败: {e}"})
            continue

        new_content, changes = replace_references_in_content(
            original, mapping, src_root, out_root, tf
        )

        entry = {"file": tf, "replacements": len(changes), "changes": changes}
        summary["total_replacements"] += len(changes)

        if len(changes) == 0:
            summary["files"].append(entry)
            continue

        summary["changed_files"] += 1
        if preview:
            diff_lines = list(unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{os.path.relpath(tf)}",
                tofile=f"b/{os.path.relpath(tf)}",
                lineterm="",
            ))
            entry["diff"] = diff_lines
        else:
            try:
                with open(tf, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except IOError as e:
                entry["error"] = f"写入失败: {e}"
        summary["files"].append(entry)

    return summary


def print_replace_summary(summary: dict, preview: bool):
    if "error" in summary:
        print(f"[引用替换] 错误: {summary['error']}")
        return
    if "warning" in summary:
        print(f"[引用替换] 跳过: {summary['warning']}")
        return

    mode = "预览" if preview else "已写入"
    print(f"\n[引用替换 {mode}] 扫描 {summary['total_files']} 个文件，"
          f"修改 {summary['changed_files']} 个文件，共 {summary['total_replacements']} 处替换")

    for entry in summary["files"]:
        if entry.get("replacements", 0) == 0 and "error" not in entry:
            continue
        print(f"  - {entry['file']}: ", end="")
        if "error" in entry:
            print(f"失败 - {entry['error']}")
            continue
        if entry["replacements"] == 0:
            print("无替换")
            continue
        print(f"{entry['replacements']} 处替换")
        for c in entry["changes"]:
            print(f"    '{c['from']}' -> '{c['to']}'")
        if preview and "diff" in entry:
            print("\n".join(entry["diff"]))
            print()


def find_all_image_references(content: str) -> list:
    """从文本中提取所有图片路径引用（HTML img src、CSS url()、引号包裹路径）"""
    refs = []
    pattern = re.compile(
        r"""(?:(['"])([^'"]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff?|svg))\1)"""
        r"""|(?:url\(\s*)(['"]?)([^'")]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff?|svg))\3(\s*\))""",
        re.IGNORECASE,
    )
    for m in pattern.finditer(content):
        if m.group(1):
            raw = m.group(2)
        else:
            raw = m.group(4)
        refs.append(raw)
    return refs


def validate_references(manifest_path: str, target_paths: list,
                         source_roots: Optional[list] = None) -> dict:
    """
    校验 HTML/CSS/JS 中图片引用：
    - stale_source: 仍指向源图目录（manifest 里存在这个源路径，说明没替换）
    - broken_link:  指向的文件磁盘上不存在
    返回 {issues, summary}
    """
    manifest = load_manifest(manifest_path)
    if not manifest:
        return {"error": f"manifest 文件不存在或无法读取: {manifest_path}",
                "issues": [], "summary": {}}

    mapping = manifest.get("source_to_output", {})
    src_root = manifest.get("source_root", "")
    out_root = manifest.get("output_root")
    all_source_keys = set(mapping.keys())

    if source_roots is None:
        source_roots = [src_root] if src_root else []
    source_roots_abs = [os.path.abspath(r) for r in source_roots if r]

    text_files = scan_text_files(target_paths)
    issues = []
    total_refs = 0
    external_count = 0

    for tf in text_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, UnicodeDecodeError) as e:
            issues.append(ReferenceIssue(
                file_path=tf, issue_type="read_error", raw_path="",
                details=f"读取失败: {e}",
            ))
            continue

        refs = find_all_image_references(content)
        text_dir_abs = os.path.abspath(os.path.dirname(tf))

        for raw in refs:
            total_refs += 1

            if raw.startswith(("data:", "#", "http://", "https://")):
                external_count += 1
                continue

            candidates = [raw]
            norm = raw.replace("\\", "/")
            stripped = re.sub(r"^(\.\./)+", "", norm)
            if stripped != norm:
                candidates.append(stripped)

            is_stale = False
            for cand in candidates:
                if cand in all_source_keys:
                    is_stale = True
                    break
                for sra in source_roots_abs:
                    try:
                        abs_p = os.path.abspath(os.path.join(text_dir_abs, raw))
                        rel = os.path.relpath(abs_p, sra).replace("\\", "/")
                        if not rel.startswith("..") and rel in all_source_keys:
                            is_stale = True
                            break
                    except ValueError:
                        pass
                if is_stale:
                    break

            if is_stale:
                issues.append(ReferenceIssue(
                    file_path=tf, issue_type="stale_source", raw_path=raw,
                    details="路径仍指向源图目录（应替换为压缩产物）",
                ))
                continue

            resolved = None
            try:
                resolved = os.path.abspath(os.path.join(text_dir_abs, raw))
            except ValueError:
                pass

            if resolved and not os.path.isfile(resolved):
                issues.append(ReferenceIssue(
                    file_path=tf, issue_type="broken_link", raw_path=raw,
                    resolved_path=resolved,
                    details=f"解析为绝对路径后不存在: {resolved}",
                ))

    stale_count = sum(1 for i in issues if i.issue_type == "stale_source")
    broken_count = sum(1 for i in issues if i.issue_type == "broken_link")
    read_errors = sum(1 for i in issues if i.issue_type == "read_error")

    summary = {
        "files_scanned": len(text_files),
        "total_refs": total_refs,
        "external_refs": external_count,
        "stale_source_refs": stale_count,
        "broken_links": broken_count,
        "read_errors": read_errors,
        "passed": stale_count == 0 and broken_count == 0,
    }
    return {"issues": issues, "summary": summary}


def print_validate_summary(result: dict):
    if "error" in result:
        print(f"\n[引用校验] 错误: {result['error']}")
        return
    s = result["summary"]
    issues = result["issues"]
    tag = "通过" if s["passed"] else "失败"
    print(f"\n[引用校验 {tag}] 扫描 {s['files_scanned']} 个文件，"
          f"共 {s['total_refs']} 处引用（外部 {s['external_refs']}）")
    print(f"  未替换旧路径(stale): {s['stale_source_refs']} | "
          f"断链(broken_link): {s['broken_links']} | 读取失败: {s['read_errors']}")
    for iss in issues[:20]:
        if iss.issue_type == "read_error":
            print(f"  [读取失败] {iss.file_path}: {iss.details}")
        elif iss.issue_type == "stale_source":
            print(f"  [未替换] {iss.file_path}: '{iss.raw_path}'  {iss.details}")
        elif iss.issue_type == "broken_link":
            print(f"  [断链] {iss.file_path}: '{iss.raw_path}' -> {iss.resolved_path}")
    if len(issues) > 20:
        print(f"  ... 其余 {len(issues) - 20} 条已省略")


def parse_budgets(config: dict) -> list:
    """从配置中解析体积预算规则"""
    budgets = []
    raw_list = config.get("budgets", [])
    for rb in raw_list:
        name = rb.get("name", "unnamed")
        if "max_bytes" in rb:
            raw_val = rb["max_bytes"]
        elif "max_kb" in rb:
            raw_val = str(rb["max_kb"]) + "KB"
        elif "max_mb" in rb:
            raw_val = str(rb["max_mb"]) + "MB"
        else:
            raw_val = 0
        max_bytes_str = str(raw_val)
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(kb|mb|gb|b)?\s*$", max_bytes_str, re.IGNORECASE)
        if m:
            num = float(m.group(1))
            unit = (m.group(2) or "b").lower()
            mult = {"b": 1, "kb": 1024, "mb": 1024 * 1024, "gb": 1024 ** 3}.get(unit, 1)
            max_bytes = int(num * mult)
        else:
            max_bytes = int(raw_val) if isinstance(raw_val, (int, float)) else 0
        scope_type = rb.get("scope", "directory")
        scope_value = rb.get("value", rb.get("directory", rb.get("strategy", "")))
        budgets.append(BudgetRule(
            name=name, max_bytes=max_bytes,
            scope_type=scope_type, scope_value=scope_value,
        ))
    return budgets


def check_budgets(results: list, budgets: list,
                  source_root: str) -> dict:
    """检查是否超出体积预算，返回 {passed, violations[], per_scope_bytes}"""
    source_root_abs = os.path.abspath(source_root)
    by_directory = {}
    by_strategy = {}

    for r in results:
        if r.status == "error":
            continue
        size = r.compressed_size or 0
        try:
            rel = os.path.relpath(os.path.abspath(r.file_path), source_root_abs).replace("\\", "/")
        except ValueError:
            rel = os.path.basename(r.file_path)
        parts = rel.split("/")
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            by_directory[prefix] = by_directory.get(prefix, 0) + size
        by_directory[""] = by_directory.get("", 0) + size
        by_strategy[r.strategy_name] = by_strategy.get(r.strategy_name, 0) + size

    violations = []
    for b in budgets:
        if b.scope_type == "strategy":
            actual = by_strategy.get(b.scope_value, 0)
            if b.max_bytes > 0 and actual > b.max_bytes:
                violations.append({
                    "budget": b.name, "scope_type": "strategy",
                    "scope_value": b.scope_value,
                    "max_bytes": b.max_bytes, "actual_bytes": actual,
                    "over_bytes": actual - b.max_bytes,
                })
        else:
            sv = b.scope_value.rstrip("/").replace("\\", "/")
            actual = by_directory.get(sv, 0)
            if b.max_bytes > 0 and actual > b.max_bytes:
                violations.append({
                    "budget": b.name, "scope_type": "directory",
                    "scope_value": sv or "(全部)",
                    "max_bytes": b.max_bytes, "actual_bytes": actual,
                    "over_bytes": actual - b.max_bytes,
                })

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "per_directory_bytes": by_directory,
        "per_strategy_bytes": by_strategy,
        "total_budgets": len(budgets),
    }


def print_budget_summary(check_result: dict):
    if check_result["total_budgets"] == 0:
        return
    tag = "通过" if check_result["passed"] else "超标"
    print(f"\n[体积预算 {tag}] {check_result['total_budgets']} 条规则，"
          f"{len(check_result['violations'])} 条超标")
    for v in check_result["violations"]:
        over_ratio = v["over_bytes"] / v["max_bytes"] if v["max_bytes"] else 0
        print(f"  [超标] {v['budget']} ({v['scope_type']}={v['scope_value']}): "
              f"{format_size(v['actual_bytes'])} > 上限 {format_size(v['max_bytes'])} "
              f"(超出 {format_size(v['over_bytes'])}, +{over_ratio:.1%})")


def parse_entries(config: dict, cli_entry: Optional[str] = None,
                  cli_source: Optional[str] = None,
                  cli_output: Optional[str] = None,
                  cli_manifest: Optional[str] = None) -> list:
    """
    解析多入口配置：
    优先 CLI 参数（单入口），否则使用 config.entries，最后退回 CLI source 参数
    """
    entries = []
    raw_entries = config.get("entries", [])

    if cli_source:
        single = EntryConfig(
            name=cli_entry or "default",
            source_root=cli_source,
            output_dir=cli_output,
            manifest_path=cli_manifest,
        )
        entries.append(single)
    elif raw_entries:
        for re_ in raw_entries:
            entries.append(EntryConfig(
                name=re_.get("name", "unnamed"),
                source_root=re_["source"],
                output_dir=re_.get("output"),
                manifest_path=re_.get("manifest"),
                report_prefix=re_.get("report_prefix"),
            ))
    return entries


def compute_exit_code(results: list, report_write_errors: list,
                       budget_violations: int = 0,
                       reference_issues: int = 0) -> int:
    """
    计算 CI 退出码（优先级从高到低）：
      5 - 引用校验失败（有未替换旧路径或断链）
      4 - 体积预算超标
      2 - 报告/manifest 写入失败
      1 - 全部图片失败
      0 - 其他情况（有成功/跳过，哪怕部分失败）
    """
    if reference_issues > 0:
        return EXIT_REFERENCE_VALIDATION_FAILED
    if budget_violations > 0:
        return EXIT_BUDGET_EXCEEDED
    if report_write_errors:
        return EXIT_REPORT_WRITE_FAILED
    if not results:
        return EXIT_OK
    has_any_success_or_skip = any(r.status != "error" for r in results)
    return EXIT_OK if has_any_success_or_skip else EXIT_ALL_FAILED


def print_ci_summary(results: list, source_root: str, output_dir: Optional[str]):
    """在终端输出末尾打印一段简短的构建摘要，适合 CI 看最后几行"""
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")

    processed = [r for r in results if r.status != "error"]
    orig = sum(r.original_size for r in processed)
    comp = sum(r.compressed_size for r in processed)
    saved = orig - comp
    ratio = (comp / orig) if orig > 0 else 0

    cfg_chg = sum(1 for r in results if r.recompress_reason == "config_changed")
    src_chg = sum(1 for r in results if r.recompress_reason == "source_changed")

    print("\n" + "=" * 80)
    print("  [构建摘要] BUILD SUMMARY")
    print("=" * 80)
    print(f"  源目录      : {source_root}")
    if output_dir:
        print(f"  输出目录    : {output_dir}")
    print(f"  总文件数    : {total}")
    print(f"  成功        : {success}")
    print(f"  跳过(增量)  : {skipped}")
    print(f"  失败        : {errors}")
    if cfg_chg or src_chg:
        print(f"  配置变更重压: {cfg_chg} | 源图变更重压: {src_chg}")
    print(f"  原始体积    : {format_size(orig)}")
    print(f"  压缩后体积  : {format_size(comp)}")
    print(f"  节省空间    : {format_size(saved)} ({ratio:.1%} 压缩比)")
    print("=" * 80)


class ImageWatchHandler(FileSystemEventHandler):
    def __init__(self, source_root: str, supported_ext: list, on_change_callback):
        super().__init__()
        self.source_root = os.path.abspath(source_root)
        self.extensions = {e.lower() for e in supported_ext}
        self.on_change = on_change_callback
        self._debounce = {}
        self._debounce_window = 0.5

    def _should_process(self, path: str) -> Optional[str]:
        if not os.path.exists(path) and not os.path.isdir(os.path.dirname(path)):
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.extensions:
            return None
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(self.source_root + os.sep):
            return None
        if is_min_file(path, list(self.extensions)):
            return None
        return abs_path

    def _fire_after_debounce(self, key: str, path: str, event_type: str):
        now = time.time()
        if key in self._debounce:
            old_time, _ = self._debounce[key]
            if now - old_time < self._debounce_window:
                return
        self._debounce[key] = (now, event_type)
        self.on_change(path, event_type)

    def on_created(self, event):
        if event.is_directory:
            return
        p = self._should_process(event.src_path)
        if p:
            time.sleep(0.05)
            self._fire_after_debounce(p, p, "created")

    def on_modified(self, event):
        if event.is_directory:
            return
        p = self._should_process(event.src_path)
        if p:
            time.sleep(0.05)
            self._fire_after_debounce(p, p, "modified")

    def on_deleted(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext in self.extensions:
            abs_p = os.path.abspath(event.src_path)
            self._fire_after_debounce(abs_p, abs_p, "deleted")

    def on_moved(self, event):
        if event.is_directory:
            return
        p_dst = self._should_process(event.dest_path)
        if p_dst:
            self._fire_after_debounce(p_dst, p_dst, "created")
        ext_src = os.path.splitext(event.src_path)[1].lower()
        if ext_src in self.extensions:
            abs_src = os.path.abspath(event.src_path)
            self._fire_after_debounce(abs_src + "_del", abs_src, "deleted")


def watch_and_compress(source_root: str, strategies: list, config: dict,
                       args_namespace):
    """watch 模式入口：启动观察者 + 串行处理变更事件"""
    general = config.get("general", {})
    supported_ext = general.get("supported_extensions", [".png", ".jpg", ".jpeg"])

    output_dir = args_namespace.output_dir
    in_place = (
        args_namespace.in_place
        and not args_namespace.no_in_place
        and not output_dir
    )

    strategies_data = [strategy_to_dict(s) for s in strategies]
    enable_cache = not args_namespace.no_cache
    cache_path = get_cache_path(config, output_dir)
    enable_clean = (args_namespace.clean or
                    (not args_namespace.no_clean and output_dir is not None))
    workers = args_namespace.workers or general.get("workers", 0) or os.cpu_count() or 4
    backup_dir = args_namespace.backup_dir or general.get("backup_dir", "_backup")
    backup_dir_arg = backup_dir if args_namespace.backup else None

    pending_events = []

    def on_change(path, event_type):
        pending_events.append((path, event_type))

    handler = ImageWatchHandler(source_root, supported_ext, on_change)
    observer = Observer()
    observer.schedule(handler, source_root, recursive=True)
    observer.start()

    abs_source_root = os.path.abspath(source_root)
    print(f"\n[监听模式] 启动: {abs_source_root} (Ctrl+C 退出)")
    print(f"   扩展名: {', '.join(supported_ext)} | 清理: {'开' if enable_clean else '关'}")

    try:
        while True:
            time.sleep(0.2)
            if not pending_events:
                continue

            batch = {}
            while pending_events:
                p, etype = pending_events.pop(0)
                batch[p] = etype

            created_modified = []
            deleted_paths = []
            for p, et in batch.items():
                if et == "deleted":
                    deleted_paths.append(p)
                elif et in ("created", "modified"):
                    if os.path.isfile(p):
                        created_modified.append(p)

            if not created_modified and not deleted_paths:
                continue

            print(f"\n[处理变更] 新增/修改 {len(created_modified)} 个，删除 {len(deleted_paths)} 个")

            all_results = []

            if created_modified:
                imgs = sorted(set(created_modified))
                task_args = [
                    (fp, abs_source_root, strategies_data, output_dir,
                     backup_dir_arg, in_place, args_namespace.no_in_place,
                     args_namespace.keep_source, cache_path, enable_cache)
                    for fp in imgs
                ]
                cache = load_cache(cache_path) if enable_cache else {
                    "version": CACHE_VERSION, "files": {}
                }
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_map = {
                        executor.submit(process_file_wrapper, a): a[0]
                        for a in task_args
                    }
                    for fut in as_completed(future_map):
                        try:
                            r = fut.result()
                            all_results.append(r)
                        except Exception as e:
                            all_results.append(CompressionResult(
                                file_path=future_map[fut],
                                output_path="", strategy_name="error",
                                original_size=0, compressed_size=0,
                                original_dimensions=(0, 0), compressed_dimensions=(0, 0),
                                compression_ratio=1.0, time_elapsed=0, status="error",
                                original_mtime=0, error=str(e),
                            ))
                all_results.sort(key=lambda r: r.file_path)

                if enable_cache:
                    for r in all_results:
                        if r.status == "success" and r.file_hash:
                            cache["files"][r.file_path] = {
                                "strategy": r.strategy_name,
                                "file_hash": r.file_hash,
                                "original_size": r.original_size,
                                "compressed_size": r.compressed_size,
                                "compression_ratio": r.compression_ratio,
                                "strategy_signature": r.strategy_signature,
                                "output_path": r.output_path,
                                "timestamp": time.time(),
                            }

            if deleted_paths and enable_cache:
                cache = cache if created_modified else load_cache(cache_path)
                for dp in deleted_paths:
                    if dp in cache.get("files", {}):
                        entry = cache["files"].pop(dp)
                        out_path = entry.get("output_path")
                        if out_path and output_dir and os.path.isfile(out_path):
                            try:
                                os.remove(out_path)
                                print(f"  [清理] 删除产物: {out_path}")
                            except Exception as e:
                                print(f"  [警告] 无法删除产物 {out_path}: {e}")
                        print(f"  [清理] 删除缓存: {dp}")

            current_sources = scan_images(
                abs_source_root, supported_ext, skip_own_output=True,
                source_root=abs_source_root, output_dir=output_dir,
                no_in_place=args_namespace.no_in_place,
            )
            if enable_clean and output_dir:
                cache = cache if (enable_cache and (created_modified or deleted_paths)) else load_cache(cache_path)
                cache = clean_stale_products(
                    abs_source_root, output_dir, current_sources, cache, supported_ext
                )

            if enable_cache:
                save_cache(cache_path, cache)

            if all_results:
                print_report(all_results)

            manifest_path = None
            if args_namespace.manifest:
                manifest_path_arg = None if args_namespace.manifest == "__DEFAULT__" else args_namespace.manifest
                manifest_path = get_manifest_path(config, output_dir, manifest_path_arg)
                full_scan = scan_images(abs_source_root, supported_ext, skip_own_output=True,
                                        source_root=abs_source_root, output_dir=output_dir,
                                        no_in_place=args_namespace.no_in_place)
                cache_for_manifest = load_cache(cache_path) if enable_cache else {"version": CACHE_VERSION, "files": {}}
                manifest_results = []
                for fp in full_scan:
                    if fp in cache_for_manifest.get("files", {}):
                        ce = cache_for_manifest["files"][fp]
                        try:
                            orig_size = os.path.getsize(fp)
                            out_path = ce.get("output_path", fp)
                            comp_size = os.path.getsize(out_path) if os.path.isfile(out_path) else orig_size
                        except OSError:
                            continue
                        manifest_results.append(CompressionResult(
                            file_path=fp,
                            output_path=ce.get("output_path", fp),
                            strategy_name=ce.get("strategy", "cached"),
                            original_size=orig_size, compressed_size=comp_size,
                            original_dimensions=(0, 0), compressed_dimensions=(0, 0),
                            compression_ratio=ce.get("compression_ratio", comp_size / orig_size if orig_size else 1),
                            time_elapsed=0, status="skipped",
                            original_mtime=0, file_hash=ce.get("file_hash"),
                            strategy_signature=ce.get("strategy_signature"),
                        ))
                try:
                    save_manifest(manifest_results, manifest_path, abs_source_root, output_dir)
                except Exception as e:
                    manifest_path = None
                    print(f"  [警告] Manifest 更新失败: {e}")

            auto_refresh_paths = getattr(args_namespace, "watch_refresh", None) or []
            changed_ref_files = []
            if manifest_path and auto_refresh_paths:
                refresh_summary = replace_references_in_files(
                    manifest_path, auto_refresh_paths, preview=False
                )
                if "error" not in refresh_summary and "warning" not in refresh_summary:
                    changed_ref_files = [
                        e["file"] for e in refresh_summary.get("files", [])
                        if e.get("replacements", 0) > 0
                    ]

            imgs_changed = []
            for r in all_results:
                try:
                    rel = os.path.relpath(r.file_path, abs_source_root).replace("\\", "/")
                except ValueError:
                    rel = os.path.basename(r.file_path)
                status_tag = "OK" if r.status == "success" else ("SKIP" if r.status == "skipped" else "ERR")
                size_info = f"{format_size(r.original_size)}->{format_size(r.compressed_size)}" if r.status != "error" else "error"
                imgs_changed.append(f"{rel}({status_tag},{size_info})")
            for dp in deleted_paths:
                try:
                    rel = os.path.relpath(dp, abs_source_root).replace("\\", "/")
                except ValueError:
                    rel = os.path.basename(dp)
                imgs_changed.append(f"{rel}(DEL)")

            print("\n" + "-" * 60)
            print(f"[本轮变更] 图片 {len(imgs_changed)} 个 | 引用刷新 {len(changed_ref_files)} 个文件")
            if imgs_changed:
                print(f"  图片: " + ", ".join(imgs_changed[:8]) + (" ..." if len(imgs_changed) > 8 else ""))
            if changed_ref_files:
                rel_refs = []
                for rf in changed_ref_files[:5]:
                    try:
                        rel_refs.append(os.path.relpath(rf).replace("\\", "/"))
                    except ValueError:
                        rel_refs.append(os.path.basename(rf))
                print(f"  引用: " + ", ".join(rel_refs) + (" ..." if len(changed_ref_files) > 5 else ""))
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n[监听停止] (Ctrl+C)")
    finally:
        observer.stop()
        observer.join()


def get_cache_path(config: dict, output_dir: Optional[str]) -> str:
    if output_dir:
        return os.path.join(output_dir, ".compress_cache.json")
    return config.get("general", {}).get("cache_file", ".compress_cache.json")


def get_manifest_path(config: dict, output_dir: Optional[str],
                      manifest_arg: Optional[str]) -> str:
    if manifest_arg:
        return manifest_arg
    if output_dir:
        return os.path.join(output_dir, "manifest.json")
    return config.get("general", {}).get("manifest_file", "manifest.json")


def main():
    parser = argparse.ArgumentParser(
        description="批量图片压缩工具 - 适合构建流程，支持输出目录、增量压缩、Manifest、路径替换、Watch 模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 输出目录模式 + 清理过期产物 + 生成 manifest
  python img_compress.py ./assets -o dist-images --clean --manifest

  # CI 构建：输出目录 + 清理 + 多格式报告 + manifest
  python img_compress.py ./assets -o dist-images --clean --backup \\
    --report reports/compress.json --csv reports/compress.csv \\
    --html reports/compress.html --manifest dist-images/manifest.json

  # 根据 manifest 替换 HTML/CSS/JS 中的图片路径（先预览差异）
  python img_compress.py ./assets -o dist-images --manifest \\
    --replace-preview ./templates ./src --replace ./templates ./src

  # 本地开发：监听源目录变化，自动压缩 + 更新 manifest
  python img_compress.py ./assets -o dist-images --manifest --watch

  # --no-in-place 模式：完全不碰原文件，自动跳过 .min 文件
  python img_compress.py ./assets --no-in-place

  # 修改配置后强制重新压缩（增量缓存会自动检测配置变化）
  python img_compress.py ./assets -o dist-images

  # 禁用清理过期产物
  python img_compress.py ./assets -o dist-images --no-clean
        """,
    )
    parser.add_argument("directory", nargs="?", default=None,
                        help="要扫描的图片源目录（CLI 单入口模式）；配置里有 entries 时可不传")
    parser.add_argument("--entry", default=None,
                        help="多入口模式下指定只跑某个入口名")
    parser.add_argument("-c", "--config", default="compress_config.yaml",
                        help="配置文件路径 (默认: compress_config.yaml)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录，压缩结果按原目录结构放入此目录")
    parser.add_argument("--backup", action="store_true",
                        help="将原图备份（先备份再压缩，确保原图安全）")
    parser.add_argument("--backup-dir", default=None,
                        help="自定义备份目录路径 (默认: 配置文件中的 backup_dir)")
    parser.add_argument("--in-place", action="store_true", default=True,
                        help="原地压缩，替换原文件，转 WebP 时会删除源文件 (默认)")
    parser.add_argument("--no-in-place", action="store_true",
                        help="完全不碰原文件，压缩结果另存为 .min.xxx 新文件")
    parser.add_argument("--keep-source", action="store_true",
                        help="原地压缩且转 WebP 时保留源文件，不删除原 PNG/JPG")
    parser.add_argument("--clean", action="store_true",
                        help="清理过期产物：源文件删除后，自动删除 dist 中对应产物 (默认开启)")
    parser.add_argument("--no-clean", action="store_true",
                        help="禁止清理过期产物")
    parser.add_argument("--manifest", nargs="?", const="__DEFAULT__",
                        help="生成 manifest.json，可指定路径 (默认: 输出目录/manifest.json)")
    parser.add_argument("--replace", nargs="+", default=None, metavar="PATH",
                        help="根据 manifest 替换指定目录/文件中的 HTML/CSS/JS 图片路径 (写回文件)")
    parser.add_argument("--replace-preview", nargs="+", default=None, metavar="PATH",
                        help="根据 manifest 预览路径替换的差异 (不写回文件，只打印 diff)")
    parser.add_argument("--validate", nargs="+", default=None, metavar="PATH",
                        help="引用校验：检查 HTML/CSS/JS 中是否还有未替换旧路径、是否有断链")
    parser.add_argument("--watch", action="store_true",
                        help="watch 模式：监听源目录图片变化，自动压缩并更新 manifest")
    parser.add_argument("--watch-refresh", nargs="+", default=None, metavar="PATH",
                        help="watch 模式下自动刷新这些目录/文件中的 HTML/CSS/JS 图片引用")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行工作进程数 (默认: 配置文件中的 workers 或 CPU核心数)")
    parser.add_argument("--report", default=None,
                        help="JSON 报告输出路径 (默认: 配置文件中的 report_file)")
    parser.add_argument("--csv", default=None,
                        help="CSV 报告输出路径")
    parser.add_argument("--html", default=None,
                        help="HTML 报告输出路径")
    parser.add_argument("--no-cache", action="store_true",
                        help="禁用增量压缩缓存，强制全量处理")
    parser.add_argument("--clear-cache", action="store_true",
                        help="清除现有缓存后再运行")
    parser.add_argument("--dry-run", action="store_true",
                        help="只扫描和匹配策略，不实际压缩")
    parser.add_argument("--budget-strict", action="store_true",
                        help="体积预算超标时返回非 0 退出码（默认仅告警）")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        sys.exit(f"错误: 配置文件不存在 - {config_path}\n"
                 f"请创建配置文件或使用 -c 指定路径")

    if args.watch and not WATCHDOG_AVAILABLE:
        sys.exit("错误: --watch 需要 watchdog 库，请安装: pip install watchdog")

    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    strategies = parse_strategies(config)
    budgets = parse_budgets(config)
    general = config.get("general", {})

    entries = parse_entries(
        config, cli_entry=args.entry, cli_source=args.directory,
        cli_output=args.output_dir,
        cli_manifest=(None if args.manifest == "__DEFAULT__" else args.manifest)
        if args.manifest else None,
    )

    if not entries:
        sys.exit("错误: 未找到源目录。请用位置参数指定目录，或在配置里写 entries")

    if args.entry:
        entries = [e for e in entries if e.name == args.entry]
        if not entries:
            sys.exit(f"错误: 入口 '{args.entry}' 未在配置中找到")

    if args.watch and len(entries) > 1:
        sys.exit("错误: --watch 模式暂不支持多入口，请使用 --entry 指定一个入口")

    def process_single_entry(entry: EntryConfig) -> dict:
        """处理单个入口（源目录 + 输出目录 + manifest），返回处理结果汇总"""
        supported_ext = general.get("supported_extensions", [".png", ".jpg", ".jpeg"])
        backup_dir = args.backup_dir or general.get("backup_dir", "_backup")
        workers = args.workers or general.get("workers", 0) or os.cpu_count() or 4
        entry_output = entry.output_dir or args.output_dir
        cache_path = get_cache_path(config, entry_output)
        enable_cache = not args.no_cache
        enable_clean = (args.clean or
                        (not args.no_clean and entry_output is not None))
        entry_manifest_arg = (
            entry.manifest_path
            or (None if args.manifest == "__DEFAULT__" else args.manifest)
            if args.manifest else None
        )
        entry_report_prefix = entry.report_prefix or f"{entry.name}_"

        if entry.name != "default" or len(entries) > 1:
            print(f"\n{'=' * 72}")
            print(f"  [入口] {entry.name}  源: {entry.source_root}"
                  + (f"  输出: {entry_output}" if entry_output else ""))
            print(f"{'=' * 72}")

        if not os.path.isdir(entry.source_root):
            print(f"[跳过] 入口 {entry.name}: 源目录不存在 {entry.source_root}")
            return {"entry": entry.name, "results": [],
                    "report_write_errors": [f"源目录不存在: {entry.source_root}"],
                    "manifest_path": None, "source_root": entry.source_root,
                    "output_dir": entry_output}

        source_root_abs = os.path.abspath(entry.source_root)

        if args.watch:
            watch_and_compress(source_root_abs, strategies, config, args)
            return {"entry": entry.name, "results": [], "report_write_errors": [],
                    "manifest_path": None, "source_root": source_root_abs,
                    "output_dir": entry_output}

        print(f"扫描目录: {source_root_abs}")
        print(f"支持的格式: {', '.join(supported_ext)}")

        images = scan_images(
            source_root_abs, supported_ext,
            skip_own_output=True,
            source_root=source_root_abs,
            output_dir=entry_output,
            no_in_place=args.no_in_place,
        )
        print(f"找到 {len(images)} 张图片 (已跳过自身生成的产物)")

        results = []
        report_write_errors = []

        if not images:
            print("没有找到可处理的图片")
            if entry_output and enable_clean:
                print("\n执行过期产物清理...")
                cache = load_cache(cache_path) if enable_cache else {
                    "version": CACHE_VERSION, "files": {}
                }
                cache = clean_stale_products(
                    source_root_abs, entry_output, [], cache, supported_ext
                )
                if enable_cache:
                    save_cache(cache_path, cache)
        else:
            if entry_output:
                print(f"输出目录: {os.path.abspath(entry_output)}")

            print(f"压缩策略: {', '.join(s.name for s in strategies)}")
            print(f"并行进程数: {workers}")
            print(f"增量缓存: {'已启用' if enable_cache else '已禁用'} ({cache_path})")
            print(f"过期清理: {'已启用' if enable_clean else '已禁用'}")

            if args.backup:
                print(f"备份目录: {os.path.abspath(backup_dir)}")

            if args.no_in_place:
                print("模式: --no-in-place (完全不碰原文件，自动跳过 .min 产物)")
            elif args.keep_source:
                print("模式: 原地压缩 + 保留源文件")
            else:
                print("模式: 原地压缩 (转 WebP 后删除源文件)")

            if args.clear_cache and enable_cache:
                if os.path.isfile(cache_path):
                    os.remove(cache_path)
                    print(f"已清除缓存: {cache_path}")

            if args.dry_run:
                print("\n[试运行模式] 策略匹配预览:")
                print("-" * 120)
                for img_path in images:
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                        strategy = resolve_strategy(img_path, w, h, strategies)
                        original_ext = os.path.splitext(img_path)[1].lower()
                        in_place = (
                            args.in_place
                            and not args.no_in_place
                            and not entry_output
                        )
                        output_path = compute_output_path(
                            img_path, source_root_abs, entry_output,
                            in_place, args.no_in_place, strategy, original_ext
                        )
                        print(
                            f"  {img_path:<55} {w}x{h:<8} "
                            f"-> {strategy.name:<12} -> {output_path}"
                        )
                    except Exception as e:
                        print(f"  {img_path:<55} [错误: {e}]")
            else:
                if args.clear_cache and enable_cache:
                    cache = {"version": CACHE_VERSION, "files": {}}
                elif enable_cache:
                    cache = load_cache(cache_path)
                else:
                    cache = {"version": CACHE_VERSION, "files": {}}

                strategies_data = [strategy_to_dict(s) for s in strategies]
                in_place = (
                    args.in_place
                    and not args.no_in_place
                    and not entry_output
                )

                task_args = [
                    (img_path, source_root_abs, strategies_data, entry_output,
                     backup_dir if args.backup else None, in_place, args.no_in_place,
                     args.keep_source, cache_path, enable_cache)
                    for img_path in images
                ]

                print(f"\n开始压缩 ({workers} 个进程)...")

                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(process_file_wrapper, arg): arg[0]
                        for arg in task_args
                    }
                    completed = 0
                    total = len(futures)
                    for future in as_completed(futures):
                        file_path = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            results.append(CompressionResult(
                                file_path=file_path,
                                output_path="",
                                strategy_name="error",
                                original_size=0,
                                compressed_size=0,
                                original_dimensions=(0, 0),
                                compressed_dimensions=(0, 0),
                                compression_ratio=1.0,
                                time_elapsed=0,
                                status="error",
                                original_mtime=0,
                                error=str(e),
                            ))
                        completed += 1
                        if completed % 10 == 0 or completed == total:
                            print(f"  进度: {completed}/{total} ({completed / total:.0%})")

                results.sort(key=lambda r: r.file_path)

                if enable_cache:
                    for r in results:
                        if r.status == "success" and r.file_hash:
                            cache["files"][r.file_path] = {
                                "strategy": r.strategy_name,
                                "file_hash": r.file_hash,
                                "original_size": r.original_size,
                                "compressed_size": r.compressed_size,
                                "compression_ratio": r.compression_ratio,
                                "strategy_signature": r.strategy_signature,
                                "output_path": r.output_path,
                                "timestamp": time.time(),
                            }
                        elif r.status == "skipped" and r.file_hash:
                            if r.file_path in cache["files"]:
                                cache["files"][r.file_path]["timestamp"] = time.time()

                if enable_clean and entry_output:
                    print("\n清理过期产物...")
                    cache = clean_stale_products(
                        source_root_abs, entry_output, images, cache, supported_ext
                    )

                if enable_cache:
                    save_cache(cache_path, cache)

                print_report(results)

        default_report_file = args.report or general.get("report_file", "compression_report.json")
        if len(entries) == 1 and entry.name == "default":
            json_path = default_report_file
            csv_path = args.csv
            html_path = args.html
        else:
            base_dir = os.path.dirname(default_report_file) or "."
            base_name = os.path.splitext(os.path.basename(default_report_file))[0]
            json_path = os.path.join(base_dir, f"{entry_report_prefix}{base_name}.json")
            csv_path = (os.path.join(os.path.dirname(args.csv),
                                     f"{entry_report_prefix}{os.path.basename(args.csv)}")
                        if args.csv else None)
            html_path = (os.path.join(os.path.dirname(args.html),
                                      f"{entry_report_prefix}{os.path.basename(args.html)}")
                         if args.html else None)

        try:
            save_json_report(results, json_path, strategies)
            print(f"\nJSON 报告: {os.path.abspath(json_path)}")
        except Exception as e:
            print(f"\n[警告] JSON 报告导出失败: {e}")
            report_write_errors.append(f"JSON报告: {e}")

        if csv_path:
            try:
                save_csv_report(results, csv_path)
                print(f"CSV 报告: {os.path.abspath(csv_path)}")
            except Exception as e:
                print(f"[警告] CSV 报告导出失败: {e}")
                report_write_errors.append(f"CSV报告: {e}")

        if html_path:
            try:
                save_html_report(results, html_path, strategies)
                print(f"HTML 报告: {os.path.abspath(html_path)}")
            except Exception as e:
                print(f"[警告] HTML 报告导出失败: {e}")
                report_write_errors.append(f"HTML报告: {e}")

        manifest_path_final = None
        if args.manifest:
            manifest_path_final = get_manifest_path(config, entry_output, entry_manifest_arg)
            try:
                save_manifest(results, manifest_path_final, source_root_abs, entry_output)
                print(f"Manifest: {os.path.abspath(manifest_path_final)}")
            except Exception as e:
                print(f"[警告] Manifest 导出失败: {e}")
                report_write_errors.append(f"Manifest: {e}")

        budget_check = check_budgets(results, budgets, source_root_abs)
        print_budget_summary(budget_check)

        return {
            "entry": entry.name,
            "results": results,
            "report_write_errors": report_write_errors,
            "manifest_path": manifest_path_final,
            "source_root": source_root_abs,
            "output_dir": entry_output,
            "budget_check": budget_check,
        }

    entry_results = []
    for ent in entries:
        if args.watch:
            process_single_entry(ent)
            return
        ent_res = process_single_entry(ent)
        entry_results.append(ent_res)

    all_results = []
    all_report_errors = []
    all_manifests = []
    all_source_roots = []
    for er in entry_results:
        all_results.extend(er["results"])
        all_report_errors.extend(er.get("report_write_errors", []))
        if er.get("manifest_path"):
            all_manifests.append(er["manifest_path"])
        if er.get("source_root"):
            all_source_roots.append(er["source_root"])

    if len(entry_results) > 1:
        print(f"\n{'=' * 72}")
        print(f"  [多入口汇总] 共 {len(entry_results)} 个入口")
        for er in entry_results:
            n_ok = sum(1 for r in er["results"] if r.status == "success")
            n_skip = sum(1 for r in er["results"] if r.status == "skipped")
            n_err = sum(1 for r in er["results"] if r.status == "error")
            print(f"    - {er['entry']}: 成功 {n_ok} | 跳过 {n_skip} | 失败 {n_err}")
        print(f"{'=' * 72}")

    if args.replace_preview and all_manifests:
        for mp in all_manifests:
            summary = replace_references_in_files(mp, args.replace_preview, preview=True)
            print_replace_summary(summary, preview=True)

    if args.replace and all_manifests:
        for mp in all_manifests:
            summary = replace_references_in_files(mp, args.replace, preview=False)
            print_replace_summary(summary, preview=False)

    validate_issues_count = 0
    if args.validate:
        print(f"\n{'=' * 72}")
        print("  [引用校验] 跨所有入口 manifest 检查")
        print(f"{'=' * 72}")
        for er in entry_results:
            mp = er.get("manifest_path")
            if not mp:
                continue
            print(f"\n入口 {er['entry']} ({os.path.basename(mp)}):")
            vres = validate_references(mp, args.validate,
                                        source_roots=all_source_roots)
            print_validate_summary(vres)
            if "summary" in vres and not vres["summary"]["passed"]:
                validate_issues_count += (
                    vres["summary"]["stale_source_refs"]
                    + vres["summary"]["broken_links"]
                )

    if len(entry_results) == 1:
        print_ci_summary(
            entry_results[0]["results"],
            entry_results[0]["source_root"],
            entry_results[0]["output_dir"],
        )
    else:
        print_ci_summary(all_results, "(多入口)", None)

    total_budget_violations = sum(
        len(er.get("budget_check", {}).get("violations", [])) for er in entry_results
    )
    budget_violations_for_exit = total_budget_violations if args.budget_strict else 0

    exit_code = compute_exit_code(
        all_results, all_report_errors,
        budget_violations=budget_violations_for_exit,
        reference_issues=validate_issues_count,
    )
    if exit_code == EXIT_ALL_FAILED:
        print(f"\n[构建失败] 所有 {len(all_results)} 张图片全部压缩失败，"
              f"返回退出码 {EXIT_ALL_FAILED}")
    elif exit_code == EXIT_REPORT_WRITE_FAILED:
        print(f"\n[构建失败] 报告/Manifest 写入失败: {all_report_errors}，"
              f"返回退出码 {EXIT_REPORT_WRITE_FAILED}")
    elif exit_code == EXIT_BUDGET_EXCEEDED:
        print(f"\n[构建失败] 体积预算超标 {total_budget_violations} 条，"
              f"返回退出码 {EXIT_BUDGET_EXCEEDED}")
    elif exit_code == EXIT_REFERENCE_VALIDATION_FAILED:
        print(f"\n[构建失败] 引用校验失败（未替换/断链 {validate_issues_count} 处），"
              f"返回退出码 {EXIT_REFERENCE_VALIDATION_FAILED}")
    elif exit_code != EXIT_OK:
        print(f"\n[构建失败] 返回退出码 {exit_code}")
    else:
        success = sum(1 for r in all_results if r.status == "success")
        skipped = sum(1 for r in all_results if r.status == "skipped")
        errors = sum(1 for r in all_results if r.status == "error")
        msg_parts = ["[构建完成]"]
        if success > 0:
            msg_parts.append(f"成功 {success}")
        if skipped > 0:
            msg_parts.append(f"跳过(增量) {skipped}")
        if errors > 0:
            msg_parts.append(f"部分失败 {errors}")
        if total_budget_violations > 0 and not args.budget_strict:
            msg_parts.append(f"预算告警 {total_budget_violations} 条(仅告警)")
        if validate_issues_count > 0:
            msg_parts.append(f"引用告警 {validate_issues_count} 处")
        print(f"\n{' | '.join(msg_parts)}，返回退出码 0")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
