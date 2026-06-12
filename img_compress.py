#!/usr/bin/env python3
"""批量图片压缩脚本 - 根据图片用途分别处理，支持多进程并行压缩"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    sys.exit("需要 PyYAML: pip install PyYAML")

try:
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow: pip install Pillow")


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


@dataclass
class CompressionResult:
    file_path: str
    strategy_name: str
    original_size: int
    compressed_size: int
    original_dimensions: tuple
    compressed_dimensions: tuple
    compression_ratio: float
    time_elapsed: float
    status: str
    error: Optional[str] = None


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
    output_format = os.path.splitext(output_path)[1].upper().lstrip(".")
    if output_format == "JPG":
        output_format = "JPEG"

    best_path = output_path + ".tmp"
    best_size = original_size

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

        if img.mode in ("RGBA", "P"):
            rgb_img = img.convert("RGB")
            rgb_img.save(best_path, **save_kwargs)
        else:
            img.save(best_path, **save_kwargs)

        current_size = os.path.getsize(best_path)

        if current_size <= target_size:
            best_size = current_size
            break

        if current_size < best_size:
            best_size = current_size

        quality -= 5

    if best_size < original_size:
        shutil.move(best_path, output_path)
    else:
        if os.path.exists(best_path):
            os.remove(best_path)
        if img.mode in ("RGBA", "P"):
            rgb_img = img.convert("RGB")
            rgb_img.save(output_path, format=output_format, quality=quality,
                         optimize=True, subsampling="4:2:0")
        else:
            img.save(output_path, format=output_format, quality=quality,
                     optimize=True, subsampling="4:2:0")
        best_size = os.path.getsize(output_path)

    return best_size


def compress_single(file_path: str, strategies: list,
                    backup_dir: Optional[str] = None,
                    in_place: bool = True) -> CompressionResult:
    start_time = time.time()
    original_size = os.path.getsize(file_path)

    try:
        img = Image.open(file_path)
        img.load()
        width, height = img.size
        original_dimensions = (width, height)
    except Exception as e:
        return CompressionResult(
            file_path=file_path,
            strategy_name="error",
            original_size=original_size,
            compressed_size=original_size,
            original_dimensions=(0, 0),
            compressed_dimensions=(0, 0),
            compression_ratio=1.0,
            time_elapsed=time.time() - start_time,
            status="error",
            error=f"无法打开图片: {e}",
        )

    strategy = resolve_strategy(file_path, width, height, strategies)
    exif_data = extract_exif(img)

    if not in_place:
        output_path = file_path
    else:
        output_path = file_path

    if strategy.format == "webp":
        if in_place:
            output_path = os.path.splitext(file_path)[0] + ".webp"
        compressed_size = compress_to_webp(img, output_path, strategy, exif_data)
        try:
            new_img = Image.open(output_path)
            compressed_dimensions = new_img.size
            new_img.close()
        except Exception:
            compressed_dimensions = original_dimensions
    else:
        compressed_size = compress_with_target_ratio(
            img, output_path, original_size, strategy, exif_data
        )
        try:
            new_img = Image.open(output_path)
            compressed_dimensions = new_img.size
            new_img.close()
        except Exception:
            compressed_dimensions = original_dimensions

    if backup_dir:
        abs_backup = os.path.abspath(backup_dir)
        rel_path = os.path.relpath(file_path)
        backup_path = os.path.join(abs_backup, rel_path)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        if file_path != output_path or not in_place:
            shutil.copy2(file_path, backup_path)
        else:
            shutil.move(file_path, backup_path)

    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    time_elapsed = time.time() - start_time

    img.close()

    return CompressionResult(
        file_path=file_path,
        strategy_name=strategy.name,
        original_size=original_size,
        compressed_size=compressed_size,
        original_dimensions=original_dimensions,
        compressed_dimensions=compressed_dimensions,
        compression_ratio=compression_ratio,
        time_elapsed=time_elapsed,
        status="success",
    )


def process_file_wrapper(args):
    file_path, strategies_data, backup_dir, in_place = args
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
    return compress_single(file_path, strategies, backup_dir, in_place)


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
    }


def scan_images(directory: str, extensions: list) -> list:
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                images.append(os.path.join(root, f))
    return sorted(images)


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_report(results: list):
    print("\n" + "=" * 100)
    print("  图片压缩对比报告")
    print("=" * 100)

    header = f"{'文件':<40} {'策略':<12} {'原始大小':>10} {'压缩后':>10} {'压缩比':>8} {'耗时':>8} {'状态':<8}"
    print(header)
    print("-" * 100)

    total_original = 0
    total_compressed = 0
    total_time = 0
    success_count = 0
    error_count = 0

    for r in results:
        short_path = r.file_path
        if len(short_path) > 38:
            short_path = "..." + short_path[-35:]
        ratio_str = f"{r.compression_ratio:.1%}"
        time_str = f"{r.time_elapsed:.2f}s"

        print(
            f"{short_path:<40} {r.strategy_name:<12} "
            f"{format_size(r.original_size):>10} "
            f"{format_size(r.compressed_size):>10} "
            f"{ratio_str:>8} {time_str:>8} {r.status:<8}"
        )

        if r.status == "success":
            total_original += r.original_size
            total_compressed += r.compressed_size
            total_time += r.time_elapsed
            success_count += 1
        else:
            error_count += 1

    print("-" * 100)
    if success_count > 0:
        overall_ratio = total_compressed / total_original if total_original > 0 else 0
        saved = total_original - total_compressed
        print(f"\n  总计: {success_count} 张成功, {error_count} 张失败")
        print(f"  原始总体积: {format_size(total_original)}")
        print(f"  压缩后总体积: {format_size(total_compressed)}")
        print(f"  总压缩比: {overall_ratio:.1%}")
        print(f"  节省空间: {format_size(saved)}")
        print(f"  总耗时: {total_time:.2f}s")
    print("=" * 100)


def save_json_report(results: list, report_path: str):
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_files": len(results),
            "success": sum(1 for r in results if r.status == "success"),
            "error": sum(1 for r in results if r.status == "error"),
            "total_original_size": sum(r.original_size for r in results if r.status == "success"),
            "total_compressed_size": sum(r.compressed_size for r in results if r.status == "success"),
        },
        "details": [],
    }

    for r in results:
        report_data["details"].append({
            "file": r.file_path,
            "strategy": r.strategy_name,
            "original_size": r.original_size,
            "compressed_size": r.compressed_size,
            "original_dimensions": list(r.original_dimensions),
            "compressed_dimensions": list(r.compressed_dimensions),
            "compression_ratio": round(r.compression_ratio, 4),
            "time_elapsed": round(r.time_elapsed, 3),
            "status": r.status,
            "error": r.error,
        })

    overall = report_data["summary"]
    if overall["total_original_size"] > 0:
        report_data["summary"]["overall_ratio"] = round(
            overall["total_compressed_size"] / overall["total_original_size"], 4
        )

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="批量图片压缩工具 - 根据图片用途分别处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python img_compress.py ./images
  python img_compress.py ./images -c compress_config.yaml
  python img_compress.py ./images -c config.yaml --backup --workers 4
  python img_compress.py ./images --backup-dir ./originals --report report.json
        """,
    )
    parser.add_argument("directory", help="要扫描的图片目录")
    parser.add_argument("-c", "--config", default="compress_config.yaml",
                        help="配置文件路径 (默认: compress_config.yaml)")
    parser.add_argument("--backup", action="store_true",
                        help="将原图移动到备份文件夹")
    parser.add_argument("--backup-dir", default=None,
                        help="自定义备份目录路径 (默认: 配置文件中的 backup_dir)")
    parser.add_argument("--no-in-place", action="store_true",
                        help="不替换原图，将压缩结果保存到新文件")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行工作进程数 (默认: 配置文件中的 workers 或 CPU核心数)")
    parser.add_argument("--report", default=None,
                        help="JSON报告输出路径 (默认: 配置文件中的 report_file)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只扫描和匹配策略，不实际压缩")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        sys.exit(f"错误: 目录不存在 - {args.directory}")

    config_path = args.config
    if not os.path.isfile(config_path):
        sys.exit(f"错误: 配置文件不存在 - {config_path}\n"
                 f"请创建配置文件或使用 -c 指定路径")

    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    strategies = parse_strategies(config)
    general = config.get("general", {})

    supported_ext = general.get("supported_extensions", [".png", ".jpg", ".jpeg"])
    backup_dir = args.backup_dir or general.get("backup_dir", "_backup")
    report_file = args.report or general.get("report_file", "compression_report.json")
    workers = args.workers or general.get("workers", 0) or os.cpu_count() or 4

    print(f"扫描目录: {args.directory}")
    print(f"支持的格式: {', '.join(supported_ext)}")
    images = scan_images(args.directory, supported_ext)
    print(f"找到 {len(images)} 张图片")

    if not images:
        print("没有找到可处理的图片")
        return

    print(f"压缩策略: {', '.join(s.name for s in strategies)}")
    print(f"并行进程数: {workers}")

    if args.backup:
        print(f"备份目录: {os.path.abspath(backup_dir)}")

    if args.dry_run:
        print("\n[试运行模式] 策略匹配预览:")
        print("-" * 80)
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                strategy = resolve_strategy(img_path, w, h, strategies)
                print(f"  {img_path:<50} {w}x{h:<8} -> {strategy.name}")
            except Exception as e:
                print(f"  {img_path:<50} [错误: {e}]")
        return

    strategies_data = [strategy_to_dict(s) for s in strategies]
    in_place = not args.no_in_place

    task_args = [
        (img_path, strategies_data, backup_dir if args.backup else None, in_place)
        for img_path in images
    ]

    results = []
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
                    strategy_name="error",
                    original_size=0,
                    compressed_size=0,
                    original_dimensions=(0, 0),
                    compressed_dimensions=(0, 0),
                    compression_ratio=1.0,
                    time_elapsed=0,
                    status="error",
                    error=str(e),
                ))
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"  进度: {completed}/{total} ({completed / total:.0%})")

    results.sort(key=lambda r: r.file_path)
    print_report(results)

    save_json_report(results, report_file)
    print(f"\n详细报告已保存到: {os.path.abspath(report_file)}")


if __name__ == "__main__":
    main()
