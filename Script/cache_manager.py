#!/usr/bin/env python3
"""
缓存管理工具CLI
支持查看、清理、验证缓存
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_layer.cache_layer import (
    LabelCache, 
    FeatureCache, 
    cache_stats, 
    clear_all_caches,
    get_label_cache,
    get_feature_cache,
)


def cmd_stats(args):
    """显示缓存统计信息"""
    print("\n" + "=" * 70)
    print("缓存统计信息")
    print("=" * 70)
    
    stats = cache_stats()
    
    # 标签缓存
    print("\n【标签缓存】")
    label_stats = stats.get("labels", {})
    print(f"  路径: {label_stats.get('cache_dir', 'N/A')}")
    print(f"  文件数: {label_stats.get('total_files', 0)}")
    print(f"  大小: {label_stats.get('total_size_mb', '0.00')} MB")
    print(f"  命中数: {label_stats.get('hits', 0)}")
    print(f"  未命中数: {label_stats.get('misses', 0)}")
    print(f"  命中率: {label_stats.get('hit_rate', 'N/A')}")
    
    # 特征缓存
    print("\n【特征缓存】")
    feature_stats = stats.get("features", {})
    print(f"  路径: {feature_stats.get('cache_dir', 'N/A')}")
    print(f"  文件数: {feature_stats.get('total_files', 0)}")
    print(f"  大小: {feature_stats.get('total_size_mb', '0.00')} MB")
    print(f"  命中数: {feature_stats.get('hits', 0)}")
    print(f"  未命中数: {feature_stats.get('misses', 0)}")
    print(f"  命中率: {feature_stats.get('hit_rate', 'N/A')}")
    
    # 总计
    total_size_mb = float(label_stats.get('total_size_mb', '0.00')) + float(feature_stats.get('total_size_mb', '0.00'))
    total_files = label_stats.get('total_files', 0) + feature_stats.get('total_files', 0)
    print(f"\n【合计】")
    print(f"  总文件数: {total_files}")
    print(f"  总大小: {total_size_mb:.2f} MB")
    print()


def cmd_clear(args):
    """清空所有缓存"""
    if not args.yes:
        response = input("确认清空所有缓存? (yes/no) [no]: ").strip().lower()
        if response not in ["yes", "y"]:
            print("已取消")
            return
    
    print("\n清空缓存中...")
    clear_all_caches()
    print("✓ 已清空所有缓存")
    print()


def cmd_clear_labels(args):
    """清空标签缓存"""
    if not args.yes:
        response = input("确认清空标签缓存? (yes/no) [no]: ").strip().lower()
        if response not in ["yes", "y"]:
            print("已取消")
            return
    
    print("\n清空标签缓存中...")
    get_label_cache().clear_all()
    print("✓ 已清空标签缓存")
    print()


def cmd_clear_features(args):
    """清空特征缓存"""
    if not args.yes:
        response = input("确认清空特征缓存? (yes/no) [no]: ").strip().lower()
        if response not in ["yes", "y"]:
            print("已取消")
            return
    
    print("\n清空特征缓存中...")
    get_feature_cache().clear_all()
    print("✓ 已清空特征缓存")
    print()


def cmd_list_labels(args):
    """列出所有标签缓存文件"""
    cache_dir = Path("data/cache/labels")
    if not cache_dir.exists():
        print(f"\n标签缓存目录不存在: {cache_dir}")
        return
    
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        print(f"\n标签缓存目录为空: {cache_dir}")
        return
    
    print(f"\n【标签缓存文件列表】({len(files)} 个文件)")
    print("-" * 70)
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:50s}  {size_mb:8.2f} MB")
    print()


def cmd_list_features(args):
    """列出所有特征缓存文件"""
    cache_dir = Path("data/cache/features")
    if not cache_dir.exists():
        print(f"\n特征缓存目录不存在: {cache_dir}")
        return
    
    files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        print(f"\n特征缓存目录为空: {cache_dir}")
        return
    
    print(f"\n【特征缓存文件列表】({len(files)} 个文件)")
    print("-" * 70)
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:50s}  {size_mb:8.2f} MB")
    print()


def build_parser():
    """构建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="缓存管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cache_manager.py stats              # 查看缓存统计
  python cache_manager.py clear              # 清空所有缓存
  python cache_manager.py clear-labels       # 清空标签缓存
  python cache_manager.py clear-features     # 清空特征缓存
  python cache_manager.py list-labels        # 列出标签缓存文件
  python cache_manager.py list-features      # 列出特征缓存文件
  python cache_manager.py clear --yes        # 无提示清空
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # stats 子命令
    subparsers.add_parser(
        "stats",
        help="显示缓存统计信息"
    )
    
    # clear 子命令
    clear_parser = subparsers.add_parser(
        "clear",
        help="清空所有缓存"
    )
    clear_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="无提示确认"
    )
    
    # clear-labels 子命令
    clear_labels_parser = subparsers.add_parser(
        "clear-labels",
        help="清空标签缓存"
    )
    clear_labels_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="无提示确认"
    )
    
    # clear-features 子命令
    clear_features_parser = subparsers.add_parser(
        "clear-features",
        help="清空特征缓存"
    )
    clear_features_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="无提示确认"
    )
    
    # list-labels 子命令
    subparsers.add_parser(
        "list-labels",
        help="列出所有标签缓存文件"
    )
    
    # list-features 子命令
    subparsers.add_parser(
        "list-features",
        help="列出所有特征缓存文件"
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    command_map = {
        "stats": cmd_stats,
        "clear": cmd_clear,
        "clear-labels": cmd_clear_labels,
        "clear-features": cmd_clear_features,
        "list-labels": cmd_list_labels,
        "list-features": cmd_list_features,
    }
    
    if args.command in command_map:
        command_map[args.command](args)
    else:
        print(f"未知命令: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中止")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
