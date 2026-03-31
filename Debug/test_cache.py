#!/usr/bin/env python3
"""
缓存功能测试脚本
验证标签缓存和特征缓存工作正常
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_layer.cache_layer import LabelCache, FeatureCache, cache_stats, clear_all_caches
import pandas as pd
import numpy as np


def test_label_cache():
    """测试标签缓存"""
    print("\n=== 测试标签缓存 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LabelCache(cache_dir=tmpdir, enable=True)
        
        # 测试1: 缓存不存在时返回None
        result = cache.load("BTC", "hash123")
        assert result is None, "未缓存时应返回None"
        assert cache.misses == 1, "miss计数应为1"
        print("✓ 未缓存查询返回None")
        
        # 测试2: 保存标签
        labels = [
            {"symbol": "BTC", "klu_idx": 1, "label": 1, "price": 50000},
            {"symbol": "BTC", "klu_idx": 2, "label": 0, "price": 49000},
        ]
        saved = cache.save("BTC", "hash123", labels)
        assert saved, "保存应返回True"
        print("✓ 标签保存成功")
        
        # 测试3: 从缓存加载标签
        loaded = cache.load("BTC", "hash123")
        assert loaded is not None, "缓存存在时应返回数据"
        assert len(loaded) == 2, "应加载2条记录"
        assert cache.hits == 1, "hit计数应为1"
        assert loaded[0]["price"] == 50000, "数据应正确"
        print("✓ 标签缓存加载成功")
        
        # 测试4: 不同hash返回None
        result = cache.load("BTC", "hash456")
        assert result is None, "不同hash应返回None"
        print("✓ 不同hash查询返回None")
        
        # 测试5: 打印统计
        stats = cache.stats()
        assert stats["hits"] == 1, f"统计hit应为1，实际{stats['hits']}"
        assert stats["misses"] == 2, f"统计miss应为2，实际{stats['misses']}"
        print(f"✓ 缓存统计正确: {stats}")


def test_feature_cache():
    """测试特征缓存"""
    print("\n=== 测试特征缓存 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = FeatureCache(cache_dir=tmpdir, enable=True)
        
        # 测试1: 缓存不存在时返回None
        result = cache.load("ETH", "hash789")
        assert result is None, "未缓存时应返回None"
        print("✓ 未缓存查询返回None")
        
        # 测试2: 保存特征
        features = pd.DataFrame({
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [0.5, 0.6, 0.7],
        })
        saved = cache.save("ETH", "hash789", features)
        assert saved, "保存应返回True"
        print("✓ 特征保存成功")
        
        # 测试3: 从缓存加载特征
        loaded = cache.load("ETH", "hash789")
        assert loaded is not None, "缓存存在时应返回数据"
        assert len(loaded) == 3, f"应加载3行，实际{len(loaded)}"
        assert list(loaded.columns) == ["feature_1", "feature_2"], "列名应正确"
        assert loaded["feature_1"].iloc[0] == 1.0, "数据值应正确"
        print("✓ 特征缓存加载成功")
        
        # 测试4: 打印统计
        stats = cache.stats()
        assert stats["hits"] == 1, f"统计hit应为1，实际{stats['hits']}"
        assert stats["misses"] == 1, f"统计miss应为1，实际{stats['misses']}"
        print(f"✓ 缓存统计正确: {stats}")


def test_cache_disabled():
    """测试禁用缓存"""
    print("\n=== 测试缓存禁用 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 禁用缓存
        cache = LabelCache(cache_dir=tmpdir, enable=False)
        
        labels = [{"symbol": "BTC", "label": 1}]
        saved = cache.save("BTC", "hash1", labels)
        assert not saved, "禁用缓存时save应返回False"
        
        loaded = cache.load("BTC", "hash1")
        assert loaded is None, "禁用缓存时load应返回None"
        print("✓ 缓存禁用工作正常")


def test_clear_cache():
    """测试清空缓存"""
    print("\n=== 测试清空缓存 ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LabelCache(cache_dir=tmpdir, enable=True)
        
        labels = [{"symbol": "BTC", "label": 1}]
        cache.save("BTC", "hash1", labels)
        cache.save("ETH", "hash1", labels)
        
        # 验证文件存在
        cache_path = Path(tmpdir)
        files_before = len(list(cache_path.glob("*.parquet")))
        assert files_before == 2, f"应有2个缓存文件，实际{files_before}"
        
        # 清空缓存
        cache.clear_all()
        
        # 验证文件被删除
        files_after = len(list(cache_path.glob("*.parquet")))
        assert files_after == 0, f"清空后应无缓存文件，实际{files_after}"
        print("✓ 缓存清空工作正常")


if __name__ == "__main__":
    try:
        test_label_cache()
        test_feature_cache()
        test_cache_disabled()
        test_clear_cache()
        
        print("\n" + "=" * 60)
        print("✓ 所有缓存测试通过！")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
