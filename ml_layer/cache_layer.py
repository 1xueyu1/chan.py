"""
缓存层: 管理标签和特征的Parquet缓存
支持 cold/hot 启动优化
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import re
import shutil

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_namespace(namespace: str) -> str:
    """规范化缓存命名空间，避免目录名异常。"""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(namespace or "").strip()).strip("._-")
    return token or "default"


def _resolve_cache_dir(base_dir: str, namespace: str) -> Path:
    """
    兼容历史目录结构：
    - default 命名空间沿用旧目录（不新增子目录），避免历史缓存失效。
    - 非 default 命名空间使用子目录隔离。
    """
    ns = _normalize_namespace(namespace)
    base = Path(base_dir)
    return base if ns == "default" else base / ns


def _config_hash(config_dict) -> str:
    """使用 dataclass dict 计算 MD5 hash"""
    if hasattr(config_dict, '__dict__'):
        config_dict = asdict(config_dict)
    config_str = str(sorted(config_dict.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


class LabelCache:
    """标签缓存管理"""
    
    def __init__(self, cache_dir: str = "data/cache/labels", enable: bool = True, namespace: str = "default"):
        self.namespace = _normalize_namespace(namespace)
        self.cache_dir = _resolve_cache_dir(cache_dir, self.namespace)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable = enable
        self.hits = 0
        self.misses = 0
    
    def get_cache_path(self, symbol: str, config_hash: str) -> Path:
        """生成缓存文件路径"""
        filename = f"{symbol}_{config_hash}.parquet"
        return self.cache_dir / filename
    
    def load(self, symbol: str, config_hash: str) -> Optional[List[Dict]]:
        """
        从缓存加载标签
        Returns: List[Dict] or None if not found
        """
        if not self.enable:
            return None
        
        cache_path = self.get_cache_path(symbol, config_hash)
        if not cache_path.exists():
            self.misses += 1
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            labels = df.to_dict(orient='records')
            self.hits += 1
            logger.debug(f"Label cache HIT: {symbol} ({len(labels)} records)")
            return labels
        except Exception as e:
            logger.warning(f"Failed to load label cache for {symbol}: {e}")
            self.misses += 1
            return None
    
    def save(self, symbol: str, config_hash: str, labels: List[Dict]) -> bool:
        """
        保存标签到缓存
        Returns: bool indicating success
        """
        if not self.enable or not labels:
            return False
        
        cache_path = self.get_cache_path(symbol, config_hash)
        try:
            df = pd.DataFrame(labels)
            df.to_parquet(cache_path, compression='snappy', index=False)
            logger.debug(f"Label cache SAVED: {symbol} ({len(labels)} records)")
            return True
        except Exception as e:
            logger.warning(f"Failed to save label cache for {symbol}: {e}")
            return False
    
    def clear_all(self):
        """清空所有缓存"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared all label cache in {self.cache_dir}")
    
    def stats(self) -> Dict:
        """缓存统计"""
        total_files = len(list(self.cache_dir.glob("*.parquet")))
        total_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet")) / (1024 * 1024)
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_mb": f"{total_size_mb:.2f}",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate*100:.1f}%",
        }


class FeatureCache:
    """特征缓存管理"""
    
    def __init__(self, cache_dir: str = "data/cache/features", enable: bool = True, namespace: str = "default"):
        self.namespace = _normalize_namespace(namespace)
        self.cache_dir = _resolve_cache_dir(cache_dir, self.namespace)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable = enable
        self.hits = 0
        self.misses = 0
    
    def get_cache_path(self, symbol: str, config_hash: str) -> Path:
        """生成缓存文件路径"""
        filename = f"{symbol}_{config_hash}.parquet"
        return self.cache_dir / filename
    
    def load(self, symbol: str, config_hash: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载特征
        Returns: pd.DataFrame or None if not found
        """
        if not self.enable:
            return None
        
        cache_path = self.get_cache_path(symbol, config_hash)
        if not cache_path.exists():
            self.misses += 1
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            self.hits += 1
            logger.debug(f"Feature cache HIT: {symbol} ({len(df)} rows, {len(df.columns)} cols)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load feature cache for {symbol}: {e}")
            self.misses += 1
            return None
    
    def save(self, symbol: str, config_hash: str, features: pd.DataFrame) -> bool:
        """
        保存特征到缓存
        Returns: bool indicating success
        """
        if not self.enable or features is None or len(features) == 0:
            return False
        
        cache_path = self.get_cache_path(symbol, config_hash)
        try:
            features.to_parquet(cache_path, compression='snappy', index=False)
            logger.debug(f"Feature cache SAVED: {symbol} ({len(features)} rows)")
            return True
        except Exception as e:
            logger.warning(f"Failed to save feature cache for {symbol}: {e}")
            return False
    
    def clear_all(self):
        """清空所有缓存"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared all feature cache in {self.cache_dir}")
    
    def stats(self) -> Dict:
        """缓存统计"""
        total_files = len(list(self.cache_dir.glob("*.parquet")))
        total_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet")) / (1024 * 1024)
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_mb": f"{total_size_mb:.2f}",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate*100:.1f}%",
        }


# Singleton instances (per namespace)
_label_caches: Dict[str, LabelCache] = {}
_feature_caches: Dict[str, FeatureCache] = {}


def get_label_cache(enable: bool = True, namespace: str = "default") -> LabelCache:
    """获取全局标签缓存实例（按namespace隔离）。"""
    ns = _normalize_namespace(namespace)
    cache = _label_caches.get(ns)
    if cache is None:
        cache = LabelCache(enable=enable, namespace=ns)
        _label_caches[ns] = cache
    cache.enable = bool(enable)
    return cache


def get_feature_cache(enable: bool = True, namespace: str = "default") -> FeatureCache:
    """获取全局特征缓存实例（按namespace隔离）。"""
    ns = _normalize_namespace(namespace)
    cache = _feature_caches.get(ns)
    if cache is None:
        cache = FeatureCache(enable=enable, namespace=ns)
        _feature_caches[ns] = cache
    cache.enable = bool(enable)
    return cache


def clear_cache_namespace(namespace: str = "default"):
    """清空指定命名空间缓存。"""
    ns = _normalize_namespace(namespace)
    for base_dir in ["data/cache/labels", "data/cache/features"]:
        target = _resolve_cache_dir(base_dir, ns)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    _label_caches.pop(ns, None)
    _feature_caches.pop(ns, None)
    logger.info(f"Cleared cache namespace: {ns}")


def clear_all_caches():
    """清空所有缓存命名空间。"""
    for base_dir in [Path("data/cache/labels"), Path("data/cache/features")]:
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    _label_caches.clear()
    _feature_caches.clear()
    logger.info("All caches cleared")


def cache_stats(namespace: str = "default") -> Dict:
    """获取缓存统计（默认仅统计当前命名空间）。"""
    ns = _normalize_namespace(namespace)
    return {
        "namespace": ns,
        "labels": get_label_cache(namespace=ns).stats(),
        "features": get_feature_cache(namespace=ns).stats(),
    }
