"""
特征缓存包装器
支持 per-symbol 缓存 + batch 合并
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd

from .engine import FeatureEngine
from ..cache_layer import get_feature_cache, _config_hash

logger = logging.getLogger(__name__)


class CachedFeatureEngine(FeatureEngine):
    """支持缓存的 FeatureEngine 包装"""
    
    def __init__(self, config, enable_cache: bool = True, cache_namespace: str = "default"):
        super().__init__(config)
        self.enable_cache = enable_cache
        self.cache_namespace = str(cache_namespace or "default")
        self.feature_cache = get_feature_cache(
            enable=enable_cache,
            namespace=self.cache_namespace,
        )
        self.config_hash = _config_hash(config)
    
    def transform_with_cache(
        self,
        samples: List[Dict],
        bars_by_symbol: Dict[str, List[Dict]],
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        带缓存的 transform 实现:
        1. 按 symbol 分离 samples
        2. 对每个 symbol, 尝试从缓存加载
        3. 未缓存的 symbol 进行计算
        4. 合并所有结果
        """
        # 按 symbol 分组 samples
        sample_df = self._build_sample_df(samples)
        sample_df = sample_df.reset_index(drop=True)
        symbol_frames = {
            symbol: sdf.sort_values("t0_ts")
            for symbol, sdf in sample_df.groupby("symbol", sort=False)
        }
        bar_df_map = self._build_symbol_bar_frame(bars_by_symbol)
        
        # 分类: cache hit vs. cache miss
        symbols_to_compute = []
        cached_results = {}  # symbol -> (feature_index, feature_rows)
        
        for symbol in symbol_frames:
            cached_df = self.feature_cache.load(symbol, self.config_hash)
            if cached_df is not None:
                symbol_sample_df = symbol_frames[symbol]
                if len(cached_df) != len(symbol_sample_df):
                    logger.warning(
                        "Feature cache row mismatch: %s cache=%d sample=%d, fallback compute",
                        symbol,
                        len(cached_df),
                        len(symbol_sample_df),
                    )
                    symbols_to_compute.append(symbol)
                    continue

                # 缓存命中：使用当前样本在全局DataFrame中的唯一行位置，
                # 避免重复RangeIndex导致下游标准化出现标签扩容。
                cached_results[symbol] = (
                    symbol_sample_df.index.tolist(),
                    cached_df.to_dict(orient='records'),
                )
                logger.info(f"Feature cache HIT: {symbol} ({len(cached_df)} rows)")
            else:
                symbols_to_compute.append(symbol)
        
        # 计算未缓存的 symbols
        computed_results = {}  # symbol -> (feature_index, feature_rows)
        
        if symbols_to_compute:
            workers = max(1, int(getattr(self.config, "symbol_workers", 1)))
            workers = min(workers, len(symbols_to_compute)) if symbols_to_compute else 1
            
            if workers <= 1:
                for symbol in symbols_to_compute:
                    sdf = symbol_frames[symbol]
                    idx, rows = self._transform_symbol_samples(symbol, sdf, bar_df_map.get(symbol))
                    computed_results[symbol] = (idx, rows)
                    
                    # 保存到缓存
                    feature_df = pd.DataFrame(rows, index=idx).sort_index()
                    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
                    feature_df = feature_df.fillna(0.0)
                    self.feature_cache.save(symbol, self.config_hash, feature_df)
                    logger.info(f"Feature cache SAVED: {symbol} ({len(feature_df)} rows)")
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {
                        symbol: ex.submit(
                            self._transform_symbol_samples,
                            symbol,
                            symbol_frames[symbol],
                            bar_df_map.get(symbol),
                        )
                        for symbol in symbols_to_compute
                    }
                    for symbol, fut in futures.items():
                        idx, rows = fut.result()
                        computed_results[symbol] = (idx, rows)
                        
                        # 保存到缓存
                        feature_df = pd.DataFrame(rows, index=idx).sort_index()
                        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
                        feature_df = feature_df.fillna(0.0)
                        self.feature_cache.save(symbol, self.config_hash, feature_df)
                        logger.info(f"Feature cache SAVED: {symbol} ({len(feature_df)} rows)")
        
        # 合并缓存和计算结果
        feature_rows: List[Dict[str, float]] = []
        feature_index: List[pd.Timestamp] = []
        
        for symbol in symbol_frames:
            if symbol in cached_results:
                idx, rows = cached_results[symbol]
            elif symbol in computed_results:
                idx, rows = computed_results[symbol]
            else:
                continue
            
            feature_index.extend(idx)
            feature_rows.extend(rows)
        
        # 构建最终 DataFrame
        feature_df = pd.DataFrame(feature_rows, index=feature_index).sort_index()
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0.0)
        
        if normalize and len(feature_df) > 0:
            feature_df = self.normalizer.fit_transform(feature_df, ordered_index=feature_df.index)
        
        return feature_df
    
    def transform(
        self,
        samples: List[Dict],
        bars_by_symbol: Dict[str, List[Dict]],
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        主入口: 根据 enable_cache 选择是否使用缓存路径
        """
        if self.enable_cache:
            return self.transform_with_cache(samples, bars_by_symbol, normalize)
        else:
            # 回退到原始实现
            return super().transform(samples, bars_by_symbol, normalize)
