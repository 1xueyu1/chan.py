# RustCore adapter

Optional Rust acceleration layer for chan.py.

## Build

```powershell
cd D:\WorkSpace\python\chan-core-2026-final
.\build_python_extension.ps1
```

## Direct adapter usage

```python
from RustCore import RustChanEngine

engine = RustChanEngine("1m")
engine.push_bar("2024-10-01 09:30:00", 1.0, 1.1, 0.9, 1.05, 100)
print(engine.counts())
print(engine.latest_bi_bsp())
```

Existing `CKLine_Unit` lists can be pushed without converting them manually:

```python
engine = RustChanEngine.from_klus(klus)
print(engine.bi_bsp(latest_first=True))
```

## CChan switch

For single-level runs:

```python
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE

conf = CChanConfig({
    "use_rust_core": True,
    "trigger_step": False,
})

chan = CChan(
    code="BTCUSDT",
    data_src=DATA_SRC.CSV,
    lv_list=[KL_TYPE.K_1M],
    config=conf,
    autype=AUTYPE.NONE,
)

print(chan.get_rust_counts())
print(chan.get_latest_bsp())
```

Multi-level runs are also supported. Rust mode reads the smallest frequency as
the base data and lets `CMultiChan` aggregate higher levels:

```python
chan = CChan(
    code="BTCUSDT",
    data_src=DATA_SRC.CSV,
    lv_list=[KL_TYPE.K_15M, KL_TYPE.K_5M, KL_TYPE.K_1M],
    config=conf,
    autype=AUTYPE.NONE,
)

print(chan.get_rust_counts())             # all Rust levels
print(chan.get_rust_counts(KL_TYPE.K_5M)) # one Rust level
print(chan.get_latest_bsp(KL_TYPE.K_1M))
```

Rust BSPs are returned as `RustBSP` compatibility objects. They support common
attributes used by strategies:

```python
bsp = chan.get_latest_bsp(KL_TYPE.K_1M)[0]
print(bsp.klu.idx, bsp.klu.time, bsp.is_buy, bsp.type2str())
for name, value in bsp.features.items():
    print(name, value)
```

## ML and backtest usage

The backtest and ML dataset paths can opt into Rust without changing strategy
code:

```powershell
python Backtest\examples\run_vectorbt_backtest.py --use-rust-core
python -m ML.build_dataset --use-rust-core
```

The Rust step path uses a fast latest-BSP/counts check on every bar and only
builds the richer snapshot object when a new BSP event needs ML context
features.

Current constraints:

- `trigger_step=True` is supported through a lightweight `RustStepSnapshot`
  view for common strategy paths such as `snapshot[0][-1][-1]`,
  `snapshot[0][-2].idx`, `get_latest_bsp()`, `get_bsp()`, and Rust query
  helpers.
- Rust BSP, bi, seg, and zs compatibility objects cover the fields and methods
  used by the current ML feature extractor, including common BSP feature names
  and context feature methods.
- The step snapshot is still a lightweight view, not a full Python
  `CKLine_List` object graph; unsupported plotting or deep graph mutation code
  should remain on Python mode.

## Benchmark

```powershell
cd D:\WorkSpace\python\chan.py
python Debug\benchmark_rust_core.py --limits 500,2000,10000,50000 --rounds 3
python Debug\benchmark_rust_core.py --limits 500,2000 --rounds 3 --step-ml
```
