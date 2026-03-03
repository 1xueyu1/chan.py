
import json
from typing import Dict, TypedDict

import xgboost as xgb

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from ChanModel.feature_center import build_features  # ✅ 改这里
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


# ==============================
# 预测函数（保持不变）
# ==============================
def predict_bsp(model: xgb.Booster, last_bsp: CBS_Point, meta: Dict[str, int]):
    missing = -9999999
    feature_arr = [missing] * len(meta)

    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value

    feature_arr = [feature_arr]
    dtest = xgb.DMatrix(feature_arr, missing=missing)

    return model.predict(dtest)


if __name__ == "__main__":

    code = "BTCUSDT"
    begin_time = "2024-12-04"
    end_time = "2026-02-07"
    data_src = DATA_SRC.CSV
    lv_list = [KL_TYPE.K_15M]

    config = CChanConfig({
        "trigger_step": True,
        "bi_strict": True,        # ✅ 保持和训练一致
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    model = xgb.Booster()
    model.load_model("Debug/model.json")
    meta = json.load(open("Debug/feature.meta", "r"))

    treated_bsp_idx = set()

    for chan_snapshot in chan.step_load():

        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        last_bsp = bsp_list[0]
        cur_lv_chan = chan_snapshot[0]

        # ✅ 完全对齐 demo5 触发逻辑
        if last_bsp.klu.idx in treated_bsp_idx:
            continue

        if cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        # ===============================
        # 🔥 关键修改：使用 build_features
        # ===============================
        extra_feat = build_features(
            klu=last_klu,
            history=cur_lv_chan.lst,
            chan=cur_lv_chan
        )

        last_bsp.features.add_feat(extra_feat)

        # ===============================
        # 预测
        # ===============================
        score = predict_bsp(model, last_bsp, meta)

        print(last_bsp.klu.time, score)

        treated_bsp_idx.add(last_bsp.klu.idx)