"""向量化技术指标计算层。

从 signal_engine.py 的 _compute_indicators 方法提取为独立纯函数。
所有函数接受 DataFrame 或 numpy 数组，返回增强后的 DataFrame。
无状态，可独立测试。

插件式管道架构:
- IndicatorCalculator: 指标计算器基类
- IndicatorPipeline: 可组合的指标管道
- 各独立 Calculator 类（ATR, ADX, SAR, EMA, RSI, MA/HighLow, Volume 等）
- compute_indicators(): 向后兼容的快捷函数
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IndicatorParams:
    """技术指标参数集合。"""

    atr_period: int = 14
    adx_period: int = 14
    adx_smooth: int = 6
    ema_fast: int = 12
    ema_slow: int = 26
    sar_accel: float = 0.02
    sar_max_accel: float = 0.2
    rsi_period: int = 14
    ma_period: int = 20
    atr_ma_period: int = 60


# ------------------------------------------------------------------
# 指标计算器基类
# ------------------------------------------------------------------


class IndicatorCalculator:
    """指标计算器基类。子类实现 compute() 方法。"""

    #: 此计算器产出的列名列表
    required_columns: list[str] = []

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        """计算指标并添加到 DataFrame。

        Args:
            df: 包含 OHLCV 列的 DataFrame。
            params: 指标参数集合。

        Returns:
            添加了新列的 DataFrame。
        """
        raise NotImplementedError

    def get_columns(self) -> list[str]:
        """返回此计算器产出的列名。"""
        return list(self.required_columns)


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR (Wilder 平滑)。"""
    n = len(high)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.empty(n)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr, tr


def _calc_adx(
    high: np.ndarray,
    low: np.ndarray,
    tr: np.ndarray,
    period: int,
    smooth: int,
) -> tuple:
    """ADX / +DI / -DI / ADXR (Wilder 原始算法)。"""
    n = len(high)
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    hd = high - prev_high
    ld = prev_low - low
    plus_dm = np.where((hd > 0) & (hd > ld), hd, 0.0)
    minus_dm = np.where((ld > 0) & (hd < ld), ld, 0.0)

    sum_tr = np.empty(n)
    sum_plus = np.empty(n)
    sum_minus = np.empty(n)
    sum_tr[:period] = np.nan
    sum_plus[:period] = np.nan
    sum_minus[:period] = np.nan
    sum_tr[period - 1] = np.sum(tr[:period])
    sum_plus[period - 1] = np.sum(plus_dm[:period])
    sum_minus[period - 1] = np.sum(minus_dm[:period])
    for i in range(period, n):
        sum_tr[i] = sum_tr[i - 1] - sum_tr[i - 1] / period + tr[i]
        sum_plus[i] = sum_plus[i - 1] - sum_plus[i - 1] / period + plus_dm[i]
        sum_minus[i] = sum_minus[i - 1] - sum_minus[i - 1] / period + minus_dm[i]

    pdi = np.divide(sum_plus * 100.0, sum_tr, out=np.zeros_like(sum_plus), where=sum_tr > 0)
    mdi = np.divide(sum_minus * 100.0, sum_tr, out=np.zeros_like(sum_minus), where=sum_tr > 0)
    di_sum = pdi + mdi
    dx = np.divide(
        np.abs(pdi - mdi) * 100.0,
        di_sum,
        out=np.zeros_like(pdi),
        where=di_sum > 0,
    )

    # ADX: Wilder 平滑 (SMA then EMA with alpha=1/smooth)
    adx = np.empty(n)
    adx[:] = np.nan
    first_valid = period - 1 + smooth - 1
    if first_valid < n:
        adx[first_valid] = np.nanmean(dx[period - 1 : first_valid + 1])
        alpha_adx = 1.0 / smooth
        for i in range(first_valid + 1, n):
            adx[i] = adx[i - 1] * (1 - alpha_adx) + dx[i] * alpha_adx

    # ADXR = (ADX + ADX[smooth bars ago]) / 2
    adxr = np.empty(n)
    adxr[:] = np.nan
    for i in range(first_valid + smooth, n):
        adxr[i] = (adx[i] + adx[i - smooth]) / 2.0

    return pdi, mdi, adx, adxr


def _calc_sar(
    high: np.ndarray, low: np.ndarray, accel: float, max_accel: float
) -> tuple:
    """Parabolic SAR。"""
    n = len(high)
    sar_val = np.empty(n)
    sar_dir = np.zeros(n)  # +1=上涨(bullish), -1=下跌(bearish)
    sar_val[0] = low[0]
    sar_dir[0] = 1
    af = accel
    ep = high[0]
    for i in range(1, n):
        prev_sar = sar_val[i - 1]
        if sar_dir[i - 1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(
                new_sar,
                low[max(i - 1, 0)],
                low[max(i - 2, 0)] if i >= 2 else low[0],
            )
            if low[i] < new_sar:
                sar_dir[i] = -1
                sar_val[i] = ep
                ep = low[i]
                af = accel
            else:
                sar_dir[i] = 1
                sar_val[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + accel, max_accel)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(
                new_sar,
                high[max(i - 1, 0)],
                high[max(i - 2, 0)] if i >= 2 else high[0],
            )
            if high[i] > new_sar:
                sar_dir[i] = 1
                sar_val[i] = ep
                ep = high[i]
                af = accel
            else:
                sar_dir[i] = -1
                sar_val[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + accel, max_accel)

    # SAR 翻转信号
    sar_rev = np.zeros(n)
    for i in range(1, n):
        if sar_dir[i] != sar_dir[i - 1]:
            sar_rev[i] = sar_dir[i]

    return sar_val, sar_dir, sar_rev


def _calc_ema(close: pd.Series, fast: int, slow: int) -> tuple:
    """EMA 快/慢线。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast, ema_slow


def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    """RSI (Wilder 平滑)。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def _calc_volume_features(volume: pd.Series) -> dict:
    """量能指标: vol_ma5, vol_ma10, vol_ma20, vol_ratio, vstd。"""
    vol_ma5 = volume.rolling(5).mean()
    vol_ma10 = volume.rolling(10).mean()
    vol_ma20 = volume.rolling(20).mean()
    vol_ratio = vol_ma5 / vol_ma10.replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(1.0)
    vstd = volume.rolling(20).std()
    vstd = vstd.fillna(0)
    return {
        "vol_ma5": vol_ma5,
        "vol_ma10": vol_ma10,
        "vol_ma20": vol_ma20,
        "vol_ratio": vol_ratio,
        "vstd": vstd,
    }


# ------------------------------------------------------------------
# 独立计算器实现
# ------------------------------------------------------------------


class ATRCalculator(IndicatorCalculator):
    """ATR (Average True Range) 计算器。

    输出列: atr, _tmp_tr（中间产物，供 ADXCalculator 使用）。
    """

    required_columns: list[str] = ["atr"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        high = d["high"].values
        low = d["low"].values
        close = d["close"].values
        atr, tr = _calc_atr(high, low, close, params.atr_period)
        d["atr"] = atr
        # 保留 True Range 供 ADXCalculator 使用，管道末尾会清理
        d["_tmp_tr"] = tr
        return d


class ADXCalculator(IndicatorCalculator):
    """ADX / +DI / -DI / ADXR 计算器。

    输出列: pdi, mdi, adx, adxr。
    依赖: df 中存在 tr 列（来自 ATRCalculator 的 _tmp_tr 或直接命名 tr）。
    """

    required_columns: list[str] = ["pdi", "mdi", "adx", "adxr"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        high = d["high"].values
        low = d["low"].values
        # 优先使用已有的 tr 列，否则自己计算
        if "_tmp_tr" in d.columns:
            tr = d["_tmp_tr"].values
        elif "tr" in d.columns:
            tr = d["tr"].values
        else:
            # 自行计算 True Range
            close = d["close"].values
            _, tr = _calc_atr(high, low, close, params.atr_period)
        pdi, mdi, adx, adxr = _calc_adx(high, low, tr, params.adx_period, params.adx_smooth)
        d["pdi"] = pdi
        d["mdi"] = mdi
        d["adx"] = adx
        d["adxr"] = adxr
        return d


class SARCalculator(IndicatorCalculator):
    """Parabolic SAR 计算器。

    输出列: sar, sar_dir, sar_rev。
    """

    required_columns: list[str] = ["sar", "sar_dir", "sar_rev"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        high = d["high"].values
        low = d["low"].values
        sar_val, sar_dir, sar_rev = _calc_sar(high, low, params.sar_accel, params.sar_max_accel)
        d["sar"] = sar_val
        d["sar_dir"] = sar_dir
        d["sar_rev"] = sar_rev
        return d


class EMACalculator(IndicatorCalculator):
    """EMA 快/慢线计算器。

    输出列: ema_fast, ema_slow。
    """

    required_columns: list[str] = ["ema_fast", "ema_slow"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        ema_fast, ema_slow = _calc_ema(d["close"], params.ema_fast, params.ema_slow)
        d["ema_fast"] = ema_fast
        d["ema_slow"] = ema_slow
        return d


class RSICalculator(IndicatorCalculator):
    """RSI 计算器。

    输出列: rsi。
    """

    required_columns: list[str] = ["rsi"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        d["rsi"] = _calc_rsi(d["close"], params.rsi_period)
        return d


class MAHighLowCalculator(IndicatorCalculator):
    """MA20 + 20日高点/低点计算器。

    输出列: ma20, high_20, low_20。
    """

    required_columns: list[str] = ["ma20", "high_20", "low_20"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        d["ma20"] = d["close"].rolling(params.ma_period).mean()
        d["high_20"] = d["high"].rolling(params.ma_period).max()
        d["low_20"] = d["low"].rolling(params.ma_period).min()
        return d


class VolumeCalculator(IndicatorCalculator):
    """量能指标计算器: vol_ma5, vol_ma10, vol_ma20, vol_ratio, vstd。

    输出列: vol_ma5, vol_ma10, vol_ma20, vol_ratio, vstd。
    """

    required_columns: list[str] = ["vol_ma5", "vol_ma10", "vol_ma20", "vol_ratio", "vstd"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        vol_feats = _calc_volume_features(d["volume"])
        for k, v in vol_feats.items():
            d[k] = v
        return d


class VolLevelCalculator(IndicatorCalculator):
    """波动率水平计算器: ATR 相对自身均线的比率。

    输出列: vol_level。
    依赖: df 中存在 atr 列（来自 ATRCalculator）。
    """

    required_columns: list[str] = ["vol_level"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        if "atr" in d.columns:
            atr_series = d["atr"].astype(float)
        else:
            # fallback: 自行计算 ATR
            high = d["high"].values
            low = d["low"].values
            close = d["close"].values
            atr_vals, _ = _calc_atr(high, low, close, params.atr_period)
            atr_series = pd.Series(atr_vals, index=d.index)
        atr_ma = atr_series.rolling(params.atr_ma_period, min_periods=20).mean()
        d["vol_level"] = (atr_series / atr_ma.replace(0, np.nan)).clip(0.3, 3.0).fillna(1.0)
        return d


class BollingerBandCalculator(IndicatorCalculator):
    """布林带宽度计算器: bb_width = 4 * std / ma。

    输出列: bb_width。
    """

    required_columns: list[str] = ["bb_width"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        # 需要 ma20，如果不存在则计算
        if "ma20" not in d.columns:
            d["ma20"] = d["close"].rolling(params.ma_period).mean()
        bb_std = d["close"].rolling(params.ma_period).std()
        d["bb_width"] = (4 * bb_std / d["ma20"].replace(0, np.nan)).fillna(0)
        return d


class MomentumCalculator(IndicatorCalculator):
    """多时间框架动量指标计算器。

    输出列: momentum_10, momentum_20, momentum_60。
    """

    required_columns: list[str] = ["momentum_10", "momentum_20", "momentum_60"]

    def compute(self, df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
        d = df.copy()
        d["momentum_10"] = d["close"].pct_change(10).fillna(0)
        d["momentum_20"] = d["close"].pct_change(20).fillna(0)
        d["momentum_60"] = d["close"].pct_change(60).fillna(0)
        return d


# ------------------------------------------------------------------
# 指标管道
# ------------------------------------------------------------------


class IndicatorPipeline:
    """可组合的指标管道。

    按顺序执行多个 IndicatorCalculator，每个计算器可以依赖前面计算器的输出。
    """

    def __init__(self, calculators: list[IndicatorCalculator]):
        self.calculators = calculators

    def compute(self, df: pd.DataFrame, params: IndicatorParams | None = None) -> pd.DataFrame:
        """执行管道中所有计算器，返回增强后的 DataFrame。

        Args:
            df: 包含 OHLCV 列的 DataFrame。
            params: 指标参数，为 None 时使用默认值。

        Returns:
            增强后的 DataFrame，包含所有计算器产出的列。
        """
        if params is None:
            params = IndicatorParams()
        d = df.copy()
        for calc in self.calculators:
            d = calc.compute(d, params)
        # 清理中间产物列
        for col in list(d.columns):
            if col.startswith("_tmp_"):
                d = d.drop(columns=[col])
        return d

    def get_all_columns(self) -> list[str]:
        """返回所有计算器产出的列名列表。"""
        cols = []
        for calc in self.calculators:
            cols.extend(calc.get_columns())
        return cols


# ------------------------------------------------------------------
# 默认管道工厂 + 向后兼容入口
# ------------------------------------------------------------------


def default_pipeline(params: IndicatorParams | None = None) -> IndicatorPipeline:
    """创建包含所有标准指标的默认管道。"""
    return IndicatorPipeline([
        ATRCalculator(),
        ADXCalculator(),
        SARCalculator(),
        EMACalculator(),
        RSICalculator(),
        MAHighLowCalculator(),
        VolumeCalculator(),
        VolLevelCalculator(),
        BollingerBandCalculator(),
        MomentumCalculator(),
    ])


def compute_indicators(
    df: pd.DataFrame,
    params: IndicatorParams | None = None,
) -> pd.DataFrame:
    """向后兼容的快捷函数，使用默认管道计算所有指标。

    Args:
        df: 包含 OHLCV 列的 DataFrame。
        params: 指标参数，为 None 时使用默认值。

    Returns:
        增强后的 DataFrame，新增列:
        atr, pdi, mdi, adx, adxr, sar, sar_dir, sar_rev,
        ema_fast, ema_slow, rsi, ma20, high_20,
        vol_ma5, vol_ma10, vol_ma20, vol_ratio, vstd。
    """
    return default_pipeline(params).compute(df, params)


def build_regime_features(d: pd.DataFrame, i: int) -> dict[str, float]:
    """从增强 DataFrame 第 i 行构建 RegimeDetector 特征字典。

    这是框架层与指标层之间的桥接点。
    默认实现使用 ADX/PDI/MDI/vol_level，用户可替换此函数
    来使用完全不同的指标构建特征。

    注: 在逐 bar 循环中建议使用 build_regime_features_fast() 代替，
    后者使用预提取的 numpy 数组，性能更好。

    Args:
        d: compute_indicators() 返回的增强 DataFrame。
        i: 当前 bar 索引。

    Returns:
        特征字典，键名与 DefaultRegimeDetector 对应:
        - trend_strength: 趋势强度（默认 ADX）
        - trend_up: 趋势方向（默认 PDI > MDI）
        - vol_level: 波动率水平（默认 ATR/MA60）
    """
    adx_val = float(d["adx"].iloc[i]) if not np.isnan(d["adx"].iloc[i]) else 0.0
    pdi_val = float(d["pdi"].iloc[i]) if not np.isnan(d["pdi"].iloc[i]) else 0.0
    mdi_val = float(d["mdi"].iloc[i]) if not np.isnan(d["mdi"].iloc[i]) else 0.0
    vol_val = float(d["vol_level"].iloc[i]) if "vol_level" in d.columns and not np.isnan(d["vol_level"].iloc[i]) else 1.0

    return {
        "trend_strength": adx_val,
        "trend_up": pdi_val > mdi_val,
        "vol_level": vol_val,
    }


def build_regime_features_fast(a, i: int) -> dict[str, float]:
    """从预提取的 numpy 数组命名空间构建 RegimeDetector 特征字典。

    与 build_regime_features() 功能相同，但使用 numpy 数组直接索引，
    避免 pandas .iloc 开销，在逐 bar 循环中性能更好。

    Args:
        a: build_arrays_ns() 返回的 SimpleNamespace。
        i: 当前 bar 索引。

    Returns:
        特征字典，键同 build_regime_features()。
    """
    adx_val = float(a.adx[i]) if not np.isnan(a.adx[i]) else 0.0
    pdi_val = float(a.pdi[i]) if not np.isnan(a.pdi[i]) else 0.0
    mdi_val = float(a.mdi[i]) if not np.isnan(a.mdi[i]) else 0.0
    vol_val = float(a.vol_level[i]) if not np.isnan(a.vol_level[i]) else 1.0

    return {
        "trend_strength": adx_val,
        "trend_up": pdi_val > mdi_val,
        "vol_level": vol_val,
        # 扩展特征（供新 Regime 检测器使用）
        "ema_fast": float(a.ema_fast[i]) if not np.isnan(a.ema_fast[i]) else 0.0,
        "ema_slow": float(a.ema_slow[i]) if not np.isnan(a.ema_slow[i]) else 0.0,
        "bb_width": float(a.bb_width[i]) if not np.isnan(a.bb_width[i]) else 0.0,
        "close": float(a.close[i]),
        "rsi": float(a.rsi[i]) if not np.isnan(a.rsi[i]) else 50.0,
        "ma20": float(a.ma20[i]) if not np.isnan(a.ma20[i]) else 0.0,
        # 动量特征（供 MomentumRegimeDetector 使用）
        "momentum_10": float(a.momentum_10[i]) if hasattr(a, "momentum_10") and not np.isnan(a.momentum_10[i]) else 0.0,
        "momentum_20": float(a.momentum_20[i]) if hasattr(a, "momentum_20") and not np.isnan(a.momentum_20[i]) else 0.0,
        "momentum_60": float(a.momentum_60[i]) if hasattr(a, "momentum_60") and not np.isnan(a.momentum_60[i]) else 0.0,
    }
