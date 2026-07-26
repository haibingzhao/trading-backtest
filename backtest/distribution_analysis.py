"""月度收益分布与回撤事件分析。

提供月度粒度的收益统计和单笔回撤事件识别，
用于深入理解策略的收益结构和风险特征。

Usage: called automatically by run_validation when config["validation"]["distribution"]
is present, or invoked directly on backtest outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownEvent:
    """单次回撤事件。

    Attributes:
        peak_date: 回撤前高点日期。
        trough_date: 回撤最低点日期。
        recovery_date: 恢复到前高的日期（None = 尚未恢复）。
        max_drawdown: 最大回撤幅度（负值）。
        duration_bars: 从峰值到恢复（或当前）的总 bar 数。
        recovery_bars: 从最低点到恢复的 bar 数（-1 = 尚未恢复）。
    """

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    max_drawdown: float
    duration_bars: int
    recovery_bars: int


def monthly_return_distribution(
    equity_curve: pd.Series,
) -> Dict[str, Any]:
    """计算月度收益分布统计。

    Args:
        equity_curve: 净值时间序列（index=timestamp, values=equity）。

    Returns:
        Dict 包含:
        - monthly_returns: 月收益率 Series
        - monthly_stats: 统计汇总 dict
        - return_matrix: 年×月收益率矩阵（用于热力图）
    """
    if len(equity_curve) < 2:
        return {"error": "need at least 2 bars", "monthly_stats": {}}

    # 月末重采样
    monthly_eq = equity_curve.resample("ME").last().dropna()
    if len(monthly_eq) < 2:
        return {"error": "need at least 2 months of data", "monthly_stats": {}}

    monthly_ret = monthly_eq.pct_change().dropna()

    if len(monthly_ret) == 0:
        return {"error": "no monthly returns computed", "monthly_stats": {}}

    # 基础统计
    n_months = len(monthly_ret)
    winning = monthly_ret[monthly_ret > 0]
    losing = monthly_ret[monthly_ret < 0]
    n_winning = len(winning)
    n_losing = len(losing)

    win_rate = n_winning / n_months if n_months > 0 else 0.0
    avg_win = float(winning.mean()) if n_winning > 0 else 0.0
    avg_loss = float(losing.mean()) if n_losing > 0 else 0.0
    profit_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-10 else 0.0

    gross_profit = float(winning.sum()) if n_winning > 0 else 0.0
    gross_loss = abs(float(losing.sum())) if n_losing > 0 else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    # 连续盈亏
    signs = (monthly_ret > 0).astype(int).values
    max_consec_win = _max_consecutive(signs, 1)
    max_consec_loss = _max_consecutive(signs, 0)

    # 年×月矩阵
    ret_df = monthly_ret.to_frame("return")
    ret_df["year"] = ret_df.index.year
    ret_df["month"] = ret_df.index.month
    return_matrix = ret_df.pivot_table(index="year", columns="month", values="return")

    monthly_stats = {
        "n_months": n_months,
        "n_winning": n_winning,
        "n_losing": n_losing,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "profit_loss_ratio": round(profit_loss_ratio, 4),
        "profit_factor": round(profit_factor, 4),
        "best_month": round(float(monthly_ret.max()), 6),
        "worst_month": round(float(monthly_ret.min()), 6),
        "median_month": round(float(monthly_ret.median()), 6),
        "skewness": round(float(monthly_ret.skew()), 4),
        "kurtosis": round(float(monthly_ret.kurtosis()), 4),
        "max_consecutive_wins": max_consec_win,
        "max_consecutive_losses": max_consec_loss,
    }

    # 序列化 monthly_returns
    monthly_returns_list = [
        {"date": str(d.date()), "return": round(float(v), 6)}
        for d, v in monthly_ret.items()
    ]

    return {
        "monthly_returns": monthly_returns_list,
        "monthly_stats": monthly_stats,
        "return_matrix": return_matrix.round(6).to_dict(),
    }


def identify_drawdown_events(
    equity_curve: pd.Series,
    min_depth: float = 0.05,
) -> Dict[str, Any]:
    """识别所有超过指定深度的回撤事件。

    Args:
        equity_curve: 净值时间序列。
        min_depth: 最小回撤深度（正值，如 0.05 = 5%）。

    Returns:
        Dict 包含:
        - events: DrawdownEvent 列表（按深度排序）
        - summary: 汇总统计
    """
    if len(equity_curve) < 3:
        return {"events": [], "summary": {}}

    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak.replace(0, 1)

    # 识别回撤区间：dd < -min_depth 的连续区间
    in_drawdown = dd < -min_depth
    if not in_drawdown.any():
        return {"events": [], "summary": {"n_events": 0}}

    # 找到回撤开始和结束的索引
    changes = in_drawdown.astype(int).diff().fillna(in_drawdown.iloc[0].astype(int))
    starts = changes[changes == 1].index.tolist()
    ends = changes[changes == -1].index.tolist()

    # 如果最后一个回撤未结束
    if in_drawdown.iloc[-1] and (not ends or starts[-1] > ends[-1]):
        ends.append(equity_curve.index[-1])

    events: List[DrawdownEvent] = []
    indices = equity_curve.index

    for start, end in zip(starts, ends):
        # 找到峰值点（回撤开始前的最高点）
        start_idx = indices.get_loc(start)
        if start_idx == 0:
            continue
        peak_idx = start_idx - 1
        peak_date = indices[peak_idx]

        # 回撤区间
        dd_slice = dd.loc[start:end]
        if dd_slice.empty:
            continue

        trough_idx_in_slice = dd_slice.idxmin()
        max_dd = float(dd_slice.min())

        # 恢复点：回撤结束后第一次恢复到 0 以上
        post_trough = dd.loc[trough_idx_in_slice:]
        recovered = post_trough[post_trough >= 0]
        if len(recovered) > 0:
            recovery_date = recovered.index[0]
            recovery_bars = len(dd.loc[trough_idx_in_slice:recovery_date]) - 1
        else:
            recovery_date = None
            recovery_bars = -1

        # 总持续 bar 数
        peak_loc = indices.get_loc(peak_date)
        end_loc = indices.get_loc(end)
        duration_bars = end_loc - peak_loc

        events.append(DrawdownEvent(
            peak_date=peak_date,
            trough_date=trough_idx_in_slice,
            recovery_date=recovery_date,
            max_drawdown=round(max_dd, 6),
            duration_bars=duration_bars,
            recovery_bars=recovery_bars,
        ))

    # 按深度排序（最深的在前）
    events.sort(key=lambda e: e.max_drawdown)

    # 汇总
    if events:
        depths = [e.max_drawdown for e in events]
        durations = [e.duration_bars for e in events]
        recovery_times = [e.recovery_bars for e in events if e.recovery_bars >= 0]
        summary = {
            "n_events": len(events),
            "max_drawdown": round(min(depths), 6),
            "avg_drawdown": round(float(np.mean(depths)), 6),
            "avg_duration_bars": round(float(np.mean(durations)), 1),
            "avg_recovery_bars": round(float(np.mean(recovery_times)), 1) if recovery_times else None,
            "max_recovery_bars": max(recovery_times) if recovery_times else None,
            "n_recovered": sum(1 for e in events if e.recovery_bars >= 0),
            "n_unrecovered": sum(1 for e in events if e.recovery_bars < 0),
        }
    else:
        summary = {"n_events": 0}

    # 序列化 events
    events_list = [
        {
            "peak_date": str(e.peak_date.date()) if hasattr(e.peak_date, "date") else str(e.peak_date),
            "trough_date": str(e.trough_date.date()) if hasattr(e.trough_date, "date") else str(e.trough_date),
            "recovery_date": str(e.recovery_date.date()) if e.recovery_date and hasattr(e.recovery_date, "date") else (str(e.recovery_date) if e.recovery_date else None),
            "max_drawdown": e.max_drawdown,
            "duration_bars": e.duration_bars,
            "recovery_bars": e.recovery_bars,
        }
        for e in events
    ]

    return {"events": events_list, "summary": summary}


def _max_consecutive(arr: np.ndarray, value: int) -> int:
    """计算数组中连续出现 value 的最大次数。"""
    max_count = 0
    count = 0
    for v in arr:
        if v == value:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count
