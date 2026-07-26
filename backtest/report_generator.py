"""HTML report generator for backtest results.

Generates chart_data.json and report.html from backtest artifacts.
Uses a template file at backtest/templates/report_template.html.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.models import EquitySnapshot, TradeRecord


def generate_html_report(
    out_dir: Path,
    config: Dict[str, Any],
    trades: List[TradeRecord],
    equity_snapshots: List[EquitySnapshot],
    target_pos: pd.DataFrame,
    data_map: Dict[str, pd.DataFrame],
    metrics: Dict[str, Any],
    codes: List[str],
) -> None:
    """Generate chart_data.json and report.html in the artifacts directory.

    Args:
        out_dir: Artifacts output directory (e.g. run_dir/artifacts).
        config: Backtest configuration dict.
        trades: Completed round-trip trades.
        equity_snapshots: Equity time series snapshots.
        target_pos: Target position weights DataFrame (dates x symbols).
        data_map: code -> OHLCV DataFrame.
        metrics: Metrics dictionary from calc_metrics().
        codes: List of symbol codes used in the backtest.
    """
    # 1. Generate chart_data.json
    chart_data = _build_chart_data(
        trades=trades,
        equity_snapshots=equity_snapshots,
        target_pos=target_pos,
        data_map=data_map,
        codes=codes,
        metrics=metrics,
    )
    chart_data_path = out_dir / "chart_data.json"
    chart_data_path.write_text(
        json.dumps(chart_data, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 2. Generate report.html from template
    template_path = Path(__file__).parent / "templates" / "report_template.html"
    if not template_path.exists():
        return  # no template, skip report generation

    template = template_path.read_text(encoding="utf-8")

    # Determine primary symbol
    primary_symbol = codes[0] if codes else "UNKNOWN"

    # Build title and subtitle
    title = f"回测报告 — {primary_symbol}"
    start_date = str(config.get("start_date", ""))
    end_date = str(config.get("end_date", ""))
    commission_rate = float(config.get("us_commission", 0.0))
    slippage_rate = float(config.get("slippage_us", 0.0005))
    # HK market: sum all HK fees (commission + stamp_tax + levy + settlement)
    if any(".HK" in c.upper() for c in codes):
        hk_total = (
            float(config.get("hk_commission", 0.00015))
            + float(config.get("hk_stamp_tax", 0.001))
            + float(config.get("hk_levy", 0.0000565))
            + float(config.get("hk_settlement", 0.00002))
        )
        commission_rate = hk_total
        slippage_rate = float(config.get("slippage_hk", 0.001))
    # A-share market: commission + transfer_fee + stamp_tax (sell-only, averaged)
    elif any(c.upper().endswith((".SZ", ".SH", ".BJ")) for c in codes):
        a_comm = float(config.get("commission_rate", 0.00025))
        a_transfer = float(config.get("transfer_fee", 0.00001))
        a_stamp = float(config.get("stamp_tax", 0.0005))
        commission_rate = a_comm + a_transfer + a_stamp * 0.5  # stamp tax sell-only, averaged
        slippage_rate = float(config.get("slippage", 0.001))
    subtitle = (
        f"{primary_symbol} | {start_date} ~ {end_date} | "
        f"手续费: {commission_rate * 100:.1f}% | 滑点: {slippage_rate * 100:.2f}%"
    )

    # Build metrics JSON array
    metrics_json = _build_metrics_json(metrics)

    # Fill template placeholders
    html = template
    html = html.replace("__TITLE__", title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__SYMBOL__", primary_symbol)
    html = html.replace("__METRICS_JSON__", json.dumps(metrics_json, ensure_ascii=False))
    # Embed chart data directly into HTML (avoids fetch issues with file:// protocol)
    html = html.replace("__CHART_DATA_JSON__", json.dumps(chart_data, ensure_ascii=False, default=str))
    html = html.replace("__COMMISSION_RATE__", str(commission_rate))
    html = html.replace("__SLIPPAGE_RATE__", str(slippage_rate))

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")


def _build_chart_data(
    trades: List[TradeRecord],
    equity_snapshots: List[EquitySnapshot],
    target_pos: pd.DataFrame,
    data_map: Dict[str, pd.DataFrame],
    codes: List[str],
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the chart_data.json structure from backtest results.

    Returns:
        Dict with keys: buy_opens, sell_opens, buy_closes, sell_closes,
        positions, prices, equity, and optionally validation/comparison data.
    """
    # Trade events
    buy_opens: List[Dict[str, Any]] = []    # long entries
    sell_opens: List[Dict[str, Any]] = []   # short entries
    buy_closes: List[Dict[str, Any]] = []   # short exits (buy to cover)
    sell_closes: List[Dict[str, Any]] = []  # long exits (sell to close)

    for t in trades:
        entry_date = _fmt_date(t.entry_time)
        exit_date = _fmt_date(t.exit_time)
        holding_days = (t.exit_time - t.entry_time).days

        if t.direction == 1:  # Long trade
            buy_opens.append({
                "date": entry_date,
                "open_price": round(t.entry_price, 4),
                "size": t.size,
            })
            sell_closes.append({
                "date": exit_date,
                "close_price": round(t.exit_price, 4),
                "pnl": round(t.pnl, 4),
                "return_pct": round(t.pnl_pct, 2),
                "holding_days": holding_days,
                "size": t.size,
            })
        else:  # Short trade (direction == -1)
            sell_opens.append({
                "date": entry_date,
                "open_price": round(t.entry_price, 4),
                "size": t.size,
            })
            buy_closes.append({
                "date": exit_date,
                "close_price": round(t.exit_price, 4),
                "pnl": round(t.pnl, 4),
                "return_pct": round(t.pnl_pct, 2),
                "holding_days": holding_days,
                "size": t.size,
            })

    # Sort all trade arrays by date
    buy_opens.sort(key=lambda x: x["date"])
    sell_opens.sort(key=lambda x: x["date"])
    buy_closes.sort(key=lambda x: x["date"])
    sell_closes.sort(key=lambda x: x["date"])

    # Prices from primary symbol
    primary = codes[0] if codes else None
    prices: List[Dict[str, Any]] = []
    if primary and primary in data_map:
        df = data_map[primary]
        for ts, row in df.iterrows():
            prices.append({
                "date": _fmt_date(ts),
                "open": round(float(row.get("open", 0)), 4),
                "high": round(float(row.get("high", 0)), 4),
                "low": round(float(row.get("low", 0)), 4),
                "close": round(float(row.get("close", 0)), 4),
                "volume": int(row.get("volume", 0)) if "volume" in row.index else 0,
            })

    # Positions from target_pos (sum across all symbols per date)
    positions: List[Dict[str, Any]] = []
    if target_pos is not None and len(target_pos) > 0:
        total_pos = target_pos.sum(axis=1)
        for ts in target_pos.index:
            positions.append({
                "date": _fmt_date(ts),
                "pos": round(float(total_pos.loc[ts]), 4),
            })

    # Equity curve with drawdown
    equity: List[Dict[str, Any]] = []
    if equity_snapshots:
        eq_values = [s.equity for s in equity_snapshots]
        peak = eq_values[0]
        for s in equity_snapshots:
            peak = max(peak, s.equity)
            dd = (s.equity - peak) / peak if peak > 0 else 0.0
            equity.append({
                "date": _fmt_date(s.timestamp),
                "equity": round(s.equity, 2),
                "drawdown": round(dd, 4),
            })

    result = {
        "buy_opens": buy_opens,
        "sell_opens": sell_opens,
        "buy_closes": buy_closes,
        "sell_closes": sell_closes,
        "positions": positions,
        "prices": prices,
        "equity": equity,
    }

    # Add validation data if available
    if metrics:
        validation = metrics.get("validation", {})
        if validation:
            result["validation"] = validation
        
        comparison = metrics.get("comparison", {})
        if comparison:
            result["comparison"] = comparison

        # Add periodic breakdown (yearly/monthly) if available
        by_year = metrics.get("by_year", {})
        if by_year:
            result["by_year"] = by_year

        by_month = metrics.get("by_month", {})
        if by_month:
            result["by_month"] = by_month

        # Add rolling monitor data if available
        rolling = metrics.get("rolling", {})
        if rolling:
            result["rolling"] = rolling

    return result


def _build_metrics_json(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the metrics card array for the HTML report.

    Returns:
        List of dicts with keys: label, value, cls.
    """
    total_ret = metrics.get("total_return", 0)
    ann_ret = metrics.get("annual_return", 0)
    max_dd = metrics.get("max_drawdown", 0)
    sharpe = metrics.get("sharpe", 0)
    calmar = metrics.get("calmar", 0)
    sortino = metrics.get("sortino", 0)
    win_rate = metrics.get("win_rate", 0)
    pl_ratio = metrics.get("profit_loss_ratio", 0)
    profit_factor = metrics.get("profit_factor", 0)
    trade_count = metrics.get("trade_count", 0)
    avg_holding = metrics.get("avg_holding_days", 0)
    final_value = metrics.get("final_value", 0)

    return [
        {
            "label": "总收益率",
            "value": f"{'+' if total_ret >= 0 else ''}{total_ret * 100:.2f}%",
            "cls": "pos" if total_ret >= 0 else "neg",
        },
        {
            "label": "年化收益率",
            "value": f"{'+' if ann_ret >= 0 else ''}{ann_ret * 100:.2f}%",
            "cls": "pos" if ann_ret >= 0 else "neg",
        },
        {
            "label": "最大回撤",
            "value": f"{max_dd * 100:.2f}%",
            "cls": "neg",
        },
        {
            "label": "夏普比率",
            "value": f"{sharpe:.2f}",
            "cls": "neutral",
        },
        {
            "label": "卡尔玛比率",
            "value": f"{calmar:.2f}",
            "cls": "neutral",
        },
        {
            "label": "索提诺比率",
            "value": f"{sortino:.2f}",
            "cls": "neutral",
        },
        {
            "label": "胜率",
            "value": f"{win_rate * 100:.1f}%",
            "cls": "neutral",
        },
        {
            "label": "盈亏比",
            "value": f"{pl_ratio:.2f}",
            "cls": "pos" if pl_ratio > 1 else "neg",
        },
        {
            "label": "利润因子",
            "value": f"{profit_factor:.2f}",
            "cls": "pos" if profit_factor > 1 else "neg",
        },
        {
            "label": "交易次数",
            "value": str(trade_count),
            "cls": "neutral",
        },
        {
            "label": "平均持仓天数",
            "value": f"{avg_holding:.1f}",
            "cls": "neutral",
        },
        {
            "label": "最终净值",
            "value": f"${final_value:,.0f}",
            "cls": "pos" if final_value >= 1_000_000 else "neg",
        },
    ]


def _fmt_date(ts) -> str:
    """Format a pandas Timestamp to yyyy-mm-dd string."""
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]
