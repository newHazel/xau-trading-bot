"""Streamlit runner — Phase 10: `streamlit run dashboard/streamlit_app.py`."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

from dashboard.data_loader import (
    load_signals, load_backtest_results, load_walk_forward,
    load_journal, load_health, get_table_counts,
)
from dashboard.components.charts import plot_equity_curve, plot_drawdown_curve
from dashboard.components.filters import FilterConfig, apply_filters
from dashboard.components.export_csv import export_dataframe

DB_PATH = str(Path(__file__).parent.parent / "data" / "database" / "trading_bot.sqlite")


def main():
    st.set_page_config(page_title="XAU Trading System", layout="wide")
    st.title("XAU Trading System")

    if not Path(DB_PATH).exists():
        st.error(f"Database not found: {DB_PATH}")
        st.info("Run the trading bot at least once to create the database.")
        return

    counts = get_table_counts(DB_PATH)
    page = st.sidebar.selectbox("Page", ["Overview", "Signals", "Backtest", "Walk-Forward", "Journal", "Health"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**DB Records**")
    for table, count in counts.items():
        st.sidebar.text(f"{table}: {count}")

    if page == "Overview":
        _render_overview(counts)
    elif page == "Signals":
        _render_signals()
    elif page == "Backtest":
        _render_backtest()
    elif page == "Walk-Forward":
        _render_walk_forward()
    elif page == "Journal":
        _render_journal()
    elif page == "Health":
        _render_health()


def _render_overview(counts):
    st.subheader("System Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signals", counts.get("signals", 0))
    col2.metric("Trades", counts.get("trades", 0))
    col3.metric("Experiments", counts.get("experiments", 0))
    col4.metric("Candles", counts.get("candles", 0))

    health = load_health(DB_PATH, check_limit=10)
    state_color = {"healthy": "🟢", "degraded": "🟡", "error": "🔴"}.get(health.system_state, "⚪")
    st.markdown(f"### System State: {state_color} {health.system_state.upper()}")

    if counts.get("trades", 0) > 0:
        journal = load_journal(DB_PATH)
        closed = [e for e in journal.entries if e.net_r is not None]
        if closed:
            rs = [e.net_r for e in closed]
            wins = sum(1 for r in rs if r > 0)
            st.markdown(f"**Closed trades:** {len(closed)} | **Win rate:** {wins/len(closed)*100:.1f}% | **Total R:** {sum(rs):.2f}")


def _render_signals():
    st.subheader("Signals")
    data = load_signals(DB_PATH)
    if data.total == 0:
        st.info("No signals in database yet.")
        return

    summary = data.get_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", summary["total"])
    col2.metric("Grades", ", ".join(f"{k}:{v}" for k, v in summary["by_grade"].items()))
    col3.metric("Status", ", ".join(f"{k}:{v}" for k, v in summary["by_status"].items()))

    with st.expander("Filters"):
        grades = st.multiselect("Grade", ["A+", "A", "B", "C", "D"])
        directions = st.multiselect("Direction", ["LONG", "SHORT"])

    records = data.to_records()
    fc = FilterConfig(
        grades=grades if grades else None,
        directions=directions if directions else None,
    )
    filtered = apply_filters(records, fc)
    st.dataframe(pd.DataFrame(filtered), use_container_width=True)

    csv = export_dataframe(filtered)
    if csv:
        st.download_button("Export CSV", csv, "signals.csv", "text/csv")


def _render_backtest():
    st.subheader("Backtest Results")
    data = load_backtest_results(DB_PATH)
    if data.total_experiments == 0:
        st.info("No experiments in database yet.")
        return

    summary = data.get_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("Experiments", summary["total_experiments"])
    col2.metric("Avg Win Rate", f"{summary['avg_win_rate']*100:.1f}%")
    col3.metric("Best Total R", f"{summary['best_total_r']:.2f}")

    st.dataframe(pd.DataFrame(data.to_records()), use_container_width=True)


def _render_walk_forward():
    st.subheader("Walk-Forward Validation")
    data = load_walk_forward(DB_PATH)
    if data.total_folds == 0:
        st.info("No walk-forward runs in database yet.")
        return

    summary = data.get_summary()
    passed_icon = "✅" if summary["overall_passed"] else "❌"
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Folds", summary["total_folds"])
    col2.metric("Passed", f"{summary['passed_folds']} / {summary['total_folds']}")
    col3.metric("Overall", f"{passed_icon} {'PASSED' if summary['overall_passed'] else 'FAILED'}")

    st.dataframe(pd.DataFrame(data.to_records()), use_container_width=True)


def _render_journal():
    st.subheader("Trade Journal")
    data = load_journal(DB_PATH)
    if data.total == 0:
        st.info("No trades in database yet.")
        return

    summary = data.get_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", summary["total"])
    col2.metric("Avg Net R", f"{summary['avg_net_r']:.2f}")
    col3.metric("Violations", summary["violation_count"])
    results_str = ", ".join(f"{k}:{v}" for k, v in summary["by_result"].items())
    col4.metric("Results", results_str)

    closed = [e for e in data.entries if e.net_r is not None]
    if closed:
        rs = [e.net_r for e in closed]
        st.markdown("### Equity Curve")
        eq = plot_equity_curve(rs)
        eq_df = pd.DataFrame(eq["points"])
        st.line_chart(eq_df.set_index("trade_index")["cumulative_r"])

        st.markdown("### Drawdown")
        dd = plot_drawdown_curve(rs)
        dd_df = pd.DataFrame(dd["points"])
        st.area_chart(dd_df.set_index("trade_index")["drawdown_r"])

    with st.expander("Filters"):
        results_filter = st.multiselect("Result", ["win", "loss", "breakeven", "open"])
        grades_filter = st.multiselect("Grade", ["A+", "A", "B", "C", "D"], key="j_grade")

    records = data.to_records()
    fc = FilterConfig(
        statuses=results_filter if results_filter else None,
        grades=grades_filter if grades_filter else None,
    )
    filtered = apply_filters(records, fc)
    st.dataframe(pd.DataFrame(filtered), use_container_width=True)

    csv = export_dataframe(filtered)
    if csv:
        st.download_button("Export CSV", csv, "journal.csv", "text/csv")


def _render_health():
    st.subheader("System Health")
    data = load_health(DB_PATH)

    summary = data.get_summary()
    state_color = {"healthy": "🟢", "degraded": "🟡", "error": "🔴"}.get(summary["system_state"], "⚪")
    col1, col2, col3 = st.columns(3)
    col1.metric("State", f"{state_color} {summary['system_state'].upper()}")
    col2.metric("Failed Checks", summary["failed_checks"])
    col3.metric("Warnings", summary["warned_checks"])

    if data.checks:
        st.markdown("### Recent Health Checks")
        st.dataframe(pd.DataFrame(data.checks_to_records()), use_container_width=True)
    else:
        st.info("No health check records yet.")


if __name__ == "__main__":
    main()
