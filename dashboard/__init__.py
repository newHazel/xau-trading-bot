"""Dashboard — Phase 10: Streamlit-based monitoring UI."""

from dashboard.app import create_app
from dashboard.components.charts import plot_equity_curve, plot_drawdown_curve, plot_trade_scatter
from dashboard.components.filters import apply_filters, FilterConfig
from dashboard.components.export_csv import export_dataframe

__all__ = [
    "create_app",
    "plot_equity_curve",
    "plot_drawdown_curve",
    "plot_trade_scatter",
    "apply_filters",
    "FilterConfig",
    "export_dataframe",
]
