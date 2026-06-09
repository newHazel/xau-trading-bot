"""Main Streamlit app entry point — Phase 10."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DashboardApp:
    title: str = "XAU Trading System"
    pages: Dict[str, Any] = field(default_factory=dict)

    def register_page(self, name: str, render_fn) -> None:
        self.pages[name] = render_fn

    @property
    def page_names(self) -> list:
        return list(self.pages.keys())

    def get_page(self, name: str):
        return self.pages.get(name)


def create_app(config: Optional[Dict] = None) -> DashboardApp:
    config = config or {}
    app = DashboardApp(title=config.get("title", "XAU Trading System"))

    from dashboard.pages.signals_page import render_signals
    from dashboard.pages.backtest_page import render_backtest
    from dashboard.pages.walk_forward_page import render_walk_forward
    from dashboard.pages.journal_page import render_journal
    from dashboard.pages.health_page import render_health

    app.register_page("Signals", render_signals)
    app.register_page("Backtest", render_backtest)
    app.register_page("Walk-Forward", render_walk_forward)
    app.register_page("Journal", render_journal)
    app.register_page("Health", render_health)
    return app
