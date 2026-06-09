"""Tests for DashboardApp — Phase 10."""

import pytest
from dashboard.app import DashboardApp, create_app


class TestDashboardApp:
    def test_create_default(self):
        app = DashboardApp()
        assert app.title == "XAU Trading System"
        assert app.page_names == []

    def test_register_page(self):
        app = DashboardApp()
        app.register_page("Test", lambda: {"page": "Test"})
        assert "Test" in app.page_names
        assert app.get_page("Test")() == {"page": "Test"}

    def test_get_nonexistent_page(self):
        app = DashboardApp()
        assert app.get_page("missing") is None


class TestCreateApp:
    def test_default_pages(self):
        app = create_app()
        assert app.title == "XAU Trading System"
        names = app.page_names
        assert "Signals" in names
        assert "Backtest" in names
        assert "Walk-Forward" in names
        assert "Journal" in names
        assert "Health" in names
        assert len(names) == 5

    def test_custom_title(self):
        app = create_app({"title": "My Dashboard"})
        assert app.title == "My Dashboard"

    def test_all_pages_render(self):
        app = create_app()
        for name in app.page_names:
            result = app.get_page(name)()
            assert result["page"] == name
            assert "summary" in result
