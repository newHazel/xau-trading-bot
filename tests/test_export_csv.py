"""Tests for CSV export — Phase 10.2."""

import pytest
import os
import tempfile
from dashboard.components.export_csv import export_dataframe, export_to_file, get_csv_summary


RECORDS = [
    {"setup_id": "XAU-001", "grade": "A+", "net_r": 2.0},
    {"setup_id": "XAU-002", "grade": "A", "net_r": -1.0},
    {"setup_id": "XAU-003", "grade": "B", "net_r": 1.5},
]


class TestExportDataframe:
    def test_empty(self):
        assert export_dataframe([]) == ""

    def test_basic(self):
        csv = export_dataframe(RECORDS)
        lines = csv.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "setup_id" in lines[0]
        assert "XAU-001" in lines[1]

    def test_custom_columns(self):
        csv = export_dataframe(RECORDS, columns=["setup_id", "net_r"])
        lines = csv.strip().split("\n")
        assert "grade" not in lines[0]
        assert "setup_id" in lines[0]

    def test_extra_columns_ignored(self):
        csv = export_dataframe(RECORDS, columns=["setup_id"])
        lines = csv.strip().split("\n")
        assert "grade" not in lines[0]


class TestExportToFile:
    def test_write_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            result = export_to_file(RECORDS, path)
            assert result == path
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "XAU-001" in content
        finally:
            os.unlink(path)

    def test_empty_records(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            result = export_to_file([], path)
            assert result == ""
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestGetCsvSummary:
    def test_empty(self):
        s = get_csv_summary("")
        assert s["rows"] == 0

    def test_basic(self):
        csv = export_dataframe(RECORDS)
        s = get_csv_summary(csv)
        assert s["rows"] == 3
        assert s["columns"] == 3
        assert s["size_bytes"] > 0
