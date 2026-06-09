"""CSV Export — Phase 10.2: Export any page data to CSV format."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import csv
import io


def export_dataframe(records: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    if not records:
        return ""
    if columns is None:
        columns = list(records[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return output.getvalue()


def export_to_file(records: List[Dict[str, Any]], filepath: str, columns: Optional[List[str]] = None) -> str:
    csv_str = export_dataframe(records, columns)
    if not csv_str:
        return ""
    with open(filepath, "w", newline="") as f:
        f.write(csv_str)
    return filepath


def get_csv_summary(csv_content: str) -> Dict[str, Any]:
    if not csv_content:
        return {"rows": 0, "columns": 0, "size_bytes": 0}
    lines = csv_content.strip().split("\n")
    header = lines[0] if lines else ""
    col_count = len(header.split(",")) if header else 0
    return {
        "rows": max(0, len(lines) - 1),
        "columns": col_count,
        "size_bytes": len(csv_content.encode("utf-8")),
    }
