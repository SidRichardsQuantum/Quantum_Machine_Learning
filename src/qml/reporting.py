"""
qml.reporting
=============

Small text-reporting helpers for examples, notebooks, and CLIs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Row = Mapping[str, Any] | Sequence[Any]


def _format_value(value: Any, *, float_digits: int) -> str:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return f"{value:.{float_digits}g}"

    if isinstance(value, np.ndarray):
        return _format_value(value.tolist(), float_digits=float_digits)

    if isinstance(value, (list, tuple)):
        return (
            "[" + ", ".join(_format_value(item, float_digits=float_digits) for item in value) + "]"
        )

    return str(value)


def _normalize_rows(
    rows: Mapping[str, Any] | Iterable[Row],
    *,
    columns: Sequence[str] | None,
    float_digits: int,
) -> tuple[list[str], list[list[str]]]:
    if isinstance(rows, Mapping):
        headers = ["Metric", "Value"] if columns is None else list(columns)
        body = [
            [
                _format_value(key, float_digits=float_digits),
                _format_value(value, float_digits=float_digits),
            ]
            for key, value in rows.items()
        ]
        return headers, body

    materialized = list(rows)
    if not materialized:
        headers = ["Metric", "Value"] if columns is None else list(columns)
        return headers, []

    first = materialized[0]
    if isinstance(first, Mapping):
        headers = list(columns) if columns is not None else list(first.keys())
        body = [
            [_format_value(row.get(header, ""), float_digits=float_digits) for header in headers]
            for row in materialized
            if isinstance(row, Mapping)
        ]
        return headers, body

    if columns is None:
        max_width = max(len(tuple(row)) for row in materialized if not isinstance(row, Mapping))
        headers = (
            ["Metric", "Value"]
            if max_width == 2
            else [f"Column {idx + 1}" for idx in range(max_width)]
        )
    else:
        headers = list(columns)

    body = []
    for row in materialized:
        if isinstance(row, Mapping):
            body.append(
                [
                    _format_value(row.get(header, ""), float_digits=float_digits)
                    for header in headers
                ]
            )
            continue
        values = list(row)
        values.extend([""] * (len(headers) - len(values)))
        body.append(
            [_format_value(value, float_digits=float_digits) for value in values[: len(headers)]]
        )
    return headers, body


def format_table(
    rows: Mapping[str, Any] | Iterable[Row],
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    float_digits: int = 6,
) -> str:
    """Return a compact ASCII table for mappings, tuple rows, or dict rows."""
    headers, body = _normalize_rows(rows, columns=columns, float_digits=float_digits)
    widths = [
        max(len(header), *(len(row[idx]) for row in body)) if body else len(header)
        for idx, header in enumerate(headers)
    ]

    def line(left: str, fill: str, separator: str, right: str) -> str:
        return left + separator.join(fill * (width + 2) for width in widths) + right

    def row_line(values: Sequence[str]) -> str:
        return (
            "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"
        )

    lines = []
    if title:
        lines.append(title)
    lines.append(line("+", "-", "+", "+"))
    lines.append(row_line(headers))
    lines.append(line("+", "-", "+", "+"))
    lines.extend(row_line(row) for row in body)
    lines.append(line("+", "-", "+", "+"))
    return "\n".join(lines)


def print_table(
    rows: Mapping[str, Any] | Iterable[Row],
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    float_digits: int = 6,
) -> None:
    """Print a compact ASCII table."""
    print(format_table(rows, title=title, columns=columns, float_digits=float_digits))
    print()


def print_section(title: str, rows: Mapping[str, Any] | Iterable[Row]) -> None:
    """Compatibility wrapper for notebook result sections."""
    print_table(rows, title=title)
