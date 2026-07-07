"""Shared utility helpers for POP."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional


def resolve_output_dir(
    output_dir: Optional[str | Path] = None,
    base_dir: Optional[str | Path] = None,
    script_path: Optional[str] = None,
) -> Path:
    """Resolve output directory for visualization assets.

    Defaults to: <base_dir>/output/<script_stem>/
    """
    if output_dir is not None:
        return Path(output_dir)

    base = Path(base_dir) if base_dir is not None else Path.cwd()
    if script_path is None:
        script_path = sys.argv[0] if sys.argv else ""
    script_stem = Path(script_path).stem if script_path else ""
    if script_stem in ("", "-c", "<stdin>", "ipykernel_launcher"):
        script_stem = "pop_output"
    return base / "output" / script_stem


def format_trace_context(trace_context: Optional[object]) -> Optional[str]:
    """格式化传播上下文信息，供报警/日志打印使用。"""
    if trace_context is None:
        return None
    if isinstance(trace_context, str):
        return trace_context
    if isinstance(trace_context, dict):
        status_line = trace_context.get("status_line")
        if status_line:
            return str(status_line)
    return str(trace_context)


__all__ = ["resolve_output_dir", "format_trace_context"]
