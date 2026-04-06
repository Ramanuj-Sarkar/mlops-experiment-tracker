"""
Report generation utilities for comparing and summarising experiment runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .storage import Storage


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def compare_runs(
    metric: Optional[str] = None,
    base_dir: str = "runs",
    ascending: bool = False,
    status: str = "completed",
):
    """
    Load all runs and return a list of dicts sorted by a given metric.

    Args:
        metric:    Metric name to sort by (e.g. "val_accuracy"). If None,
                   runs are sorted by start_time descending.
        base_dir:  Directory where runs are stored.
        ascending: Sort direction (False = best-first for most metrics).
        status:    Filter by run status ("completed", "failed", "running", or "all").

    Returns:
        List of run dicts enriched with a `last_<metric>` key.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for compare_runs. Install with: pip install pandas")

    runs = Storage.load_all_runs(base_dir)

    if status != "all":
        runs = [r for r in runs if r.get("status") == status]

    rows = []
    for r in runs:
        row = {
            "run_id": r["run_id"],
            "name": r["name"],
            "status": r.get("status", "unknown"),
            "duration_s": r.get("duration_s"),
            "start_time": r.get("start_time"),
            **r.get("params", {}),
            **r.get("tags", {}),
        }
        # Flatten metrics → last recorded value for each key
        for k, history in r.get("metrics", {}).items():
            if history:
                row[f"{k}"] = history[-1]["value"]
        rows.append(row)

    df = pd.DataFrame(rows)

    if metric and metric in df.columns:
        df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    elif "start_time" in df.columns:
        df = df.sort_values("start_time", ascending=False).reset_index(drop=True)

    return df


def generate_html_report(
    metric: Optional[str] = None,
    base_dir: str = "runs",
    output_path: str = "report.html",
) -> str:
    """
    Generate a self-contained HTML report of all completed runs.

    Args:
        metric:      Metric to highlight best value for (green = best).
        base_dir:    Directory where runs are stored.
        output_path: Where to write the HTML file.

    Returns:
        Absolute path to the written report.
    """
    df = compare_runs(metric=metric, base_dir=base_dir)

    if df.empty:
        html_table = "<p>No completed runs found.</p>"
    else:
        # Drop internal columns that aren't useful in the report
        display_cols = [c for c in df.columns if c not in ("start_time",)]
        df_display = df[display_cols]

        # Highlight the best value in the target metric column
        def highlight_best(col):
            if metric and col.name == metric and col.dtype in ("float64", "int64"):
                best_idx = col.idxmax()
                return [
                    "background-color: #d4edda; font-weight: bold;"
                    if i == best_idx else ""
                    for i in col.index
                ]
            return [""] * len(col)

        styled = df_display.style.apply(highlight_best)
        html_table = styled.to_html()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLOps Experiment Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 1200px; margin: 40px auto; padding: 0 20px;
      background: #f8f9fa; color: #212529;
    }}
    h1 {{ color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 12px; }}
    .meta {{ color: #6c757d; font-size: 0.9em; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; background: white;
             border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    th {{ background: #343a40; color: white; padding: 12px 16px;
          text-align: left; font-size: 0.85em; text-transform: uppercase;
          letter-spacing: 0.05em; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #dee2e6; font-size: 0.9em; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f1f3f5; }}
    .badge-completed {{ background: #d4edda; color: #155724;
                        padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
    .badge-failed {{ background: #f8d7da; color: #721c24;
                     padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
    .badge-running {{ background: #cce5ff; color: #004085;
                      padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
  </style>
</head>
<body>
  <h1>🧪 Experiment Report</h1>
  <p class="meta">
    Sorted by: <strong>{metric or 'start time'}</strong> &nbsp;|&nbsp;
    Total runs: <strong>{len(df)}</strong> &nbsp;|&nbsp;
    Base dir: <code>{base_dir}</code>
  </p>
  {html_table}
</body>
</html>
"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    print(f"📊 Report written → {out.resolve()}")
    return str(out.resolve())
