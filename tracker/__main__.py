"""
CLI entry point.

Usage:
    python -m tracker list
    python -m tracker compare --metric val_accuracy
    python -m tracker show <run_id>
    python -m tracker report --metric val_accuracy --output report.html
    python -m tracker clean --status failed
"""
import argparse
import json
import sys

from .storage import Storage
from .report import compare_runs, generate_html_report


def cmd_list(args):
    runs = Storage.load_all_runs(args.dir)
    if not runs:
        print("No runs found.")
        return
    print(f"\n{'RUN ID':<10} {'NAME':<30} {'STATUS':<12} {'DURATION':>10}")
    print("-" * 66)
    for r in sorted(runs, key=lambda x: x.get("start_time", 0), reverse=True):
        status_icon = {"completed": "✅", "failed": "❌", "running": "🔄"}.get(r.get("status", ""), "❓")
        print(
            f"{r['run_id']:<10} {r['name']:<30} "
            f"{status_icon} {r.get('status','?'):<10} "
            f"{str(r.get('duration_s','?')) + 's':>10}"
        )
    print()


def cmd_compare(args):
    try:
        df = compare_runs(metric=args.metric, base_dir=args.dir)
        if df.empty:
            print("No completed runs found.")
            return
        print(f"\n📊 Runs sorted by: {args.metric or 'start_time'}\n")
        print(df.to_string(index=False))
        print()
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_show(args):
    try:
        run = Storage.load_run(args.run_id, base_dir=args.dir)
        print(json.dumps(run, indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_report(args):
    generate_html_report(
        metric=args.metric,
        base_dir=args.dir,
        output_path=args.output,
    )


def cmd_clean(args):
    runs = Storage.load_all_runs(args.dir)
    targets = [r for r in runs if r.get("status") == args.status]
    if not targets:
        print(f"No runs with status '{args.status}' found.")
        return
    print(f"About to delete {len(targets)} run(s) with status '{args.status}':")
    for r in targets:
        print(f"  {r['run_id']}  {r['name']}")
    confirm = input("Confirm? [y/N] ")
    if confirm.lower() == "y":
        for r in targets:
            Storage.delete_run(r["run_id"], base_dir=args.dir)
        print(f"🗑️  Deleted {len(targets)} run(s).")
    else:
        print("Aborted.")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m tracker",
        description="MLOps Experiment Tracker CLI",
    )
    parser.add_argument("--dir", default="runs", help="Base directory for runs (default: runs)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all runs")

    p_compare = sub.add_parser("compare", help="Compare runs by metric")
    p_compare.add_argument("--metric", default=None, help="Metric to sort by")

    p_show = sub.add_parser("show", help="Show full details of a single run")
    p_show.add_argument("run_id", help="Run ID to display")

    p_report = sub.add_parser("report", help="Generate an HTML report")
    p_report.add_argument("--metric", default=None, help="Metric to highlight")
    p_report.add_argument("--output", default="report.html", help="Output HTML path")

    p_clean = sub.add_parser("clean", help="Delete runs by status")
    p_clean.add_argument("--status", default="failed", choices=["failed", "running", "completed"])

    args = parser.parse_args()

    dispatch = {
        "list": cmd_list,
        "compare": cmd_compare,
        "show": cmd_show,
        "report": cmd_report,
        "clean": cmd_clean,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
