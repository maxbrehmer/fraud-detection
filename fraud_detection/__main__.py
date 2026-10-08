import argparse
import json
from pathlib import Path
import tempfile

from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run and reproduce an educational fraud detection study.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate data, train models and write the report.")
    run.add_argument("--rows", type=int, default=40000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--review-budget", type=float, default=.02)
    run.add_argument("--output", type=Path, default=Path.cwd())
    verify = commands.add_parser("verify", help="Repeat the saved experiment and compare results exactly.")
    verify.add_argument("--output", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "run":
        if args.rows < 5000 or not 0 < args.review_budget < 1:
            parser.error("--rows must be at least 5000; --review-budget must be between 0 and 1.")
        result = run_experiment(args.output, args.rows, args.seed, args.review_budget)
        print(f"Report: {args.output.resolve() / 'reports/report.html'}")
        print(f"Selected model: {result['selected_model']}")
        print(json.dumps(result["models"][result["selected_model"]]["test"], indent=2))
    else:
        saved = json.loads((args.output / "reports/metrics.json").read_text())
        config = saved["config"]
        with tempfile.TemporaryDirectory(prefix="fraud-reproduce-") as temp:
            repeated = run_experiment(temp, config["rows"], config["seed"], config["review_budget"])
            if saved != repeated:
                raise RuntimeError("Reproducibility check failed: metrics or environment metadata differ.")
            for name in ["test_predictions.csv", "split_assignments.csv"]:
                if (args.output / "artifacts" / name).read_bytes() != (Path(temp) / "artifacts" / name).read_bytes():
                    raise RuntimeError(f"Reproducibility check failed: {name} differs.")
            for original in sorted((args.output / "reports").rglob("*")):
                if original.is_file():
                    relative = original.relative_to(args.output)
                    if original.read_bytes() != (Path(temp) / relative).read_bytes():
                        raise RuntimeError(f"Reproducibility check failed: {relative} differs.")
        print("PASS: data hash, split assignments, predictions, metrics, intervals, importance, report, and figures match exactly.")


if __name__ == "__main__":
    main()
