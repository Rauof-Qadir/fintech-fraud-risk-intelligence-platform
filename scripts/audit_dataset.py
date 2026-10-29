
import argparse
import json
from pathlib import Path

from src.data.ingestion import audit_csv


def main():
    parser = argparse.ArgumentParser(
        description="Audit a transaction CSV dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source transaction CSV",
    )

    parser.add_argument(
        "--output",
        default="reports/data_quality_report.json",
        help="Path for the JSON quality report",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per batch",
    )

    args = parser.parse_args()

    report = audit_csv(
        csv_path=Path(args.input),
        report_path=Path(args.output),
        chunk_size=args.chunk_size,
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
