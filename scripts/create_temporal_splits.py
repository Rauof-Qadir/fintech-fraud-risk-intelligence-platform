
import argparse
import json

from src.data.splitting import split_csv_by_time


def main():
    parser = argparse.ArgumentParser(
        description="Create chronological train, validation and test splits."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source CSV",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/splits",
        help="Output folder for generated splits",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    summary = split_csv_by_time(
        input_path=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )

    report_path = "reports/temporal_split_report.json"

    import os
    os.makedirs("reports", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {report_path}")


if __name__ == "__main__":
    main()
