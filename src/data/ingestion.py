
import json
from pathlib import Path

import pandas as pd

from src.data.quality import REQUIRED_COLUMNS, audit_dataframe


def audit_csv(
    csv_path: str | Path,
    report_path: str | Path,
    chunk_size: int = 100_000,
) -> dict:
    """
    Audit a large CSV in chunks without loading
    the entire dataset into memory.
    """
    csv_path = Path(csv_path)
    report_path = Path(report_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    if not csv_path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {csv_path}"
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    report_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "dataset": csv_path.name,
        "rows_inspected": 0,
        "chunks_processed": 0,
        "columns_found": 0,
        "missing_required_columns": [],
        "missing_values": {},
        "duplicate_rows_within_chunks": 0,
        "invalid_amount_rows": 0,
        "invalid_transaction_type_rows": 0,
        "invalid_fraud_label_rows": 0,

        "invalid_numeric_values": {},
        "invalid_labels": {},
        "negative_balance_rows": 0,
        "invalid_step_rows": 0,

        "status": "RUNNING",
    }

    try:
        chunks = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
        )

        for chunk in chunks:
            if report["chunks_processed"] == 0:
                report["columns_found"] = len(
                    chunk.columns
                )

                report["missing_required_columns"] = (
                    sorted(
                        REQUIRED_COLUMNS
                        - set(chunk.columns)
                    )
                )

            batch_report = audit_dataframe(chunk)

            for column, count in (
                batch_report["invalid_numeric_values"].items()
            ):
                report["invalid_numeric_values"][column] = (
                    report["invalid_numeric_values"].get(
                        column, 0
                    ) + count
            )

            for column, count in batch_report["invalid_labels"].items():
                report["invalid_labels"][column] = (
                    report["invalid_labels"].get(column, 0)
                    + count
                )

            report["negative_balance_rows"] += (
                batch_report["negative_balance_rows"]
            )

            report["invalid_step_rows"] += (
                batch_report["invalid_step_rows"]
            )



            report["rows_inspected"] += len(chunk)
            report["chunks_processed"] += 1

            for column, count in (
                batch_report["missing_values"].items()
            ):
                report["missing_values"][column] = (
                    report["missing_values"].get(
                        column, 0
                    ) + count
                )

            report[
                "duplicate_rows_within_chunks"
            ] += batch_report[
                "duplicate_rows_in_batch"
            ]

            report["invalid_amount_rows"] += (
                batch_report["invalid_amount_rows"]
            )

            report[
                "invalid_transaction_type_rows"
            ] += batch_report[
                "invalid_transaction_type_rows"
            ]

            report["invalid_fraud_label_rows"] += (
                batch_report["invalid_fraud_label_rows"]
            )

        report["status"] = (
            "COMPLETED"
            if report["chunks_processed"] > 0
            else "EMPTY_DATASET"
        )

    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        report["status"] = "FAILED"
        report["error"] = str(exc)

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if report["status"] == "FAILED":
        raise ValueError(
            f"CSV parsing failed. See report: {report_path}"
        )

    return report
