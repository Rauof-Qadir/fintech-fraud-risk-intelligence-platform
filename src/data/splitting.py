
from pathlib import Path

import pandas as pd


SPLITS = {
    "train": (1, 520),
    "validation": (521, 631),
    "test": (632, 743),
}


def split_csv_by_time(
    input_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 100_000,
) -> dict:
    """Split transactions chronologically using bounded-memory CSV reads."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        name: output_dir / f"{name}.csv"
        for name in SPLITS
    }

    # Rebuild only these generated split artifacts.
    for path in outputs.values():
        path.unlink(missing_ok=True)

    stats = {
        name: {"rows": 0, "fraud": 0}
        for name in SPLITS
    }

    stats["excluded"] = {
        "rows": 0,
        "invalid_step_rows": 0,
        "outside_configured_range": 0,
    }

    required = {"step", "isFraud"}
    first_chunk = True

    for chunk in pd.read_csv(
        input_path, chunksize=chunk_size
    ):
        missing = required - set(chunk.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        step = pd.to_numeric(
            chunk["step"], errors="coerce"
        )

        target = pd.to_numeric(
            chunk["isFraud"], errors="coerce"
        )

        valid_step = (
            step.notna()
            & step.between(1, 743)
            & (step % 1 == 0)
        )

        valid_target = (
            target.notna()
            & target.isin([0, 1])
        )

        if not valid_target.all():
            raise ValueError(
                "Invalid fraud labels found; "
                "fix the data before creating model splits."
            )

        stats["excluded"]["invalid_step_rows"] += int(
            (~valid_step).sum()
        )

        stats["excluded"]["rows"] += int(
            (~valid_step).sum()
        )

        stats["excluded"]["outside_configured_range"] += int(
            (step.notna() & ~step.between(1, 743)).sum()
        )

        # Keep original rows and columns unchanged.
        for name, (low, high) in SPLITS.items():
            mask = valid_step & step.between(low, high)
            selected = chunk.loc[mask]

            if selected.empty:
                continue

            path = outputs[name]
            selected.to_csv(
                path,
                index=False,
                mode="a" if path.exists() else "w",
                header=not path.exists(),
            )

            stats[name]["rows"] += len(selected)
            stats[name]["fraud"] += int(
                target.loc[mask].sum()
            )

        first_chunk = False

    for name in SPLITS:
        rows = stats[name]["rows"]
        fraud = stats[name]["fraud"]

        stats[name]["fraud_rate"] = (
            fraud / rows if rows else None
        )

        stats[name]["output_path"] = str(outputs[name])

    stats["excluded"]["rows"] = int(
        stats["excluded"]["rows"]
    )

    return stats
