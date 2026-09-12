#!/usr/bin/env python
"""Prepare probabilistic wildfire-risk backtests.

For each selected initialization, the matching LDAS file is treated as the
forecast. Files from the same calendar month in every other year of the
hindcast period form the reference climatology.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
import json
from pathlib import Path

# import modules.get_prob_fcst as prob
import modules.fireRisk.angstrom as afi
import modules.utils as utils


CWD = Path.cwd()
SURFACE_MODEL_DIR = Path(
    "/mnt/vast/prakrut/backup/lis_runs/malaria_amazon/forecast/monthly"
)
VARIABLES = {
    afi.HUMIDITY_VARIABLE: "Specific Humidity",
    afi.TEMP_VARIABLE: "Average air temperature",
    afi.SURFACEP_VARIABLE: "Surface Pressure",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface-model-dir",
        type=Path,
        default=SURFACE_MODEL_DIR,
        help="Directory containing ldas_fcst_*.nc files.",
    )
    parser.add_argument("--hcst-start-year", type=int, default=2001)
    parser.add_argument("--hcst-end-year", type=int, default=2020)

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--fcst-init-date",
        nargs=2,
        type=int,
        metavar=("YEAR", "MONTH"),
        help="Run one fixed climatology initialization.",
    )
    target.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        metavar="MONTH",
        help="Run every available hindcast year for one calendar month.",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        choices=tuple(VARIABLES),
        default=list(VARIABLES),
    )
    parser.add_argument(
        "--fire-risk-method",
        choices=("tp", "soilmoist", "both"),
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CWD / "wildfire_backtest_output",
    )
    parser.add_argument("--minimum-probability", type=float, default=60.0)
    parser.add_argument("--soil-profile-index", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0.0 <= args.minimum_probability <= 100.0:
        parser.error("--minimum-probability must be between 0 and 100")
    if args.soil_profile_index < 0:
        parser.error("--soil-profile-index must be non-negative")
    return args


def _discover_initializations(directory: Path) -> dict[datetime, Path]:
    """Return one deterministic source file for each initialization date."""
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    candidates: dict[datetime, list[Path]] = {}
    for path in directory.glob("ldas_fcst_*.nc"):
        initialization = utils._parse_date_from_name(path.name)
        if initialization is not None:
            candidates.setdefault(initialization, []).append(path)

    if not candidates:
        raise FileNotFoundError(f"No supported LDAS forecast files found in {directory}")

    return {
        initialization: max(
            paths,
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        for initialization, paths in candidates.items()
    }


def _select_targets(
    files: dict[datetime, Path],
    args: argparse.Namespace,
) -> list[tuple[datetime, Path]]:
    start_year = args.hcst_start_year
    end_year = args.hcst_end_year
    if start_year > end_year:
        raise SystemExit("Hindcast start year must not exceed end year.")

    if args.fcst_init_date:
        try:
            requested = datetime(*args.fcst_init_date, 1)
        except ValueError as exc:
            raise SystemExit(f"Invalid forecast initialization month: {exc}") from exc
        if requested.year <= end_year:
            raise SystemExit(
                f"Target year must be after the climatology period {start_year}-{end_year}"
                f"{start_year}-{end_year}."
            )
        matches = [
            (date, path)
            for date, path in files.items()
            if date == requested
        ]
    else:
        matches = [
            (date, path)
            for date, path in files.items()
            if date.year > end_year
            and date.month == args.month
            and date.day == 1
        ]

    matches.sort(key=lambda item: item[0])
    if not matches:
        selection = (
            f"{args.fcst_init_date[0]:04d}-{args.fcst_init_date[1]:02d}"
            if args.fcst_init_date
            else f"month {args.month:02d} in {start_year}-{end_year}"
        )
        raise FileNotFoundError(f"No forecast initialization found for {selection}")
    return matches


# def _reference_files(
#     files: dict[datetime, Path],
#     target_date: datetime,
#     start_year: int,
#     end_year: int,
# ) -> list[Path]:
#     references = [
#         path
#         for date, path in files.items()
#         if start_year <= date.year <= end_year
#         and date.month == target_date.month
#         and date.day == target_date.day
#         and date.year != target_date.year
#     ]
#     references.sort(key=lambda path: utils._parse_date_from_name(path.name))
#     expected = end_year - start_year + 1
#     if len(references) != expected:
#         raise FileNotFoundError(
#             f"Expected {expected} reference files for "
#             f"{target_date:%Y-%m}, but found {len(references)}. Check that every "
#             f"year from {start_year} through {end_year} has this initialization."
#         )
#     return references


# def _validate_method_variables(method: str, variables: set[str]) -> None:
#     required: set[str] = set()
#     if method in ("tp", "both"):
#         required.update((fire.RAINF_VARIABLE, fire.TEMPERATURE_VARIABLE))
#     if method in ("soilmoist", "both"):
#         required.add(fire.SOIL_MOISTURE_VARIABLE)
#     missing = required.difference(variables)
#     if missing:
#         raise SystemExit(
#             f"Fire-risk method {method!r} requires variables: {sorted(missing)}"
#         )


def _write_angstrom_index(
    args: argparse.Namespace,
    model_dir: Path,
    result_dir: Path,
    init_date: str,
) -> list[Path]:
    written: list[Path] = []
    if args.fire_risk_method in ("tp", "both"): # <-- replace with angstrom
        tp_risk = fire.build_fire_risk_tp(
            probability_dir,
            init_date=init_date,
            variables=args.variables,
            minimum_probability=args.minimum_probability,
        )
        tp_path = result_dir / "fire_risk_tp.zarr"
        tp_risk.to_zarr(tp_path, zarr_format=2, mode="w")
        written.append(tp_path)

    # if args.fire_risk_method in ("soilmoist", "both"):
    #     soil_store = fire.probability_store(
    #         probability_dir, init_date, fire.SOIL_MOISTURE_VARIABLE
    #     )
    #     soil_risk = fire.build_fire_risk_soilm(
    #         soil_store,
    #         minimum_probability=args.minimum_probability,
    #         soil_profile_index=args.soil_profile_index,
    #     )
    #     soil_path = result_dir / "fire_risk_soilmoist.zarr"
    #     soil_risk.to_zarr(soil_path, zarr_format=2, mode="w")
    #     written.append(soil_path)
    
    return written


def run_initialization(
    args: argparse.Namespace,
    files: dict[datetime, Path],
    target_date: datetime,
    target_file: Path,
) -> None:
    references = _reference_files(
        files,
        target_date,
        args.hcst_start_year,
        args.hcst_end_year,
    )
    init_date = f"{target_date.year}_{target_date.strftime('%b').lower()}"
    result_dir = args.output_dir.expanduser().resolve() / init_date
    probability_dir = result_dir / "probabilities"
    probability_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"Target initialization: {target_date:%Y-%m-%d} ({target_file.name})")
    print(
        f"Reference files: {len(references)} "
        f"(fixed climatology "
        f"{args.hcst_start_year}-{args.hcst_end_year})"
    )
    print(f"Output directory: {result_dir}")

    selected_variables = {name: VARIABLES[name] for name in args.variables}
    prob.mainloop(
        variables=selected_variables,
        fcst_file=target_file,
        hdct_files=references,
        prob_fcst_cache_dir=probability_dir,
        init_date=init_date,
        river_mask_file=None,
        mask_zeros=False, # <-- change this to True, if zeros in the data mean NaN, confirm  w/ prakrut...
    )
    fire_risk_paths = _write_fire_risk(
        args, probability_dir, result_dir, init_date
    )

    metadata = {
        "initialization_date": target_date.isoformat(),
        "forecast_file": str(target_file.resolve()),
        "reference_files": [str(path.resolve()) for path in references],
        "reference_count": len(references),
        "cross_validation": f"{args.hcst_start_year}-{args.hcst_end_year} fixed period climatology",
        "variables": args.variables,
        "fire_risk_method": args.fire_risk_method,
        "minimum_probability_percent": args.minimum_probability,
        "outputs": [str(path.resolve()) for path in fire_risk_paths],
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_method_variables(args.fire_risk_method, set(args.variables))
    files = _discover_initializations(args.surface_model_dir.expanduser().resolve())
    targets = _select_targets(files, args)
    for target_date, target_file in targets:
        run_initialization(args, files, target_date, target_file)

    print(f"\nCompleted {len(targets)} initialization(s).")


if __name__ == "__main__":
    main()
