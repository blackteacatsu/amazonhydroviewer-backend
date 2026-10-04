#!/usr/bin/env python
"""

For each selected initialization, the matching LDAS file is treated as the
forecast. Files from the same calendar month in every other year of the
hindcast period form the reference climatology.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import modules.fireRisk.angstrom as afi
import modules.utils as utils


CWD = Path.cwd()
SURFACE_MODEL_DIR = Path(
    "/mnt/c/Users/Kris/Documents/amazonforecast/backup/monthly"
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
    parser.add_argument(
        "--fcst-init-date",
        nargs=2,
        type=int,
        metavar=("YEAR", "MONTH"),
        help="Run one fixed climatology initialization.",
    )
    parser.add_argument("--variables", nargs="+", choices=tuple(VARIABLES), default=list(VARIABLES))
    parser.add_argument("--output-dir", type=Path, default=CWD / "output_angstrom_index",)
    args = parser.parse_args(argv)

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
    files:dict[datetime, Path],
    args: argparse.Namespace,
) -> list[tuple[datetime, Path]]:

    if args.fcst_init_date:
        try:
            requested = datetime(*args.fcst_init_date, 1)
        except ValueError as exc:
            raise SystemError(f'Invalid forecast initilization month: {exc}') from exc
        matches = [(date, path) for date, path in files.items() if date == requested]

    matches.sort(key=lambda item: item[0])
    if not matches:
        if args.fcst_init_date:
            selection = f"{args.fcst_init_date[0]:04d}-{args.fcst_init_date[1]:02d}" 
            # else f"month {args.month:02d} in {start_year}-{end_year}"
        
        raise FileNotFoundError(f"No forecast initialization found for {selection}")
    
    return matches


def _write_angstrom_index(
    file: Path,
    angstrom_result_dir: Path,
    variables:  dict[str, str],
) -> list[Path]:
    written: list[Path] = []

    inputs = {}
    for variable in variables.keys():
        inputs[f'da_{variable}'] = utils.read_trim_fcst(file, variable)
    da_afi = afi.get_afi(**inputs)

    afi_result = angstrom_result_dir / "angstrom_fire_index.zarr"
    da_afi.to_zarr(afi_result, zarr_format=2, mode="w")
    written.append(afi_result)

    return written


def main(argv: Sequence[str] | None = None) -> None:

    args = parse_args(argv)
    files = _discover_initializations(args.surface_model_dir.expanduser().resolve())
    targets = _select_targets(files, args)

    for target_date, target_file in targets:
        init_date = f"{target_date.year}_{target_date.strftime('%b').lower()}"
        angstromi_result_dir = args.output_dir.expanduser().resolve() / f'{init_date}_angstrom'
        angstromi_result_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 72}")
        print(f"Working on initialization: {target_date:%Y-%m-%d} ({target_file.name})")

        print(f"Output directory: {angstromi_result_dir}")

        selected_variables = {name: VARIABLES[name] for name in args.variables}
        _write_angstrom_index(target_file, angstromi_result_dir, selected_variables)

    print(f"\n Completed Angstrom Index for period {args.fcst_init_date}")


if __name__ == "__main__":
    main()
