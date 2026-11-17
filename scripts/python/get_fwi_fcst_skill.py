from __future__ import annotations

import os
import sys
import json

import argparse

import numpy as np
import xarray as xr

from pathlib import Path
from datetime import date

from dotenv import load_dotenv


load_dotenv()

SRC = Path(__file__).resolve().parents[2] / 'src' # <== replace with .env in the future
sys.path.insert(0, str(SRC))
from modules.fireRisk.fwi import statistic
from modules.fireRisk.fwi.utils import write_netcdf


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fcst-dir', '--fcst_dir', type=Path)
    p.add_argument('--retro-dir', '--retro_dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--start-yr', type=int, default=2001)
    p.add_argument('--end-yr', type=int, default=2024)
    p.add_argument('--init-mmdd', default='07-01') # <-- currently start with 07-01
    p.add_argument('--months', type=int, default=6)
    p.add_argument('--min-years', type=int, default=10)
    p.add_argument('--stage', choices=('prepare', 'correlation', 'all'), default='all')
    args = p.parse_args(argv)

    if args.start_yr > args.end_yr or args.months < 1:
        p.error('--min-years exceeds the requested number of years') 
    if args.min_years < 3:
        p.error('--min-years must be at least 3')
    if args.stage != 'prepare' and args.min_years > args.end_yr - args.start_yr + 1:
        p.error('--min-years exceeds the requested number of years')
    try:
        initializations = {
            year : date.fromisoformat(f'{year}-{args.init_mmdd}')
            for year in range(args.start_yr, args.end_yr + 1)
        }
    except ValueError as exc:
        p.error(str(exc))

    if args.stage != 'correlation' and (args.fcst_dir is None or args.retro_dir is None):
        p.error('Preparing monthly data requires --fcst-dir and --retro-dir')
    out = args.output_dir.expanduser().resolve()
    cache_dir = out / '.work' / 'monthly'
    cache_dir.mkdir(parents=True, exist_ok=True)
    caches=[cache_dir/f'monthly_lead_{lead:02d}.nc' for lead in range(1, args.months+1)]
    if args.stage != 'correlation':
        if any(path.exists() for path in caches):
            raise FileExistsError(
                'Monthly caches already exist; use correlation stage '
                'or a new output directory'
            )
        retro_dir=args.retro_dir.expanduser().resolve()
        fcst_dir=args.fcst_dir.expanduser().resolve()
        retro_files=statistic.inventory(retro_dir)
        fcst_files={
            year: statistic.inventory(fcst_dir / f'init_{init_date}') 
            for year, init_date in initializations.items()
        }
        previous_grid=None
        expected_members=None
        for lead, path in enumerate(caches, 1):
            monthly=statistic.prep_monthly_cache(
                fcst_files,
                retro_files,
                initializations,
                lead
            )
            member_ids={
                year:monthly['ensemble'].values[
                    monthly['member_present'].sel(year=year).values==1
                ].tolist() for year in initializations
            }
            if expected_members is not None and member_ids!=expected_members:
                raise ValueError()
            if previous_grid is not None:
                statistic.same_grid(previous_grid, monthly)
                statistic.same_configuration(previous_grid, monthly)
            expected_members=member_ids
            monthly.attrs.update(
                fcst_dir=str(fcst_dir),
                retro_dir=str(retro_dir),
            )
            write_netcdf(monthly, path)
            previous_grid=monthly[['lat', 'lon']].copy()
            previous_grid.attrs=dict(monthly.attrs)
    if args.stage == 'prepare':
        print(f'Monthly caches ready: {cache_dir}')
        return
    report={'stage': 'correlation', 'periods':[], 'min_years':args.min_years}
    for lead, path in enumerate(caches, 1):
        tgt=out / f'corr_lead_{lead:02d}.nc'
        figs={
            code: out / f'corr_lead_{lead:02d}_{code}.png'
            for code in statistic.CODES
        }
        if tgt.exists() or any(fig.exists() for fig in figs.values()):
            raise FileExistsError(
                f'Correlation output exists for period {lead}; backup before rerunning'
            )

        with xr.open_dataset(path) as source:
            needed=[
                'member_present',
                'member_count',
                *[f'fcst_{code}' for code in statistic.CODES],
                *[f'retro_{code}' for code in statistic.CODES],
            ]
            if source['year'].values.tolist() != list(initializations) or source.attrs.get('lead_month') != lead:
                raise ValueError(
                    f'Cached years of lead period differ: {path}'
                )
            for year, init in initializations.items():
                start, end=statistic.month_windows(init, lead)
                for name, expected in [
                    ('initialization', init),
                    ('period_start', start),
                    ('period_end', end)
                ]:
                    if source[name].sel(year=year).values != np.datetime64(expected, 'ns'):
                        raise ValueError(f'Cached {name} differs for {year}: {path}')
            monthly=source[needed].load()
        result=statistic.eval_monthly_deterministic(monthly, args.min_years)
        write_netcdf(result, tgt)
        for code, fig in figs.items():
            plt_da=result[
                [f'pearson_r_{code}', f'paired_years_{code}']
            ].rename({
                f'pearson_r_{code}':'correlation',
                f'paired_years_{code}':'paired_yrs',
            })
            plt_da.attrs.update(result.attrs)
            plt_da.attrs['code']=code
            statistic.plot_corr(plt_da, fig)

        report['periods'].append(
            {
                'lead_month': lead,
                'cache': str(path),
                'member_counts': monthly['member_count'].values.tolist(),
                'codes': {
                    code: {
                        'finite_correlation_cells': 
                        int(
                            np.isfinite(result[f'pearson_r_{code}']).sum().item()
                        ),
                        'paired_years_min':
                        int(
                            result[f'paired_years_{code}'].min().item()
                        ),
                        'paired_years_max':
                        int(
                            result[f'paired_years_{code}'].max().item()
                        ),
                    } 
                    for code in statistic.CODES
                },
            }
        )
        print(f'Correlation period {lead} : {tgt}', flush=True)
        del monthly, result

    # The JSON accompanies the maps so sample coverage remains inspectable.
    temporary = out / 'deterministic_skill.json.partial'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(out / 'deterministic_skill.json')


if __name__ == '__main__': 

    from dask.distributed import Client, LocalCluster
    # Local workers use this node only; 
    # request CPUs/RAM through the job scheduler first.
    with LocalCluster(n_workers=4, threads_per_worker=1,
                        processes=True, memory_limit='32GiB',
                        dashboard_address=None) as cluster:
        with Client(cluster):
            main()