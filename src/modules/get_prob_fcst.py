from __future__ import annotations

from collections.abc import Mapping, Sequence

import xarray as xr
import numpy as np
from pathlib import Path

import modules.utils as utils


def get_thresh(icat : int,
               quantiles : list,
               xrds : xr.DataArray,
               dims : list[str, str] = ['ensemble', 'time']) -> tuple[int, int]:
    """
    Calculate threshold boundaries for a category based on quantiles.

    Args:
        icat (int): Category index (0, 1, 2 for terciles)
        quantiles (list): Quantile boundaries (e.g., [1/3, 2/3] for terciles)
        xrds (xarray.DataArray): Data array to calculate quantiles from
        dims (list): Dimensions to calculate quantiles over

    Returns:
        tuple: (lower_threshold, upper_threshold) for the category
    """
    if not all(elem in xrds.dims for elem in dims):
        raise Exception(f'Some dimensions in {dims} not present in xr.DataArray {xrds.dims}')

    # Rechunk core dimensions to single chunks for quantile computation with dask
    rechunk_dict = {d: -1 for d in dims if d in xrds.dims}
    if rechunk_dict and hasattr(xrds, 'chunks') and xrds.chunks is not None:
        xrds = xrds.chunk(rechunk_dict)

    if icat == 0:  # Below normal category
        xrds_lo = -np.inf
        xrds_hi = xrds.quantile(quantiles[icat], dim=dims)
    elif icat == len(quantiles):  # Above normal category
        xrds_lo = xrds.quantile(quantiles[icat-1], dim=dims)
        xrds_hi = np.inf
    else:  # Normal category
        xrds_lo = xrds.quantile(quantiles[icat-1], dim=dims)
        xrds_hi = xrds.quantile(quantiles[icat], dim=dims)

    return xrds_lo, xrds_hi


def calculate_probabilities(hcst : xr.DataArray,
                            fcst : xr.DataArray,
                            quantiles : Sequence[float] = (1/3., 2/3.),
                            *,
                            mask_zeros: bool = True) -> xr.DataArray:
    """
    Calculate tercile category probability exceedance for ensemble forecast.
    
    Uses hindcast to define climatological tercile boundaries (below-normal, 
    normal, above-normal), then calculates probability that forecast ensemble
    members fall into each category.
    
    Args:
        hcst (xarray.DataArray): Hindcast data with dims [time, ensemble, lat, lon]
        fcst (xarray.DataArray): Forecast data with dims [time, ensemble, lat, lon]
        quantiles (list): Category boundaries (default: terciles at [1/3, 2/3])
    
    Returns:
        xarray.DataArray: Probability (0-1) that forecast falls in each category
                         Dims: [category, time, lat, lon]
                         - Category 0 = below normal (< 33rd percentile)
                         - Category 1 = normal (33rd-67th percentile)
                         - Category 2 = above normal (> 67th percentile)
    """
    print('\n Computing probabilities...')
    numcategories = len(quantiles) + 1  # 3 categories for terciles

    # Some legacy products encode missing cells as zero. Real zero precipitation,
    # however, is meaningful for wildfire risk, so callers can retain zeros.
    fcst_masked = fcst.where(fcst != 0) if mask_zeros else fcst

    # Rechunk once for the quantile operation and compute all quantile edges once.
    q_dims = [d for d in ['ensemble', 'time'] if d in hcst.dims]
    if not q_dims:
        raise Exception(f"Expected at least one of ['ensemble', 'time'] in hcst dims, got {hcst.dims}")
    if hasattr(hcst, 'chunks') and hcst.chunks is not None:
        hcst = hcst.chunk({d: -1 for d in q_dims})
    q_edges = hcst.quantile(quantiles, dim=q_dims)

    l_probs = []
    for icat in range(numcategories):
        print(f' Category={icat}')
        if icat == 0:
            h_lo = -np.inf
            h_hi = q_edges.sel(quantile=quantiles[0])
        elif icat == len(quantiles):
            h_lo = q_edges.sel(quantile=quantiles[-1])
            h_hi = np.inf
        else:
            h_lo = q_edges.sel(quantile=quantiles[icat - 1])
            h_hi = q_edges.sel(quantile=quantiles[icat])

        # Drop scalar quantile coord to avoid carrying it into outputs.
        if hasattr(h_lo, "coords") and 'quantile' in h_lo.coords:
            h_lo = h_lo.drop_vars('quantile')
        if hasattr(h_hi, "coords") and 'quantile' in h_hi.coords:
            h_hi = h_hi.drop_vars('quantile')

        # Count fraction of ensemble members in this category.
        prob = np.logical_and(fcst_masked > h_lo, fcst_masked <= h_hi).sum('ensemble') / float(fcst_masked.sizes['ensemble'])
        l_probs.append(prob.assign_coords({'category': icat}))
    
    probs = xr.concat(l_probs, dim='category')
    return probs


def mainloop(
        variables : Mapping[str, str] | Sequence[str],
        hdct_files : list[Path],
        fcst_file : Path,
        prob_fcst_cache_dir : Path,
        init_date : str,
        river_mask_file : Path | None,
        *,
        mask_zeros: bool = True,
) -> None:
    import gc
    from tqdm import tqdm

    for variable, variable_longname in tqdm(variables.items()):  # Fixed: .items()

        print(f"\n{'='*60}")
        print(f"{variable_longname} ({variable})")
        print('='*60)

        try:
            print("Loading hindcast data...")
            hcst = utils.read_trim_hcst(hdct_files, variable)
            print(f"  Shape: {hcst.shape}")

            print("Loading forecast data...")
            fcst = utils.read_trim_fcst(fcst_file, variable)
            print(f"  Shape: {fcst.shape}")

            # Calculate probabilities (convert to percentages)
            print("Calculating tercile probabilities...")
            probs = calculate_probabilities(
                hcst,
                fcst,
                mask_zeros=mask_zeros,
            ) * 100
            print(f"\n Probability data shape: {probs.shape}")
            print(f"Dimensions: {probs.dims} Categories: {probs.category.values}")
            #print(f"Time steps: {len(probs.time)}")

            # Keep only maximum probability per category
            print("Filtering for maximum probabilities...")
            probs_with_nan = probs.where(probs == probs.max(dim='category'))

            # Output file path base
            output_file = prob_fcst_cache_dir / f'{init_date}_tercile_prob_max_{variable}'

            if variable == "Streamflow_tavg":
                if not river_mask_file.exists():
                    raise FileNotFoundError(
                        f"File not found: {river_mask_file}"
                    )
                with xr.open_dataset(river_mask_file) as river_ds:
                    river_network = river_ds['mask'].load()
                # if river_mask_file.exists():
                #     river_network = xr.open_dataset(river_mask_file)['mask']
                #     print()
                #     print()
                # else:
                #     print("File not found: {river_mask_file}")
                probs_with_nan = probs_with_nan.where(river_network)
                probs_with_nan.to_zarr(output_file, zarr_format = 2, mode = "w")
                print(f"  ✓ Saved → {Path(output_file).name}")
                
            else:
                probs_with_nan.to_zarr(output_file, zarr_format = 2, mode = "w")
                print(f" ✓ Saved → {Path(output_file).name}")

            print(f"\n ✓ Completed {variable} ")

        except Exception as exc:
            # print(f"\n✗ ERROR processing {variable}:")
            # print(f"  {type(e).__name__}: {e}")
            raise RuntimeError(
                f"\n✗ ERROR : Failed to generate probabilistic forecast for {variable}"
            ) from exc

        finally:
        # Clean up memory
            print("Cleaning up memory...")
            try:
                del hcst, fcst, probs, probs_with_nan
            except:
                pass
            gc.collect()

    print("\n" + "="*60)
    print("✓ All variables processed successfully!")
    print("="*60)
