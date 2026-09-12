"""Build wildfire-risk products from probabilistic LDAS forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import xarray as xr
import numpy as np


RAINF_VARIABLE = "Rainf_tavg"
TEMPERATURE_VARIABLE = "Tair_f_tavg"
SOIL_MOISTURE_VARIABLE = "SoilMoist_inst"


# def probability_store(
#     probability_dir: str | Path,
#     init_date: str,
#     variable: str,
# ) -> Path:
#     """Return the Zarr store produced by ``get_prob_fcst.mainloop``."""
#     return Path(probability_dir) / f"{init_date}_tercile_prob_max_{variable}"


# def _read_category(
#     store: str | Path,
#     variable: str,
#     category: int,
# ) -> xr.DataArray:
#     path = Path(store)
#     if not path.exists():
#         raise FileNotFoundError(f"Probabilistic forecast store not found: {path}")

#     with xr.open_dataarray(path, engine="zarr") as probability:
#         if probability.name not in (None, variable):
#             raise ValueError(
#                 f"Expected variable {variable!r} in {path}, "
#                 f"found {probability.name!r}."
#             )
#         if "category" not in probability.dims:
#             raise ValueError(f"Category dimension not found in {path}")
#         return probability.isel(category=category, drop=True).load()


# def build_fire_risk_soilm(
#     soil_fcst_file: str | Path,
#     *,
#     soilmoist_var: str = SOIL_MOISTURE_VARIABLE,
#     minimum_probability: float = 60.0,
#     soil_profile_index: int = 1,
# ) -> xr.DataArray:
#     """Return risk where below-normal soil moisture exceeds the threshold."""
#     below_normal = _read_category(
#         soil_fcst_file, 
#         soilmoist_var, 
#         category=0
#     )

#     profile_dim = next(
#         (
#             name
#             for name in ("SoilMoist_profiles", "soil_moisture_profile", "depth")
#             if name in below_normal.dims
#         ),
#         None,
#     )
#     if profile_dim is not None:
#         if not 0 <= soil_profile_index < below_normal.sizes[profile_dim]:
#             raise IndexError(
#                 f"Soil profile index {soil_profile_index} is outside "
#                 f"{profile_dim} (size {below_normal.sizes[profile_dim]})."
#             )
#         below_normal = below_normal.isel({profile_dim: soil_profile_index}, drop=True)

#     risk = (below_normal > minimum_probability).rename("FireRisk_fcst_soilm")
#     risk.attrs.update(
#         description="Below-normal soil-moisture probability exceeds threshold",
#         minimum_probability_percent=minimum_probability,
#         source_variable=soilmoist_var,
#     )
#     return risk.drop_encoding()


# def build_fire_risk_tp(
#     fcst_file_dir: str | Path,
#     *,
#     init_date: str,
#     variables: Sequence[str] = (RAINF_VARIABLE, TEMPERATURE_VARIABLE),
#     minimum_probability: float = 60.0,
# ) -> xr.DataArray:
#     """Return risk where below-normal rain and above-normal heat coincide."""
#     required = {RAINF_VARIABLE, TEMPERATURE_VARIABLE}
#     missing = required.difference(variables)
#     if missing:
#         raise ValueError(f"T/P fire risk requires variables: {sorted(missing)}")

#     probability_dir = Path(fcst_file_dir)
#     rain_below = _read_category(
#         probability_store(probability_dir, init_date, RAINF_VARIABLE),
#         RAINF_VARIABLE,
#         category=0, # for rainfall below normal means dry condition
#     )
#     temperature_above = _read_category(
#         probability_store(probability_dir, init_date, TEMPERATURE_VARIABLE),
#         TEMPERATURE_VARIABLE,
#         category=2, # for air temp., above normal means dry condition
#     )
#     rain_below, temperature_above = xr.align(
#         rain_below, temperature_above, join="exact"
#     )

#     risk = (
#         (rain_below > minimum_probability)
#         & (temperature_above > minimum_probability)
#     ).rename("FireRisk_fcst_tp")
#     risk.attrs.update(
#         description=(
#             "Below-normal precipitation and above-normal temperature "
#             "probabilities both exceed threshold"
#         ),
#         minimum_probability_percent=minimum_probability,
#         source_variables=f"{RAINF_VARIABLE},{TEMPERATURE_VARIABLE}",
#     )
#     return risk.drop_encoding()


"""
Code below compute the Angstrom Fire Index
"""
def get_Pvapour_f_tavg(
        da_Qair : xr.DataArray,
        da_Psurf : xr.DataArray,
) -> xr.DataArray:
    """
    Args:
        da_Qair (xr.DataArray) : data array of specific humidity 
        da_Psurf (xr.DataArray) : data array of surface level pressure
    
    Returns:
        xr.DataArray :  data array of vapour pressure in Pa
    """
    if da_Qair.units != 'kg/kg':
        da_Qair = da_Qair / 1000 
        da_Qair.attrs['units'] = 'kg/kg'

    if da_Psurf.units == 'Pa':

        try:
            da_Pvap = (da_Qair * da_Psurf) / (0.622 + 0.378 * da_Qair)
            da_Pvap.attrs.update({'long_name' : 'vapour pressure', 'units' : 'Pa'})
            da_Pvap = da_Pvap.rename('Pvap_f_tavg')
        except Exception as exc:
            raise exc
    else:
        raise ValueError
    
    return da_Pvap 


def get_Sat_Pvapour(
        da_Tair : xr.DataArray
) -> xr.DataArray:
    """
    Args: 
        da_Tair (xr.DataArray) : data array of air temperature 
    """
    try: 
        if da_Tair.units != 'Kelvin':
            da = 611 * np.exp((da_Tair * 17.27) / (da_Tair + 237.3))
            da.attrs.update({'standard_name' : 'Sat_Pvapour_f_tavg', 'long_name' : 'saturation vapour pressure', 'unit' : 'Pa'})
            da = da.rename('Sat_Pvapour_f_tavg')

        elif da_Tair.units == 'Kelvin': # the kelvin formula should be added here
            da = 611 * np.exp((da_Tair * 17.27) / (da_Tair + 237.3))
        return da
    
    except Exception as exc:
        raise exc


def get_Rel_Qair(
        da_Qair : xr.DataArray,
        da_Psurf : xr.DataArray,
        da_Tair : xr.DataArray,
) -> xr.DataArray:
    da_Pvapouer = get_Pvapour_f_tavg(da_Qair, da_Psurf)

    da_Sat_Pvapour = get_Sat_Pvapour(da_Tair)

    try : 
        da_Rel_Qair = (da_Pvapouer / da_Sat_Pvapour) * 100 
        da_Rel_Qair.attrs.update({'units' : '%'})
        da_Rel_Qair.rename('relative humidity')
    except Exception as exc:
        raise exc

    return da_Rel_Qair


def get_afi(
        da_rel_qair : xr.DataArray,
        da_Tair : xr.DataArray
)-> xr.DataArray:
    da_afi = 0.05 * da_rel_qair - 0.1 & (da_Tair - 27)
    da_afi.attrs.update({'units' : '', 'range' : '0 - 8', 'long_name' : 'angstrom fire index'})
    da_afi = da_afi.rename('afi_init')
    return da_afi


"""
Code below initialized the FWI 
"""
