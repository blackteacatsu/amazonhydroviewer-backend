"""Build wildfire-risk products from probabilistic LDAS forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import xarray as xr
import numpy as np


HUMIDITY_VARIABLE = "Qair_f_tavg"
TEMP_VARIABLE = "Tair_f_tavg"
SURFACEP_VARIABLE = "Psurf_f_tavg"


"""
Code below compute the Angstrom Fire Index
"""
def _unit_key(da: xr.DataArray) -> str:
    """Normalize common unit spellings without guessing missing units."""
    units = da.attrs.get("units")
    if not isinstance(units, str) or not units.strip():
        # raise ValueError(f"{da.name or 'Input'} must have a nonempty 'units' attribute")
        if da.name == 'Qair_f_tavg':
            raise Exception
    return units.strip().lower().replace(" ", "").replace("_", "").replace("**", "^")


def _temperature_celsius(da_Tair: xr.DataArray) -> xr.DataArray:
    units = _unit_key(da_Tair)
    if units in {"k", "kelvin", "degk", "degreekelvin", "degreeskelvin"}:
        result = da_Tair - 273.15
    elif units in {"c", "°c", "degc", "celsius", "degreecelsius", "degreescelsius"}:
        result = da_Tair.copy(deep=False)
    else:
        raise ValueError(f"Unsupported temperature units: {da_Tair.attrs['units']!r}")
    result.attrs = {**da_Tair.attrs, "units": "degC"}
    return result


def get_Pvap(
        da_Qair: xr.DataArray,
        da_Psurf: xr.DataArray,
) -> xr.DataArray:
    """Calculate vapour pressure in Pa from specific humidity and pressure.

    Accept kg/kg or g/kg humidity and Pa, hPa/mbar, or kPa pressure.
    Shared coordinates must match exactly; missing values propagate.
    """
    humidity_units = _unit_key(da_Qair)
    if humidity_units in {"g/kg", "gkg-1", "gkg^-1"}:
        da_Qair = da_Qair / 1000
    elif humidity_units not in {"kg/kg", "kgkg-1", "kgkg^-1", "1"}:
        raise ValueError(f"Unsupported specific humidity units: {da_Qair.attrs['units']!r}")

    pressure_units = _unit_key(da_Psurf)
    pressure_scales = {"pa": 1, "hpa": 100, "mbar": 100, "mb": 100, "kpa": 1000}
    if pressure_units not in pressure_scales:
        raise ValueError(f"Unsupported pressure units: {da_Psurf.attrs['units']!r}")
    da_Psurf = da_Psurf * pressure_scales[pressure_units]
    da_Qair, da_Psurf = xr.align(da_Qair, da_Psurf, join="exact", copy=False)
    da_Pvap = (da_Qair * da_Psurf) / (0.622 + 0.378 * da_Qair)
    da_Pvap.attrs = {"long_name": "vapour pressure", "units": "Pa"}
    return da_Pvap.rename("Pvap_f_tavg")


def get_Psvap(da_Tair: xr.DataArray) -> xr.DataArray:
    """Calculate saturation vapour pressure over liquid water in Pa.

    Accept Celsius or Kelvin temperature; use the existing Magnus formula.
    """
    da_Tair = _temperature_celsius(da_Tair)
    da_Psvap = 611 * np.exp((da_Tair * 17.27) / (da_Tair + 237.3))
    da_Psvap.attrs = {"long_name": "saturation vapour pressure", "units": "Pa"}
    return da_Psvap.rename("Sat_Pvapour_f_tavg")


def get_QRair(
        da_Qair: xr.DataArray,
        da_Psurf: xr.DataArray,
        da_Tair: xr.DataArray,
) -> xr.DataArray:
    """Calculate relative humidity (%) over liquid water, without clipping."""
    da_Pvap = get_Pvap(da_Qair, da_Psurf)
    da_Psvap = get_Psvap(da_Tair)
    da_Pvap, da_Psvap = xr.align(da_Pvap, da_Psvap, join="exact", copy=False)
    da_QRair = 100 * (da_Pvap / da_Psvap)
    da_QRair.attrs = {"long_name": "relative humidity", "units": "%"}
    return da_QRair.rename("relative humidity")


def get_afi(
        da_Qair: xr.DataArray,
        da_Psurf: xr.DataArray,
        da_Tair: xr.DataArray,
) -> xr.DataArray:
    """Calculate the dimensionless Angstrom fire index; lower means higher risk.

    The conventional index uses afternoon temperature and relative humidity.
    With monthly mean inputs this is an index of monthly mean conditions,
    not the monthly mean of indices calculated at the original timesteps.
    """
    da_QRair = get_QRair(da_Qair, da_Psurf, da_Tair)
    da_Tair = _temperature_celsius(da_Tair)
    da_QRair, da_Tair = xr.align(da_QRair, da_Tair, join="exact", copy=False)
    da_afi = 0.05 * da_QRair - 0.1 * (da_Tair - 27)
    da_afi.attrs = {"units": "1", "long_name": "angstrom fire index"}
    return da_afi.rename("afi_init")


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