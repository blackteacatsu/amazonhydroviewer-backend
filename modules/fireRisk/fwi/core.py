

from __future__ import annotations


import xarray as xr
import numpy as np

from dask.distributed import Client

import datetime
import modules.utils as utils


# client = Client()

INITIAL_FFMC = 85.0
INITIAL_DMC = 6.0
INITIAL_DC = 15.0





def setup_init_cond(
        da_Tair_f_tavg : xr.DataArray,
        valid_mask : xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:

    da_template, valid_mask = xr.align(da_Tair_f_tavg, valid_mask, join='exact')
    
    da_ffmc_previous = xr.full_like(
        da_template,
        INITIAL_FFMC,
        dtype=np.float64
    ).where(valid_mask).rename('ffmc')

    da_dmc_previous = xr.full_like(
        da_template, 
        INITIAL_DMC, 
        dtype=np.float64
    ).where(valid_mask).rename('dmc')

    da_dc_previous = xr.full_like(
        da_template, 
        INITIAL_DC, 
        dtype=np.float64
    ).where(valid_mask).rename('dc')

    return da_ffmc_previous, da_dmc_previous, da_dc_previous


"""
FINE FUEL MOISTURE CODE
"""
def get_ffmc(
        da_Tair_f_tavg: xr.DataArray,
        da_QRair_tavg: xr.DataArray,
        da_Wind_f_tavg: xr.DataArray,
        da_Rainf_tavg: xr.DataArray,
        da_ffmc_previous: xr.DataArray,
) -> xr.DataArray:
    """Update daily FFMC independently at each cell, preserving xarray labels.

    Inputs must be one day's 
        temperature (degC), 
        relative humidity (%), 
        wind (km/h), 
        preceding 24-hour rainfall (mm), 
        previous FFMC (0..101).
    
    Missing or out-of-range humidity, wind, rain, or state yields NaN.
    Shared coordinate indexes must match exactly. Works with lazy Dask arrays.
    """
    # Preparation A: align shared coordinate indexes (array handling).
    (
        da_Tair_f_tavg, da_QRair_tavg, da_Wind_f_tavg,
        da_Rainf_tavg, da_ffmc_previous,
    ) = xr.align(
        da_Tair_f_tavg, da_QRair_tavg, da_Wind_f_tavg,
        da_Rainf_tavg, da_ffmc_previous, join="exact", copy=False,
    )
    # Preparation B: identify valid cells (input checks, not an FWI equation).
    da_valid = (
        np.isfinite(da_Tair_f_tavg)
        & np.isfinite(da_QRair_tavg)
        & np.isfinite(da_Wind_f_tavg)
        & np.isfinite(da_Rainf_tavg)
        & np.isfinite(da_ffmc_previous)
        & (da_QRair_tavg >= 0) & (da_QRair_tavg <= 100)
        & (da_Wind_f_tavg >= 0) & (da_Rainf_tavg >= 0)
        & (da_ffmc_previous >= 0) & (da_ffmc_previous <= 101)
    )
    # Preparation C: safe arithmetic in cells that will be masked as NaN.
    # Safe working values avoid evaluating invalid powers/divisions in masked
    # cells. The original validity mask is restored on the result.
    da_Tair_f_tavg = da_Tair_f_tavg.astype(np.float64).where(da_valid, 20.0)
    da_QRair_tavg = da_QRair_tavg.astype(np.float64).where(da_valid, 50.0)
    da_Wind_f_tavg = da_Wind_f_tavg.astype(np.float64).where(da_valid, 0.0)
    da_Rainf_tavg = da_Rainf_tavg.astype(np.float64).where(da_valid, 0.0)
    da_ffmc_previous = da_ffmc_previous.astype(np.float64).where(da_valid, 85.0)

    # Step 1 / Eq. (1): convert yesterday's FFMC F_o to moisture m_o.
    da_Moist_o = 147.2 * (101 - da_ffmc_previous) / (59.5 + da_ffmc_previous)
    # Step 2 / Eq. (2): effective rain r_f = r_o - 0.5, only if r_o > 0.5 mm.
    da_Rainf_mask = da_Rainf_tavg > 0.5
    # xr.where evaluates both expressions: use a positive denominator even
    # where the rain response will not be selected.
    da_Rainf_tavg_eff : xr.DataArray = xr.where(da_Rainf_mask, da_Rainf_tavg - 0.5, 1.0)
    # Step 3 / Eqs. (3a, 3b): rain increases moisture. The extra term
    # is zero for m_o <= 150 and active for m_o > 150.
    # -expm1(-x) is the numerically stable form of 1 - exp(-x).
    da_Moist_aft_rain = (
        da_Moist_o
        + 42.5 * da_Rainf_tavg_eff * np.exp(-100 / (251 - da_Moist_o))
        * (-np.expm1(-6.93 / da_Rainf_tavg_eff))
        + 0.0015 * (da_Moist_o - 150).clip(min=0)**2
        * np.sqrt(da_Rainf_tavg_eff)
    )
    # Apply rain only to rainy cells, then enforce the moisture limit of 250.
    da_Moist_o = xr.where(
        da_Rainf_mask, da_Moist_aft_rain, da_Moist_o,
    ).clip(max=250)

    # Step 4 / Eqs. (4, 5): equilibrium moisture for drying E_d and wetting E_w.
    da_thermal = (
        0.18 * (21.1 - da_Tair_f_tavg)
        * (-np.expm1(-0.115 * da_QRair_tavg))
    )
    # Eq. (4): E_d.
    da_EMC_d = (
        0.942 * da_QRair_tavg**0.679
        + 11 * np.exp((da_QRair_tavg - 100) / 10) + da_thermal
    )
    # Eq. (5): E_w. Both equilibria are computed for every cell.
    da_EMC_w = (
        0.618 * da_QRair_tavg**0.753
        + 10 * np.exp((da_QRair_tavg - 100) / 10) + da_thermal
    )
    # Step 5 / Eqs. (6a, 6b): drying-rate intermediate k_o and rate k_d.
    da_k_o = (
        0.424 * (1 - (da_QRair_tavg / 100)**1.7)
        + 0.0694 * np.sqrt(da_Wind_f_tavg) * (1 - (da_QRair_tavg / 100)**8)
    )
    da_k_d = da_k_o * 0.581 * np.exp(0.0365 * da_Tair_f_tavg)
    # Step 6 / Eqs. (7a, 7b): wetting-rate intermediate k_1 and rate k_w.
    # da_k_i corresponds to k_1 in the report.
    da_k_i = (
        0.424 * (1 - ((100 - da_QRair_tavg) / 100)**1.7)
        + 0.0694 * np.sqrt(da_Wind_f_tavg)
        * (1 - ((100 - da_QRair_tavg) / 100)**8)
    )
    da_k_w = da_k_i * 0.581 * np.exp(0.0365 * da_Tair_f_tavg)
    # Step 7 / Eqs. (8, 9): candidate final moisture for drying and wetting.
    da_Moist_drying = da_EMC_d + (da_Moist_o - da_EMC_d) * 10.0**(-da_k_d)
    da_Moist_wetting = da_EMC_w - (da_EMC_w - da_Moist_o) * 10.0**(-da_k_w)
    # Step 8: select drying if m_o > E_d, wetting if m_o < E_w,
    # or unchanged moisture between equilibria (including equality).
    da_Moist_aft_dry = xr.where(
        da_Moist_o > da_EMC_d,
        da_Moist_drying,
        xr.where(
            (da_Moist_o < da_EMC_d) & (da_Moist_o < da_EMC_w),
            da_Moist_wetting,
            da_Moist_o,
        ),
    )
    # Step 9 / Eq. (10): convert final moisture m to today's FFMC F.
    # Bound FFMC to 0..101 and restore missing/invalid-cell masks.
    da_ffmc = (
        59.5 * (250 - da_Moist_aft_dry) / (147.2 + da_Moist_aft_dry)
    ).clip(min=0, max=101).where(da_valid).rename("ffmc")
    da_ffmc.attrs = {"long_name": "Fine Fuel Moisture Code", "units": "1"}
    return da_ffmc


"""
DUFF MOISTURE CODE
"""
def get_dmc(
        da_Tair_f_tavg : xr.DataArray,
        da_QRair_tavg : xr.DataArray,
        da_Rainf_tavg : xr.DataArray,
        da_dmc_previous : xr.DataArray,
        da_day_lengths : xr.DataArray
) -> xr.DataArray:
        """Update daily DMC independently at each cell, preserving xarray labels.
        
        Inputs are one day's
            temperature (degC),
            relative humidity (%),
            24-hour rainfall (mm),

        """
        (
            da_Tair_f_tavg, da_QRair_tavg, da_day_lengths,
            da_Rainf_tavg, da_dmc_previous,
        ) = xr.align(
            da_Tair_f_tavg, da_QRair_tavg, da_day_lengths,
            da_Rainf_tavg, da_dmc_previous, join='exact', copy=False
        )
        # Preparation B: identify valid cells (input checks, not an FWI equation).
        da_valid = (
            np.isfinite(da_Tair_f_tavg)
            & np.isfinite(da_QRair_tavg)
            & np.isfinite(da_Rainf_tavg)
            & np.isfinite(da_dmc_previous)
            & (da_QRair_tavg >= 0) & (da_QRair_tavg <= 100) & (da_Rainf_tavg >= 0)
            & (da_dmc_previous >= 0) & (da_dmc_previous <= 101)
        )
        # Preparation C: safe arithmetic in cells that will be masked as NaN.
        # Safe working values avoid evaluating invalid powers/divisions in masked
        # cells. The original validity mask is restored on the result.
        da_Tair_f_tavg = da_Tair_f_tavg.astype(np.float64).where(da_valid, 20.0)
        da_QRair_tavg = da_QRair_tavg.astype(np.float64).where(da_valid, 50.0)
        da_Rainf_tavg = da_Rainf_tavg.astype(np.float64).where(da_valid, 0.0)
        da_dmc_previous = da_dmc_previous.astype(np.float64).where(da_valid, 85.0)

        # Step 1 / Eq. (11): effective rainfall, only if r_o > 1.5 mm.
        da_Rainf_mask = da_Rainf_tavg > 1.5
        da_Rainf_eff = xr.where(da_Rainf_mask, 0.92*da_Rainf_tavg - 1.27, 1.0)
        # If r_o > 1.5, then execute the following steps (to find Pr to replace Po):
        # Step 2 / Eq. (12): calculate duff moisture content 
        # from yesterday from previous day's dmc
        da_Moist_o = (
            20 + np.exp(5.6348 - da_dmc_previous / 43.43)
        )
        # Step 3 / Eq. (12a/b/c): find variable in DMC rain effect (da_b)
        da_log_dmc = np.log(da_dmc_previous.clip(min=33))

        da_b = xr.where(
            da_dmc_previous <=33,
            (100 / (0.5 + 0.3 * da_dmc_previous)),
            xr.where(
                da_dmc_previous <= 65,
                14 - 1.3 * da_log_dmc,
                6.2 * da_log_dmc - 17.2
            ),
        )
        # Step 4 / Eq. (14): find duff moisture content after rain
        da_Moist_aft_Rain = (
            da_Moist_o + 1000 * da_Rainf_eff 
            / (48.77 + da_b * da_Rainf_eff)
        )
        # Step 5 / Eq. (15): convert duff moisture content after rain
        # to dmc after rain and replace dmc from yesterday with it
        da_dmc_aft_rain = (
            244.72 - 43.43 * np.log(da_Moist_aft_Rain - 20)
        )
        da_dmc_previous = xr.where(
            da_Rainf_mask,
            da_dmc_aft_rain.clip(min=0),
            da_dmc_previous
        )
        # Step 6 / Eq. (16): calculate log drying rate in DMC, log M/day
        da_K = (
            1.894e-6 * (da_Tair_f_tavg.clip(min=-1.1) + 1.1) 
            * (100 - da_QRair_tavg) 
            * da_day_lengths
        )
        # Step 7 / Eq. (17): calculate dmc from either from P_o or P_r
        da_dmc = (
            da_dmc_previous + 100 * da_K
        ).where(da_valid).rename('dmc')
        da_dmc.attrs = {
            "long_name" : "Duff Moisture Code",
            "units": "1",
        }
        return da_dmc



"""
DROUGHT CODE
"""
def get_drought_code(
        #da_ : xr.DataArray,
        da_Tair_f_tavg : xr.DataArray,
        da_Rainf_tavg : xr.DataArray,
        da_dc_previous : xr.DataArray,
        da_day_lengths : xr.DataArray,
)-> xr.DataArray:
    """
    Placeholder - add doc string
    """
    (
        da_Tair_f_tavg, da_Rainf_tavg,
        da_dc_previous, da_day_lengths,
    ) = xr.align(
        da_Tair_f_tavg, da_Rainf_tavg,
        da_dc_previous, da_day_lengths,
        join='exact', copy=False
    )
    da_valid = (
        np.isfinite(da_Tair_f_tavg)
        & np.isfinite(da_Rainf_tavg)
        & np.isfinite(da_dc_previous)
        & (da_Rainf_tavg >= 0)
        & (da_dc_previous >= 0) & (da_dc_previous <= 101)
    )
    da_Rainf_tavg = da_Rainf_tavg.astype(np.float64).where(da_valid, 20.0)

    # Step 1 / Eq. (18): get effective rainfall
    da_Rainf_mask = da_Rainf_tavg > 2.8
    da_Rain_eff = xr.where(da_Rainf_mask, 0.83*da_Rainf_tavg - 12.7)
    # Step 2 / Eq. 
    da_moisture_eq_previous = (
        800 * np.expm1(- da_dc_previous / 400)
    )
    da_moisture_eq_aft_rain = (
        da_moisture_eq_previous + 3.937 * da_Rain_eff
    )
    da_dc_aft_rain = (
        400 * np.log(800/da_moisture_eq_aft_rain)
    )

    da_dc_previous = xr.where(
        da_Rainf_mask,
        da_dc_aft_rain.clip(min=0),
        da_dc_previous
    )
    da_Pevap_tavg = 0.36 * (da_Tair_f_tavg.clip(min=-2.8) + 2.8) + da_day_lengths
    da_dc = (
        da_dc_previous + 0.5 * da_Pevap_tavg
    ).where(da_valid).rename("dc")
    da_dc.attrs = {
        "long_name" : "Drought Code",
        "units" : "1",
    }
    return da_dc


def get_isi(
        
)-> xr.DataArray:
    
    return 0.208 * da_f * da_F


def get_bui() -> xr.DataArray:
    return


def get_fwi() -> tuple[xr.DataArray, xr.DataArray]:

    return fwi, 0.0272 * (fwi ** 1.77)


def main() -> None:
    setup_init_cond(ds_ldas['Tair_f_tavg'].isel(time = 0, drop=True))
    return


if __name__ == "__main__":
    main()