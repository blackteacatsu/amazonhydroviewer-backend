"""Daily FWI equations from Van Wagner and Pickett (1985), using xarray.

Each call advances ONE day. Supply degC, RH percent, km/h and 24-hour rain mm;
unit conversion and valid-time selection belong in the caller. Carry FFMC/DMC/DC
without rounding. Invalid cells become NaN; coordinate mismatches raise errors.
DMC effective day length Le and DC seasonal factor Lf are distinct inputs.
These implement the written equations, not the known cold-weather quirks in F-32.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

INITIAL_FFMC = 85.0
INITIAL_DMC = 6.0
INITIAL_DC = 15.0


def _prepare_inputs(*specs):
    """A: exact alignment; B: range/finite mask; C: safe float64 operands.

    Each specification is (DataArray, lower_bound, upper_bound, safe_value).
    None means no bound. Bounds are component-specific: DMC/DC have no 101 cap.
    Safe values are computational placeholders, never imputed observations.
    No compute/values calls: Dask inputs stay lazy.
    """
    arrays = xr.align(*(spec[0] for spec in specs), join="exact", copy=False)
    valid = xr.ones_like(arrays[0], dtype=bool)
    for array, (_, lower, upper, _) in zip(arrays, specs):
        valid = valid & np.isfinite(array)
        if lower is not None:
            valid = valid & (array >= lower)
        if upper is not None:
            valid = valid & (array <= upper)
    safe = tuple(array.astype(np.float64).where(valid, spec[3])
                 for array, spec in zip(arrays, specs))
    return safe, valid


def _result(array, valid, name, long_name):
    """Restore missing cells and attach index metadata without replacing data."""
    result = array.where(valid & np.isfinite(array)).rename(name)
    result.attrs = {"long_name": long_name, "units": "1"}
    return result


def setup_init_cond(
        da_Tair_f_tavg : xr.DataArray,
        valid_mask : xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:

    """Create pre-first-day state on a grid; valid_mask must be boolean.

    Pass a single-day template with its time dimension removed. Defaults are
    assumed conditions, not inferred fuel moisture from the LDAS soil layers.
    """
    if "time" in da_Tair_f_tavg.dims:
        raise ValueError("Select a single-day template with isel(time=0, drop=True)")
    if valid_mask.dtype.kind != "b":
        raise TypeError("valid_mask must be boolean")
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
    # Preparation A-C: shared mechanics, FFMC-specific bounds/placeholders.
    (da_Tair_f_tavg, da_QRair_tavg, da_Wind_f_tavg,
     da_Rainf_tavg, da_ffmc_previous), da_valid = _prepare_inputs(
        (da_Tair_f_tavg, None, None, 20.0),
        (da_QRair_tavg, 0, 100, 50.0),
        (da_Wind_f_tavg, 0, None, 0.0),
        (da_Rainf_tavg, 0, None, 0.0),
        (da_ffmc_previous, 0, 101, INITIAL_FFMC),
    )

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
    # Shared last term in Eqs. (4) and (5), not an additional model term.
    da_thermal = (
        0.18 * (21.1 - da_Tair_f_tavg)
        * (-np.expm1(-0.115 * da_QRair_tavg))
    )
    # Eq. (4): E_d.
    da_EMC_d = (
        0.942 * da_QRair_tavg**0.679
        + 11 * np.exp((da_QRair_tavg - 100) / 10) + da_thermal
    )
    # Eq. (5): E_w - Both equilibria are computed for every cell.
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


def get_dmc(
    da_Tair_f_tavg: xr.DataArray,
    da_QRair_tavg: xr.DataArray,
    da_Rainf_tavg: xr.DataArray,
    da_dmc_previous: xr.DataArray,
    da_day_lengths: xr.DataArray,
) -> xr.DataArray:
    """Advance DMC one day; day_lengths is effective day length Le in hours.

    Temperature is degC, RH percent and rainfall 24-hour mm. DMC >= 0 has no
    fixed upper bound. Le must be finite and in [0, 24].
    """
    # Preparation A-C: include Le and allow DMC above 101.
    (da_Tair_f_tavg, da_QRair_tavg, da_Rainf_tavg,
     da_dmc_previous, da_day_lengths), da_valid = _prepare_inputs(
        (da_Tair_f_tavg, None, None, 20.0),
        (da_QRair_tavg, 0, 100, 50.0),
        (da_Rainf_tavg, 0, None, 0.0),
        (da_dmc_previous, 0, None, INITIAL_DMC),
        (da_day_lengths, 0, 24, 12.0),
    )
    # Step 1 / Eq. (11): no effective rain below or at the 1.5 mm threshold.
    da_Rainf_mask = da_Rainf_tavg > 1.5
    da_Rainf_eff = xr.where(da_Rainf_mask, 0.92 * da_Rainf_tavg - 1.27, 0.0)
    # Step 2 / Eq. (12): keep log(M_o - 20) to avoid losing tiny moisture
    # differences when adding/subtracting 20 for very high DMC.
    da_log_Moist_excess = 5.6348 - da_dmc_previous / 43.43
    # Step 3 / Eqs. (13a-c): logarithms below 33 are unused; make them safe.
    da_log_dmc = np.log(da_dmc_previous.clip(min=33))
    da_b = xr.where(
        da_dmc_previous <= 33,
        100 / (0.5 + 0.3 * da_dmc_previous),
        xr.where(da_dmc_previous <= 65,
                 14 - 1.3 * da_log_dmc, 6.2 * da_log_dmc - 17.2),
    )
    # Steps 4-5 / Eqs. (14-15): logaddexp(a,b) = log(exp(a)+exp(b)).
    # This is the published moisture sum in log space, not a new equation.
    da_added_moisture = 1000 * da_Rainf_eff / (48.77 + da_b * da_Rainf_eff)
    da_log_added = np.log(xr.where(da_Rainf_mask, da_added_moisture, 1.0))
    da_log_Moist_aft_rain = np.logaddexp(da_log_Moist_excess, da_log_added)
    da_dmc_aft_rain = (244.72 - 43.43 * da_log_Moist_aft_rain).clip(min=0)
    # Dry cells keep yesterday's state exactly, discarding the placeholder branch.
    da_dmc_aft_rain = xr.where(da_Rainf_mask, da_dmc_aft_rain, da_dmc_previous)
    # Step 6 / Eq. (16): floor T at -1.1; 1.894e-6 = 1.894 * 10**(-6).
    da_K = (1.894e-6 * (da_Tair_f_tavg.clip(min=-1.1) + 1.1)
            * (100 - da_QRair_tavg) * da_day_lengths)
    # Step 7 / Eq. (17): drying is added on both wet and dry days.
    return _result(da_dmc_aft_rain + 100 * da_K, da_valid, "dmc", "Duff Moisture Code")


def get_drought_code(
    da_Tair_f_tavg: xr.DataArray,
    da_Rainf_tavg: xr.DataArray,
    da_dc_previous: xr.DataArray,
    da_day_lengths: xr.DataArray,
) -> xr.DataArray:
    """Advance DC one day using degC, 24-hour rain mm and seasonal factor Lf.

    day_lengths here is Lf, NOT DMC's Le. Negative Lf is valid. DC has no
    101 ceiling. The written Eq. (22) restriction prevents negative drying.
    """
    (da_Tair_f_tavg, da_Rainf_tavg,
     da_dc_previous, da_day_lengths), da_valid = _prepare_inputs(
        (da_Tair_f_tavg, None, None, 20.0),
        (da_Rainf_tavg, 0, None, 0.0),
        (da_dc_previous, 0, None, INITIAL_DC),
        (da_day_lengths, None, None, 0.0),
    )
    # Step 1 / Eq. (18): r_d = 0.83*r_o - 1.27 only above 2.8 mm.
    da_Rainf_mask = da_Rainf_tavg > 2.8
    da_Rain_eff = xr.where(da_Rainf_mask, 0.83 * da_Rainf_tavg - 1.27, 0.0)
    # Steps 2-4 / Eqs. (19-21): Q_o=800*exp(-DC/400), Q_r=Q_o+3.937*r_d.
    # Evaluate log(Q_r) directly to avoid underflow at large previous DC.
    da_log_Qo = np.log(800.0) - da_dc_previous / 400
    da_log_rain = np.log(xr.where(da_Rainf_mask, 3.937 * da_Rain_eff, 1.0))
    da_log_Qr = np.logaddexp(da_log_Qo, da_log_rain)
    da_dc_aft_rain = (400 * (np.log(800.0) - da_log_Qr)).clip(min=0)
    da_dc_aft_rain = xr.where(da_Rainf_mask, da_dc_aft_rain, da_dc_previous)
    # Step 5 / Eq. (22): use the original T with its own -2.8 floor; V >= 0.
    da_Pevap_tavg = (
        0.36 * (da_Tair_f_tavg.clip(min=-2.8) + 2.8) + da_day_lengths
    ).clip(min=0)
    # Step 6 / Eq. (23): add half the potential evapotranspiration.
    return _result(da_dc_aft_rain + 0.5 * da_Pevap_tavg, da_valid, "dc", "Drought Code")


def ffmc_to_moisture(da_ffmc: xr.DataArray) -> xr.DataArray:
    """Convert today's FFMC to fine-fuel moisture (%) for get_isi, Eq. (1).

    get_isi's existing moisture-input API is retained. Never pass FFMC directly
    as its second argument; convert it here first.
    """
    (da_ffmc,), valid = _prepare_inputs((da_ffmc, 0, 101, INITIAL_FFMC))
    result = (147.2 * (101 - da_ffmc) / (59.5 + da_ffmc)).where(valid)
    result = result.rename("fine_fuel_moisture")
    result.attrs = {"long_name": "Fine fuel moisture content", "units": "%"}
    return result


def get_isi(da_Wind_f_tavg: xr.DataArray, da_Moist_aft_dry: xr.DataArray) -> xr.DataArray:
    """Calculate ISI from wind (km/h) and fine-fuel moisture (%), NOT FFMC.

    Use ffmc_to_moisture(get_ffmc(...)) for the second argument.
    """
    (da_Wind_f_tavg, da_Moist_aft_dry), valid = _prepare_inputs(
        (da_Wind_f_tavg, 0, None, 0.0),
        (da_Moist_aft_dry, 0, None, 0.0),
    )
    # Step 1 / Eq. (24): wind function.
    da_func_W = np.exp(0.05039 * da_Wind_f_tavg)
    # Step 2 / Eq. (25): fine-fuel moisture function.
    da_func_F = (91.9 * np.exp(-0.1386 * da_Moist_aft_dry)
                 * (1 + da_Moist_aft_dry**5.31 / 4.93e7))
    # Step 3 / Eq. (26): return the array, not None.
    return _result(0.208 * da_func_F * da_func_W, valid, "isi", "Initial Spread Index")


def get_bui(da_dmc: xr.DataArray, da_drought_code: xr.DataArray) -> xr.DataArray:
    """Combine nonnegative DMC and DC, Eqs. (27a-b); both zero gives BUI=0."""
    (da_dmc, da_drought_code), valid = _prepare_inputs(
        (da_dmc, 0, None, 0.0), (da_drought_code, 0, None, 0.0),
    )
    # Safe denominator prevents 0/0 even in branches discarded by xr.where.
    denominator = da_dmc + 0.4 * da_drought_code
    safe_denominator = xr.where(denominator > 0, denominator, 1.0)
    # Eq. (27a): lower DMC branch, including the all-zero case.
    da_bui_low = 0.8 * da_drought_code * da_dmc / safe_denominator
    # Eq. (27b): subtract the correction; the fraction is not BUI itself.
    da_bui_high = (
        da_dmc - (1 - 0.8 * da_drought_code / safe_denominator)
        * (0.92 + (0.0114 * da_dmc)**1.7)
    )
    da_bui = xr.where(da_dmc <= 0.4 * da_drought_code, da_bui_low, da_bui_high)
    return _result(da_bui.clip(min=0), valid, "bui", "Buildup Index")


def get_fwi(da_isi: xr.DataArray, da_bui: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Return daily (FWI, DSR) DataArrays from nonnegative ISI and BUI."""
    (da_isi, da_bui), valid = _prepare_inputs(
        (da_isi, 0, None, 0.0), (da_bui, 0, None, 0.0),
    )
    # Step 1 / Eqs. (28a-b): choose the correct side of BUI=80.
    da_func_D = xr.where(
        da_bui <= 80, 0.626 * da_bui**0.809 + 2,
        1000 / (25 + 108.64 * np.exp(-0.023 * da_bui)),
    )
    # Step 2 / Eq. (29): intermediate intensity B.
    da_inter_fwi = 0.1 * da_isi * da_func_D
    # Step 3 / Eqs. (30a-b): B<=1 passes through. Log/power are evaluated
    # on at least 1 so zero and subunit B never cause invalid intermediate math.
    da_log_fwi = 2.72 * (0.434 * np.log(da_inter_fwi.clip(min=1)))**0.647
    da_fwi = xr.where(da_inter_fwi > 1, np.exp(da_log_fwi), da_inter_fwi)
    da_fwi = _result(da_fwi, valid, "fwi", "Fire Weather Index")
    # Step 4 / Eq. (31): attach attrs, never replace either array with a dict.
    da_dsr = _result(0.0272 * da_fwi**1.77, valid, "dsr", "Daily Severity Rating")
    return da_fwi, da_dsr
