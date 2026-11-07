# Soil-moisture drought code

`get_fireW_index.py --dc-method SoilMoist_inst` uses Yang et al. (2015),
https://doi.org/10.1175/JAMC-D-14-0090.1, equation 6:
$$ \text{DC} = 400 \times \ln(\text{SM}\_\text{sat} / \text{SM})$$

The local adaptation uses 0–40 cm rather than their modelled 0–35 cm.

FFMC, DMC, ISI, BUI and FWI retain the existing
calculations; `--dc-method original` remains the default.

First preprocess four HiHydroSoil v2.0 **WCsat** GeoTIFFs, path where the files are stored, `/mnt/vast/ksu/hihydrosoil_250_WCsat/Layers_WCsat`, in depth order:

Run the following setup after activating environment, but before running the job, to resolve the error that points to an incompatible PROJ database there.

Add these diagnostics after `conda activate analytics` in your job script, or run them in an interactive compute-node session:

```bash
which python
echo "PROJ_DATA=$PROJ_DATA"
echo "PROJ_LIB=$PROJ_LIB"

python -c "import rasterio; print('Rasterio:', rasterio.__file__); print('GDAL:', rasterio.__gdal_version__)"
conda list | grep -E '^(rasterio|pyproj|proj|gdal|libgdal)[[:space:]]'
```

Then test whether removing inherited database overrides resolves it:

```bash
env -u PROJ_DATA -u PROJ_LIB python -c \
  "from rasterio.crs import CRS; print(CRS.from_epsg(4326))"
```

**If that prints `EPSG:4326`**, add this after environment activation and before running preprocessing:

```bash
unset PROJ_DATA PROJ_LIB
```

### Following code prepare a 0-40 cm saturated soil moisture volumetric data

```sh
python scripts/python/preprocess_fwi_saturation.py \
  --ldas-template /mnt/vast/ksu/hihydrosoil_250_WCsat/LIS_HIST_daily.nc \
  --wcsat /mnt/vast/ksu/hihydrosoil_250_WCsat/Layers_WCsat/WCsat_0-5cm_M_250m.tif \
          /mnt/vast/ksu/hihydrosoil_250_WCsat/Layers_WCsat/WCsat_5-15cm_M_250m.tif \
          /mnt/vast/ksu/hihydrosoil_250_WCsat/Layers_WCsat/WCsat_15-30cm_M_250m.tif \
          /mnt/vast/ksu/hihydrosoil_250_WCsat/Layers_WCsat/WCsat_30-60cm_M_250m.tif \
  --output /mnt/vast/ksu/hihydrosoil_250_WCsat/LIS_formatted_WCsat/SM_vsat_0_40.nc
```

Raw integer rasters use the default scale 0.0001; already-scaled rasters
require `--scale 1`. Rasterio streams source bands, averages valid contributing
pixels onto the regular LIS geographic grid, and weights depths by 5,10,15,10 cm.
Missing depth layers remain missing. The file stores `SM_sat`, lat/lon, units,
depth and source paths. This valid-pixel average does not enforce a minimum
coverage fraction; inspect edge and water-adjacent cells before interpretation.

Then run into a separate output directory:

```sh
python scripts/python/get_fireW_index.py \
  --input-dir /path/to/daily_ldas \
  --output-dir tests/fwi/soil_method \
  --dc-method SoilMoist_inst \
  --soil-saturation tests/fwi/SM_sat_0_40.nc \
  --initialize
```

Subsequent batches omit `--initialize` and use `--restart-root` if needed.
Restarts reject method or saturation changes. The input must provide volumetric
`SoilMoist_inst`, four `SoilMoist_profiles`, and `SOIL_LAYER_THICKNESSES`
[10,30,60,100] cm. SM is weighted 1:3 across the first two layers.

Zero, negative, missing, and above-saturation moisture are masked, not clipped.
Outputs include soil moisture, a validity flag and an above-saturation flag;
JSON summaries include counts. DC is calculated from each day's moisture,
but FFMC/DMC still require history: an invalid day masks subsequent FWI values
under the existing conservative gap policy. `--initial-dc` is unused in soil mode.
Daily-mean weather and original Canadian month factors remain pilot limitations.

The notebook `tests/soilSat_prepreocess.ipynb` is retained for inspection. Its
working helpers are now available in `modules.fireRisk.fwi.soilMoist`.

Validation: `python tests/fwi/test_soil_dc.py` writes
`tests/fwi/soil_dc_validation.json`. Existing core/restart tests also apply.
If the analytics environment reports a PROJ database version mismatch, run with
`PROJ_DATA="$CONDA_PREFIX/share/proj"` after activating the environment.
