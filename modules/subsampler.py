import numpy as np
import xarray as xr
from pathlib import Path
import json

class HydroViewerSubsampler:
    """
    Poseidon-style grid subsampling for HydroViewer xarray data
    Adapted from Poseidon's grid_subsample.py
    """
    import numpy as np
    
    def __init__(self, data, resolution=256):
        """
        Parameters
        ----------
        data : xr.DataArray
            Your data with dims (time, category, lat, lon)
        resolution : int
            Tile resolution (pixels per tile)
        """
        self.full_data = data
        self.resolution = resolution
        self.lat = data.lat.values
        self.lon = data.lon.values
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        
        # Calculate grid spacing
        self.dlat = np.abs(np.diff(self.lat).mean())
        self.dlon = np.abs(np.diff(self.lon).mean())

        # NaN aware
        # sample_slice = self.full_data.isel(time = 0, category = 0).values
        # self.nan_fraction = np.isnan(sample_slice).sum() / sample_slice.size
        
        print(f"Full grid: {self.nlat}×{self.nlon} = {self.nlat*self.nlon:,} cells")
        print(f"Grid spacing: Δlat={self.dlat:.4f}°, Δlon={self.dlon:.4f}°")
        #print(f"NaN fraction: {100*self.nan_fraction:.1f}% (land/ocean mask)")

    
    def find_common_factors(self, num1, num2):
        """Find all common factors of two numbers (from Poseidon)"""
        common = []
        g = np.gcd(num1, num2)
        for i in range(1, int(np.sqrt(g)) + 1):
            if g % i == 0:
                common.append(i)
                if g != i * i:
                    common.append(g // i)
        return np.array(sorted(common))
    
    def pick_grain_sizes(self, zooms, factor=1.0):
        """
        Calculate appropriate grain size for each zoom level
        Adapted from Poseidon's pick_grain_size
        
        Parameters
        ----------
        zooms : list or np.ndarray
            Zoom levels (e.g., [0, 1, 2, 3, 4, 5])
        factor : float
            Adjustment factor for grain selection
        
        Returns
        -------
        grains : list
            Grain size for each zoom level
        """
        # Representative grid spacing in meters (approximate)
        rep_dx = 111000 * self.dlon  # degrees to meters at equator
        
        # Calculate required resolution at each zoom level
        # Web Mercator: world width = 2^zoom * tile_resolution pixels
        interp_dx = 6371e3 * 2 * np.pi / (2**np.array(zooms)) / self.resolution
        
        # Find available grain sizes (factors of grid dimensions)
        avail = self.find_common_factors(self.nlat, self.nlon)
        avail_dx = avail * rep_dx
        
        print(f"\nAvailable grain sizes: {avail}")
        print(f"Available resolutions (m): {avail_dx}")
        
        grains = []
        for zoom, dx in zip(zooms, interp_dx):
            # Find grain that gives resolution closest to target
            idx = np.searchsorted(avail_dx, dx * factor)
            idx = min(idx, len(avail) - 1)
            grain = avail[idx]
            grains.append(grain)
            
            actual_res = avail_dx[idx]
            print(f"Zoom {zoom}: target={dx:.0f}m, actual={actual_res:.0f}m, grain={grain}")
        
        return grains
    
    def subsample_data(self, grain):
        """
        Subsample data by taking every grain-th point
        Adapted from Poseidon's subsample_ocedata
        
        Parameters
        ----------
        grain : int
            Subsampling factor
        
        Returns
        -------
        subsampled : xr.DataArray
            Coarse-grained data
        """
        if grain == 1:
            return self.full_data
        
        print(f"  Subsampling with grain {grain} (NaN-aware)...")
        subsampled = subsample_data_preserve_mask(self.full_data, grain)

        # NaN aware
        sample_slice = self.full_data.isel(time = 0, category = 0).values
        self.nan_fraction = np.isnan(sample_slice).sum() / sample_slice.size
        
        print(f"  Grain {grain}: {len(subsampled.lat)}×{len(subsampled.lon)} = "
              f"{len(subsampled.lat)*len(subsampled.lon):,} cells "
              f"({100*(1-len(subsampled.lat)*len(subsampled.lon)/(self.nlat*self.nlon)):.1f}% reduction)")
        print(f"NaN fraction: {100*self.nan_fraction:.1f}% (land/ocean mask)")
        
        return subsampled
    
    def generate_pyramid(self, zooms=None):
        """
        Generate the full resolution pyramid
        
        Parameters
        ----------
        zooms : list, optional
            Zoom levels to generate. Default: [0, 1, 2, 3, 4, 5]
        
        Returns
        -------
        pyramid : dict
            Dictionary mapping zoom level to data
        grain_map : dict
            Dictionary mapping zoom level to grain size
        """
        if zooms is None:
            zooms = list(range(6))  # 6 levels like your sketch
        
        print("\n" + "="*60)
        print("GENERATING RESOLUTION PYRAMID")
        print("="*60)
        
        # Calculate grain sizes
        grains = self.pick_grain_sizes(zooms)
        
        # Get unique grains
        unique_grains = sorted(set(grains))
        
        print(f"\nUnique grain sizes needed: {unique_grains}")
        print("\nSubsampling data...")
        
        # Create subsampled datasets
        grain_to_data = {}
        for grain in unique_grains:
            grain_to_data[grain] = self.subsample_data(grain)
        
        # Map zoom levels to data
        pyramid = {}
        grain_map = {}
        for zoom, grain in zip(zooms, grains):
            pyramid[zoom] = grain_to_data[grain]
            grain_map[zoom] = grain
        
        print("\n" + "="*60)
        print("PYRAMID SUMMARY")
        print("="*60)
        for zoom in zooms:
            data = pyramid[zoom]
            grain = grain_map[zoom]
            print(f"Zoom {zoom}: {len(data.lat):4d}×{len(data.lon):4d} cells, grain={grain:2d}")
        
        return pyramid, grain_map

def lonlat_for_tile(zoom, tile_x, tile_y, resolution=256):
    """
    Create lat/lon coordinates for a Web Mercator tile
    Adapted from Poseidon's lonlat4global_map
    
    Parameters
    ----------
    zoom : int
        Zoom level
    tile_x : int
        Tile X coordinate
    tile_y : int
        Tile Y coordinate
    resolution : int
        Pixels per tile
    
    Returns
    -------
    lon, lat : np.ndarray
        2D arrays of coordinates
    """
    n_tiles = 2 ** zoom
    data_bounds = {'lon_min': -81.975, 'lon_max': -49.025, 'lat_min': -20.975, 'lat_max': 5.975}
    # Total extent
    lon_min, lon_max = data_bounds['lon_min'], data_bounds['lon_max']
    lat_min, lat_max = data_bounds['lat_min'], data_bounds['lat_max']

    # Size of each tile in degrees
    tile_width_lon = (lon_max - lon_min) / n_tiles
    tile_height_lat = (lat_max - lat_min) / n_tiles

    # Tile bounds
    tile_lon_min = lon_min + tile_x * tile_width_lon
    tile_lon_max = lon_min + (tile_x + 1) * tile_width_lon
    tile_lat_min = lat_min + tile_y * tile_height_lat
    tile_lat_max = lat_min + (tile_y + 1) * tile_height_lat

    # Create pixel coordinates within this tile
    lon_1d = np.linspace(tile_lon_min, tile_lon_max, resolution)
    lat_1d = np.linspace(tile_lat_min, tile_lat_max, resolution)
    
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    
    return lon, lat

def get_tile_data(pyramid, zoom, tile_x, tile_y, time_idx=0, category_idx=0):
    """
    Extract data for a specific tile at a given zoom level
    
    Parameters
    ----------
    pyramid : dict
        Resolution pyramid from generate_pyramid
    zoom : int
        Zoom level
    tile_x, tile_y : int
        Tile coordinates
    time_idx : int
        Time index
    category_idx : int
        Category index
    
    Returns
    -------
    tile_data : np.ndarray
        256×256 array of data values
    lon, lat : np.ndarray
        Coordinate grids
    """
    # Get coordinates for this tile
    lon, lat = lonlat_for_tile(zoom, tile_x, tile_y)
    print(f"Tile bounds: lon=[{lon.min():.2f}, {lon.max():.2f}], "
        f"lat=[{lat.min():.2f}, {lat.max():.2f}]")
    
    # Get appropriate resolution data
    data = pyramid[zoom]
    # Select time and category
    data_slice = data.isel(time=time_idx, category=category_idx)

    tile_data_flat = data_slice.interp(
        lat=xr.DataArray(lat.ravel(), dims='points'),
        lon=xr.DataArray(lon.ravel(), dims='points'),
        method='nearest'
    ).values
    #print(f'\n Type {type(tile_data_flat)}')
    tile_data = tile_data_flat.reshape(lon.shape)
    
    return tile_data, lon, lat

def subsample_data_preserve_mask(data, grain):
    """
    Subsample data while preserving NaN patterns
    
    Key difference from original: Uses nanmean instead of mean,
    and preserves areas that should be NaN
    """
    if grain == 1:
        return data
    
    # Centered sampling offset, matching legacy behavior.
    lat_start = int(np.ceil(grain / 2) - 1)
    lon_start = int(np.ceil(grain / 2) - 1)

    shifted = data.isel(lat=slice(lat_start, None), lon=slice(lon_start, None))

    # Vectorized block reduction: much faster than Python loops.
    reduced = shifted.coarsen(lat=grain, lon=grain, boundary="trim")
    mean = reduced.mean(skipna=True)
    valid_count = shifted.notnull().coarsen(lat=grain, lon=grain, boundary="trim").sum()

    # Keep cells with at least 25% valid members.
    threshold = grain * grain * 0.25
    subsampled = mean.where(valid_count >= threshold)

    # Keep representative coordinates consistent with centered sampling.
    nlat = subsampled.sizes["lat"]
    nlon = subsampled.sizes["lon"]
    new_lat = data.lat.values[lat_start:lat_start + nlat * grain:grain]
    new_lon = data.lon.values[lon_start:lon_start + nlon * grain:grain]
    subsampled = subsampled.assign_coords(lat=new_lat, lon=new_lon)

    return subsampled

def save_pyramid_npz(subsampled_output_dir: Path, 
                     nc_file: Path, 
                     pyramid: dict, 
                     grain_map: dict, 
                     data_bounds: dict):
    stem = nc_file.stem  # e.g. prob_2024_dec_..._lvl_0
    out_dir = subsampled_output_dir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    zooms = sorted(pyramid.keys())

    # coords shared (time/category) — assume consistent across zooms
    z0 = zooms[0]
    da0 = pyramid[z0]
    time = da0["time"].values
    category = da0["category"].values
    meta = {
        "stem": stem,
        "zooms": zooms,
        "dtype": "float32",
        "files": {str(z): f"{stem}_z{z}.npz" for z in zooms},
        "grain_map": {str(z): int(grain_map[z]) for z in zooms},
        "data_bounds": data_bounds,
        "time_len": int(time.shape[0]),
        "category_len": int(category.shape[0]),
    }

    # Write per-zoom compressed arrays
    for z in zooms:
        da = pyramid[z]

        np.savez_compressed(
            out_dir / meta["files"][str(z)],
            values=da.values.astype(np.float32),  # (time, category, lat, lon)
            lat=da["lat"].values.astype(np.float32),
            lon=da["lon"].values.astype(np.float32),
            time=time,          # datetime64 ok in npz
            category=category,  # small
            grain=np.array([grain_map[z]], dtype=np.int16),
        )

    # Write small metadata json
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
