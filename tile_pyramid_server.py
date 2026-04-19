"""
Pyramid-based tile server for HydroViewer
Optimized for Shiny + ipyleaflet integration
"""

from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin
from pathlib import Path
import json
import os
import hashlib
import gc
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#import shared
#from pathlib import Path
app = Flask(__name__)
CORS(app)

# Configuration
TILE_SIZE = 256
BACKEND_DIR = os.getcwd()
PYRAMID_DIR = os.path.join(BACKEND_DIR, 'get_ldas_probabilistic_output', 'subsampled')
TILE_IMAGE_CACHE = {}  # Cache rendered tiles
API_VERSION = "2026-02-08"
MAX_META_CACHE = 1
MAX_TILE_CACHE = 50

class RegionalTileServer:
    """Serves tiles from pyramids using regional coordinates"""
    
    def __init__(self):
        # Metadata-only cache:
        # pyramids[cache_key] = {"meta": {...}}
        self.pyramids = {}
        self.data_bounds = None
        self._index_cache = None
        self._stem_cache = {}

    def _is_remote(self):
        return str(PYRAMID_DIR).startswith(("http://", "https://"))

    def _join_ref(self, base, *parts):
        if self._is_remote():
            out = str(base).rstrip("/") + "/"
            for part in parts:
                out = urljoin(out, str(part).lstrip("/"))
            return out
        return str(Path(base).joinpath(*parts))

    def _read_json_ref(self, ref):
        if str(ref).startswith(("http://", "https://")):
            r = requests.get(ref, timeout=30)
            if r.status_code == 404:
                raise FileNotFoundError(f"JSON not found at: {ref}")
            r.raise_for_status()
            return r.json()
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"JSON not found at: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _open_npz_ref(self, ref):
        if str(ref).startswith(("http://", "https://")):
            r = requests.get(ref, timeout=60)
            if r.status_code == 404:
                raise FileNotFoundError(f"NPZ not found at: {ref}")
            r.raise_for_status()
            return np.load(BytesIO(r.content), allow_pickle=False)
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"NPZ not found at: {path}")
        return np.load(str(path), allow_pickle=False)

    def load_index(self):
        if self._index_cache is not None:
            return self._index_cache
        index_ref = self._join_ref(PYRAMID_DIR, "index.json")
        try:
            self._index_cache = self._read_json_ref(index_ref)
        except FileNotFoundError:
            self._index_cache = {}
        return self._index_cache
    
    def _stem(self, variable: str) -> str:
        cache_key = variable
        if cache_key in self._stem_cache:
            return self._stem_cache[cache_key]

        if "_tercile_" in variable:
            self._stem_cache[cache_key] = variable
            return variable

        if not self._is_remote():
            base = Path(PYRAMID_DIR)
            if not base.exists():
                raise FileNotFoundError(f"Pyramid directory not found: {base}")

            suffixes = [
                f"_tercile_probability_max_{variable}",
                f"_tercile_prob_max_{variable}",
            ]
            matches = sorted(
                p.name
                for p in base.iterdir()
                if p.is_dir() and any(p.name.endswith(sfx) for sfx in suffixes)
            )
            if matches:
                self._stem_cache[cache_key] = matches[0]
                return matches[0]

        index = self.load_index()
        init_date = index.get("initialization_date")
        if init_date:
            remote_stem = f"{init_date}_tercile_prob_max_{variable}"
            self._stem_cache[cache_key] = remote_stem
            return remote_stem

        raise FileNotFoundError(
            f"Could not resolve pyramid stem for variable '{variable}' in {PYRAMID_DIR}"
        )
    
    def _base_dir_url(self, variable: str) -> str:
        """
        Return base directory reference for variable data.
        """
        stem = self._stem(variable)
        return self._join_ref(PYRAMID_DIR, stem)
    
    def load_pyramid_meta(self, variable):
        """Load pyramid metadata from remote."""
        cache_key = self._stem(variable)
        if cache_key in self.pyramids:
            return self.pyramids[cache_key]["meta"]
        
        base = self._base_dir_url(variable)
        meta_ref = self._join_ref(base, f"{cache_key}_meta.json")
        meta = self._read_json_ref(meta_ref)
        self.data_bounds = meta.get("data_bounds")

        if len(self.pyramids) >= MAX_META_CACHE:
            oldest_key = next(iter(self.pyramids))
            # print(f"[META CACHE] Evicting {oldest_key}")
            del self.pyramids[oldest_key]

        self.pyramids[cache_key] = {"meta": meta}
        return meta
    
    def load_pyramid_bundle(self, variable, z):
        """Load one zoom-level npz bundle for one request."""
        meta = self.load_pyramid_meta(variable)
        base = self._base_dir_url(variable)

        files = meta.get("files")
        z_key = str(int(z))

        if isinstance(files, dict):
            npz_name = files.get(z_key)
            if not npz_name:
                raise KeyError(f"Metadata missing files[{z_key}]")
        elif isinstance(files, str):
            npz_name = files
        else:
            raise KeyError("Metadata must contain 'files' as dict or string")

        npz_ref = self._join_ref(base, npz_name)
        return self._open_npz_ref(npz_ref)

    def _resolve_time_idx(self, time_input, time_values):
        """Resolve time selector (index or ISO date/time string) to integer index."""
        if isinstance(time_input, (int, np.integer)):
            return int(time_input)

        time_text = str(time_input).strip().strip("'\"")
        if time_text.lstrip("-").isdigit():
            return int(time_text)

        time_strings = []
        for t in time_values:
            if isinstance(t, np.datetime64):
                time_strings.append(np.datetime_as_string(t, unit='s'))
            else:
                time_strings.append(str(t))

        # First try exact timestamp match
        if time_text in time_strings:
            return time_strings.index(time_text)

        # Then allow date-only match (YYYY-MM-DD)
        date_text = time_text[:10]
        matching = [i for i, value in enumerate(time_strings) if str(value)[:10] == date_text]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            raise ValueError(
                f"Ambiguous time input '{time_input}' matched multiple timestamps for date '{date_text}'"
            )

        available_dates = sorted({str(v)[:10] for v in time_strings})
        raise ValueError(
            f"Unknown time input '{time_input}'. Available dates: {available_dates}"
        )

    def get_level_slice(self, variable: str, 
                        z: int, 
                        time_input, 
                        category_idx: int, 
                        profile_idx: int = 0):
        """Return one 2D slice (lat, lon) for a given zoom/time/category/profile."""
        meta = self.load_pyramid_meta(variable)
        bundle = self.load_pyramid_bundle(variable, z)
        try:
            z = int(z)
            available = [int(v) for v in meta.get('zooms', [])]
            if z not in available:
                raise KeyError(f'Zoom {z} not in available. Available: {available}')

            if f"z{z}_values" in bundle.files:
                values = bundle[f"z{z}_values"]
                lat = bundle[f"z{z}_lat"]
                lon = bundle[f"z{z}_lon"]
            else:
                values = bundle["values"]
                lat = bundle["lat"]
                lon = bundle["lon"]
            time = bundle["time"]
            category = bundle["category"]
            profile_dim = meta.get('profile_dim')
            time_idx = self._resolve_time_idx(time_input, time)

            if time_idx < 0 or time_idx >= len(time):
                raise IndexError(f"time_idx out of range: {time_idx} (0..{len(time)-1})")
            if category_idx < 0 or category_idx >= len(category):
                raise IndexError(f"category out of range: {category_idx} (0..{len(category)-1})")

            if profile_dim is None:
                expected_shape_tc = (len(time), len(category), len(lat), len(lon))
                expected_shape_ct = (len(category), len(time), len(lat), len(lon))
                actual_shape = tuple(values.shape)

                if actual_shape == expected_shape_tc:
                    values_2d = values[time_idx, category_idx, :, :]
                elif actual_shape == expected_shape_ct:
                    values_2d = values[category_idx, time_idx, :, :]
                else:
                    raise ValueError(
                        "Unexpected NPZ shape for "
                        f"{variable} z={z}: values={actual_shape}, "
                        f"expected (time,category,lat,lon)={expected_shape_tc} or "
                        f"(category,time,lat,lon)={expected_shape_ct}"
                    )
            else:
                profile_depth = bundle['profile_depth']
                if profile_idx < 0 or profile_idx >= len(profile_depth):
                    raise IndexError(f"profile out of range: {profile_idx} (0..{len(profile_depth)-1})")

                expected_shape_tcp = (len(time), len(category), len(profile_depth), len(lat), len(lon))
                expected_shape_ctp = (len(category), len(time), len(profile_depth), len(lat), len(lon))
                actual_shape = tuple(values.shape)

                if actual_shape == expected_shape_tcp:
                    values_2d = values[time_idx, category_idx, profile_idx, :, :]
                elif actual_shape == expected_shape_ctp:
                    values_2d = values[category_idx, time_idx, profile_idx, :, :]
                else:
                    raise ValueError(
                        f"Unexpected profiled NPZ shape for {variable} - Z={z}: "
                        f"{actual_shape}, expected {expected_shape_tcp} or {expected_shape_ctp}"
                    )

            return values_2d, lat, lon
        finally:
            bundle.close()

    def get_best_zoom(self, variable: str, requested_zoom: int) -> int:
        """Pick nearest available zoom from metadata."""
        meta = self.load_pyramid_meta(variable)
        available = sorted(int(z) for z in meta.get("zooms", []))
        if not available:
            raise KeyError("No zoom levels found in metadata 'zooms'.")
        if requested_zoom in available:
            return requested_zoom
        return min(available, key=lambda k: abs(k - requested_zoom))

    @staticmethod
    def warn_if_level_query_present(level):
        """Temporary backward compatibility path for deprecated `level` query param."""
        if level is not None:
            print(
                f"[DEPRECATED] Ignoring query parameter 'level={level}'. "
                "Depth selection now uses 'profile' only."
            )

    def tile_to_lonlat_bounds(self, zoom, x, y):
        """
        Convert Web Mercator tile coordinates to lat/lon bounds
        Uses global Web Mercator projection (world = 2^zoom tiles)

        Returns
        -------
        dict with keys: lon_min, lon_max, lat_min, lat_max
        """
        n_tiles = 2 ** zoom

        # Longitude is simple linear mapping
        lon_min = x / n_tiles * 360.0 - 180.0
        lon_max = (x + 1) / n_tiles * 360.0 - 180.0

        # Latitude uses Web Mercator inverse projection
        def mercator_to_lat(y_tile, n_tiles):
            """Convert tile Y coordinate to latitude"""
            # In TMS: y=0 is south pole, y=n_tiles is north pole
            n = np.pi - 2.0 * np.pi * y_tile / n_tiles
            return 180.0 / np.pi * np.arctan(0.5 * (np.exp(n) - np.exp(-n)))

        # TMS coordinates: y=0 at south
        lat_min = mercator_to_lat(y + 1, n_tiles)
        lat_max = mercator_to_lat(y, n_tiles)

        return {
            'lon_min': lon_min,
            'lon_max': lon_max,
            'lat_min': lat_min,
            'lat_max': lat_max
        }

    def get_tile_lonlat_grids(self, zoom, x, y, resolution=256, mode='regional'):
        """
        Create lat/lon coordinate grids for a tile

        Parameters
        ----------
        zoom, x, y : int
            Tile coordinates
        resolution : int
            Pixels per tile
        mode : str
            'regional' - tile coordinates within data bounds only
            'global' - standard Web Mercator tile coordinates

        Returns
        -------
        lon, lat : np.ndarray or None
            2D arrays of coordinates, or None if tile is outside data bounds
        """
        if not self.data_bounds:
            raise ValueError("Data bounds not loaded")

        if mode == 'global':
            # Get global tile bounds
            tile_bounds = self.tile_to_lonlat_bounds(zoom, x, y)
            tile_lon_min = tile_bounds['lon_min']
            tile_lon_max = tile_bounds['lon_max']
            tile_lat_min = tile_bounds['lat_min']
            tile_lat_max = tile_bounds['lat_max']

            # Check if tile intersects data bounds
            data_lon_min = self.data_bounds['lon_min']
            data_lon_max = self.data_bounds['lon_max']
            data_lat_min = self.data_bounds['lat_min']
            data_lat_max = self.data_bounds['lat_max']

            if (tile_lon_max < data_lon_min or tile_lon_min > data_lon_max or
                tile_lat_max < data_lat_min or tile_lat_min > data_lat_max):
                return None  # Tile is outside data bounds

        else:  # mode == 'regional'
            n_tiles = 2 ** zoom

            # Total extent
            lon_min, lon_max = self.data_bounds['lon_min'], self.data_bounds['lon_max']
            lat_min, lat_max = self.data_bounds['lat_min'], self.data_bounds['lat_max']

            # Size of each tile in degrees
            tile_width_lon = (lon_max - lon_min) / n_tiles
            tile_height_lat = (lat_max - lat_min) / n_tiles

            # TMS tile coordinates: Y=0 at south (bottom)
            tile_lon_min = lon_min + x * tile_width_lon
            tile_lon_max = lon_min + (x + 1) * tile_width_lon
            tile_lat_min = lat_min + y * tile_height_lat
            tile_lat_max = lat_min + (y + 1) * tile_height_lat

        # Create pixel coordinates within this tile
        lon_1d = np.linspace(tile_lon_min, tile_lon_max, resolution)
        # Latitude from max to min so north is at top of image (row 0)
        lat_1d = np.linspace(tile_lat_max, tile_lat_min, resolution)

        lon, lat = np.meshgrid(lon_1d, lat_1d)

        return lon, lat
    
    def _nearest_indices_1d(self, coord_1d: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Nearest-neighbor index lookup for monotonic 1D coordinates."""
        coord = np.asarray(coord_1d)
        tgt = np.asarray(target)

        if coord.ndim != 1:
            raise ValueError("Coordinate must be 1D")
        if coord.size == 0:
            raise ValueError("Coordinate is empty")
        if coord.size == 1:
            return np.zeros(tgt.shape, dtype=np.int64)

        if coord[0] <= coord[-1]:
            idx = np.searchsorted(coord, tgt, side='left')
            idx = np.clip(idx, 1, coord.size - 1)
            left = coord[idx - 1]
            right = coord[idx]
            choose_left = np.abs(tgt - left) <= np.abs(right - tgt)
            return idx - choose_left.astype(np.int64)

        # descending coordinates: search on reversed array and map back
        coord_rev = coord[::-1]
        idx_rev = np.searchsorted(coord_rev, tgt, side='left')
        idx_rev = np.clip(idx_rev, 1, coord_rev.size - 1)
        left = coord_rev[idx_rev - 1]
        right = coord_rev[idx_rev]
        choose_left = np.abs(tgt - left) <= np.abs(right - tgt)
        nearest_rev = idx_rev - choose_left.astype(np.int64)
        return (coord.size - 1) - nearest_rev

    def get_tile_data(self, values_2d, src_lon, src_lat, tile_lon, tile_lat):
        """
        Sample a 2D source grid to the tile grid with nearest-neighbor lookup.

        Parameters
        ----------
        values_2d : np.ndarray
            Source data on [lat, lon] axes
        src_lon, src_lat : np.ndarray
            1D source coordinate arrays
        tile_lon, tile_lat : np.ndarray
            Tile meshgrid arrays

        Returns
        -------
        tile_data : np.ndarray
            Resampled data for the tile
        """
        if values_2d.ndim != 2:
            raise ValueError(f"values_2d must be 2D [lat,lon], got shape {values_2d.shape}")
        if values_2d.shape != (len(src_lat), len(src_lon)):
            raise ValueError(
                f"values shape mismatch: values={values_2d.shape}, "
                f"lat={len(src_lat)}, lon={len(src_lon)}"
            )

        src_lon = np.asarray(src_lon)
        src_lat = np.asarray(src_lat)
        tile_lon_flat = tile_lon.ravel()
        tile_lat_flat = tile_lat.ravel()

        lon_min = float(np.min(src_lon))
        lon_max = float(np.max(src_lon))
        lat_min = float(np.min(src_lat))
        lat_max = float(np.max(src_lat))

        # Prevent edge-clamping artifacts: only sample pixels that are inside
        # the source coordinate extent; outside pixels become transparent later.
        in_bounds = (
            (tile_lon_flat >= lon_min) & (tile_lon_flat <= lon_max) &
            (tile_lat_flat >= lat_min) & (tile_lat_flat <= lat_max)
        )

        sampled_flat = np.full(tile_lon_flat.shape, np.nan, dtype=np.float32)
        if np.any(in_bounds):
            lon_idx = self._nearest_indices_1d(src_lon, tile_lon_flat[in_bounds])
            lat_idx = self._nearest_indices_1d(src_lat, tile_lat_flat[in_bounds])
            sampled_vals = values_2d[lat_idx, lon_idx]
            sampled_flat[in_bounds] = sampled_vals.astype(np.float32, copy=False)

        return sampled_flat.reshape(tile_lon.shape)
    
    def create_colormap_image(self, data, colormap_name, vmin, vmax):
        """Create RGBA image from data"""
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            return Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

        # Auto-scale if needed
        if vmin is None:
            vmin = np.nanpercentile(valid_data, 2)
        if vmax is None:
            vmax = np.nanpercentile(valid_data, 98)

        if vmin == vmax:
            vmax = vmin + 1

        # Create a mask for NaN values before applying colormap
        nan_mask = np.isnan(data)

        # print(f"  NaN mask shape: {nan_mask.shape}, NaN count: {nan_mask.sum()}")

        # Replace NaN with vmin temporarily (will be made transparent later)
        data_filled = np.where(nan_mask, vmin, data)

        # Normalize and colorize
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        try:
            cmap = plt.get_cmap(colormap_name)
        except:
            cmap = plt.get_cmap('Reds')

        rgba = cmap(norm(data_filled))
        rgba_uint8 = (rgba * 255).astype(np.uint8)

        # print(f"  RGBA shape: {rgba_uint8.shape}")

        # Make NaN values completely transparent
        # Set all channels to 0 for fully transparent pixels
        rgba_uint8[nan_mask, :] = 0

        # Verify transparency was applied
        transparent_count = (rgba_uint8[:, :, 3] == 0).sum()
        # print(f"  Transparent pixels: {transparent_count} (should be {nan_mask.sum()})")

        img = Image.fromarray(rgba_uint8, mode='RGBA')

        # Verify the image mode
        # print(f"  Image mode: {img.mode}, size: {img.size}")

        # Convert to palette mode with transparency (like WMS)
        # This ensures browser compatibility
        # First, create an alpha mask
        alpha = img.split()[3]  # Get alpha channel

        # Convert RGB to palette mode
        img_rgb = img.convert('RGB')
        img_p = img_rgb.convert('P', palette=Image.ADAPTIVE, colors=255)

        # Add transparency for pixels that were fully transparent
        # Find a color index to use for transparent pixels (use index 0)
        img_p.paste(0, mask=Image.eval(alpha, lambda a: 255 if a == 0 else 0))
        img_p.info['transparency'] = 0

        # print(f"  Converted to palette mode with transparency index")

        return img_p
    
    def thicken_sparse_features(self, tile_data, passes=1):
        """
        Expand sparse valid pixels into immediate neighbors.
        Useful for line-like fields (e.g., streamflow) that can look like
        they disappear at high zoom due to sub-pixel width.
        """
        data = np.asarray(tile_data, dtype=np.float32)
        out = data.copy()

        for _ in range(max(1, int(passes))):
            base = out.copy()
            nan_mask = np.isnan(out)
            if not np.any(nan_mask):
                break

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue

                    dst_y0 = max(0, dy)
                    dst_y1 = min(out.shape[0], out.shape[0] + dy)
                    dst_x0 = max(0, dx)
                    dst_x1 = min(out.shape[1], out.shape[1] + dx)

                    src_y0 = max(0, -dy)
                    src_y1 = min(base.shape[0], base.shape[0] - dy)
                    src_x0 = max(0, -dx)
                    src_x1 = min(base.shape[1], base.shape[1] - dx)

                    src = base[src_y0:src_y1, src_x0:src_x1]
                    dst = out[dst_y0:dst_y1, dst_x0:dst_x1]
                    dst_nan = np.isnan(dst)
                    src_valid = np.isfinite(src)
                    fill = dst_nan & src_valid
                    if np.any(fill):
                        dst[fill] = src[fill]

            return out

# Global server instance
tile_server = RegionalTileServer()


@app.route('/tiles/<variable>/<time_input>/<int:category>/<int:z>/<int:x>/<int:y>.png')
def get_tile(variable, time_input, category, z, x, y):
    """
    Serve a tile
    
    Query params:
    - colormap: matplotlib colormap name
    - vmin, vmax: color scale range
    - cache: enable/disable caching
    """
    colormap = request.args.get('colormap', 'Reds')
    vmin = request.args.get('vmin', type=float)
    vmax = request.args.get('vmax', type=float)
    profile = request.args.get('profile', 0, type=int)
    level = request.args.get('level', type=int)
    use_cache = request.args.get('cache', 'true').lower() == 'true'
    mode = request.args.get('mode', 'regional')  # 'regional' or 'global'
    tms = request.args.get('tms', 'false').lower() == 'true'  # TMS vs XYZ coordinates
    tile_server.warn_if_level_query_present(level)

    # print(f"Tile request {z}/{x}/{y}: vmin={vmin}, vmax={vmax}, mode={mode}, tms={tms}")

    # Build cache key BEFORE any coordinate conversion
    cache_key = f"{variable}_{time_input}_{category}_{z}_{x}_{y}_{colormap}_{vmin}_{vmax}_{profile}_{mode}_{tms}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    # Convert XYZ to TMS if needed (for global mode)
    if mode == 'global' and not tms:
        # XYZ: Y=0 at north, need to flip to TMS (Y=0 at south)
        n_tiles = 2 ** z
        y = (n_tiles - 1) - y
        # print(f"  Converted XYZ to TMS: y_new={y}")

    if use_cache and cache_hash in TILE_IMAGE_CACHE:
        response = send_file(
            BytesIO(TILE_IMAGE_CACHE[cache_hash]),
            mimetype='image/png'
        )
        response.headers['Cache-Control'] = 'public, max-age=300'
        # response.headers['Pragma'] = 'no-cache'
        # response.headers['Expires'] = '0'
        return response

    values_2d = None
    src_lat = None
    src_lon = None
    tile_data = None
    try:
        # Resolve nearest available zoom and load one 2D source slice
        z_actual = tile_server.get_best_zoom(variable, z)
        values_2d, src_lat, src_lon = tile_server.get_level_slice(
            variable, z_actual, time_input, category, profile
        )
        
        # If request overzooms beyond available data, sample from the parent tile
        # at z_actual so features stay visible instead of collapsing to NaN.
        if z > z_actual:
            dz = z - z_actual
            factor = 2 ** dz
            x_sample = x // factor
            y_sample = y // factor
            z_sample = z_actual
        else:
            x_sample = x
            y_sample = y
            z_sample = z

        # Get tile coordinate grids
        grids = tile_server.get_tile_lonlat_grids(z_sample, x_sample, y_sample, TILE_SIZE, mode=mode)

        # If tile is outside data bounds (in global mode), return transparent tile
        if grids is None:
            image = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
            img_io = BytesIO()
            image.save(img_io, 'PNG')
            response = send_file(BytesIO(img_io.getvalue()), mimetype='image/png')
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        lon, lat = grids

        # Resample source grid to tile grid with NumPy nearest neighbor
        tile_data = tile_server.get_tile_data(values_2d, src_lon, src_lat, lon, lat)

        if 'streamflow'in variable.lower():
            tile_data = tile_server.thicken_sparse_features(tile_data, passes=1)

        # Debug: Check NaN percentage
        nan_pct = np.isnan(tile_data).sum() / tile_data.size * 100
        # print(f"Tile {z}/{x}/{y}: {nan_pct:.1f}% NaN values")

        # Create image
        image = tile_server.create_colormap_image(tile_data, colormap, vmin, vmax)
        
        # Save to bytes with proper alpha channel
        img_io = BytesIO()

        # Add PNG metadata to explicitly mark alpha channel
        # from PIL import PngImagePlugin
        # pnginfo = PngImagePlugin.PngInfo()
        # pnginfo.add_text("Software", "HydroViewer Tile Server")

        # Save with explicit RGBA mode and metadata
        image.save(img_io, 'PNG') # pnginfo=pnginfo, optimize=True
        img_bytes = img_io.getvalue()
        
        # Cache
        if use_cache:
            TILE_IMAGE_CACHE[cache_hash] = img_bytes
            
            # Limit cache size
            if len(TILE_IMAGE_CACHE) > MAX_TILE_CACHE:
                TILE_IMAGE_CACHE.pop(next(iter(TILE_IMAGE_CACHE)))
        
        response = send_file(BytesIO(img_bytes), mimetype='image/png')
        # Prevent browser caching to ensure transparency updates are visible
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        # Enable CORS for cross-origin requests
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET'
        # Explicitly set content type with alpha channel info
        response.headers['Content-Type'] = 'image/png'
        return response

    except Exception as e:
        print(f"Tile generation error: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500
    finally:
        del values_2d
        del src_lat
        del src_lon
        del tile_data
        # gc.collect()
        pass


@app.route('/pyramid/info/<variable>')
def pyramid_info(variable):
    """Get pyramid information"""
    profile = request.args.get('profile', 0, type=int)
    level = request.args.get('level', type=int)
    tile_server.warn_if_level_query_present(level)

    try:
        meta = tile_server.load_pyramid_meta(variable)
        zoom_levels = sorted(int(z) for z in meta.get("zooms", []))

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj

        info = {
            'variable': variable,
            'profile': int(profile),
            'zoom_levels': zoom_levels,
            'data_bounds': convert_to_native(meta.get('data_bounds')),
            'files': convert_to_native(meta.get('files')),
            'levels': {}
        }

        for zoom in zoom_levels:
            bundle = tile_server.load_pyramid_bundle(variable, zoom)
            try:
                if f"z{zoom}_lat" in bundle.files:
                    lat = bundle[f"z{zoom}_lat"]
                    lon = bundle[f"z{zoom}_lon"]
                else:
                    lat = bundle["lat"]
                    lon = bundle["lon"]
                info['levels'][str(zoom)] = {
                    'shape': [int(len(lat)), int(len(lon))]
                }
            finally:
                bundle.close()

        return jsonify(info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/pyramid/time/<variable>')
def pyramid_time(variable):
    """Get available time coordinates for a variable."""
    profile = request.args.get('profile', 0, type=int)
    level = request.args.get('level', type=int)
    tile_server.warn_if_level_query_present(level)

    try:
        meta = tile_server.load_pyramid_meta(variable)
        zoom_levels = sorted(int(z) for z in meta.get("zooms", []))
        if not zoom_levels:
            raise KeyError("No zoom levels found in metadata 'zooms'.")

        # Time coordinate is shared across levels; read once from request-scoped bundle.
        bundle = tile_server.load_pyramid_bundle(variable, zoom_levels[0])
        try:
            time_values = bundle["time"]
        finally:
            bundle.close()

        time_iso = []
        for t in time_values:
            if isinstance(t, np.datetime64):
                time_iso.append(np.datetime_as_string(t, unit='s'))
            else:
                time_iso.append(str(t))

        return jsonify({
            'variable': variable,
            'profile': int(profile),
            'time': time_iso,
            'count': len(time_iso),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test/save_tile/<variable>/<time_input>/<int:category>/<int:z>/<int:x>/<int:y>')
def save_test_tile(variable, time_input, category, z, x, y):
    """Save a test tile to disk to verify transparency"""
    colormap = request.args.get('colormap', 'Reds')
    vmin = request.args.get('vmin', type=float)
    vmax = request.args.get('vmax', type=float)
    profile = request.args.get('profile', 0, type=int)
    level = request.args.get('level', type=int)
    mode = request.args.get('mode', 'global')
    tile_server.warn_if_level_query_present(level)

    values_2d = None
    src_lat = None
    src_lon = None
    tile_data = None
    try:
        z_actual = tile_server.get_best_zoom(variable, z)
        values_2d, src_lat, src_lon = tile_server.get_level_slice(
            variable, z_actual, time_input, category, profile
        )

        grids = tile_server.get_tile_lonlat_grids(z, x, y, TILE_SIZE, mode=mode)

        if grids is None:
            return "Tile outside data bounds", 404

        lon, lat = grids
        tile_data = tile_server.get_tile_data(values_2d, src_lon, src_lat, lon, lat)
        image = tile_server.create_colormap_image(tile_data, colormap, vmin, vmax)

        # Save to disk
        output_path = f'test_tile_{z}_{x}_{y}.png'
        image.save(output_path, 'PNG')

        return f"Saved to {output_path}. Open it to verify transparency."

    except Exception as e:
        return str(e), 500
    finally:
        del values_2d
        del src_lat
        del src_lon
        del tile_data
        # gc.collect()
        pass


@app.route('/cache/clear')
def clear_cache():
    """Clear tile cache"""
    TILE_IMAGE_CACHE.clear()
    return jsonify({'message': 'Cache cleared'})


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'api_version': API_VERSION,
        'status': 'ok',
        'pyramids_loaded': len(tile_server.pyramids),
        'tiles_cached': len(TILE_IMAGE_CACHE)
    })


if __name__ == '__main__':
    print("="*60)
    print("HydroViewer Pyramid Tile Server")
    print("="*60)

    port = int(os.environ.get('PORT', 4000))
    print(f"Starting on http://localhost:{port}")
    print("Tiles: /tiles/{var}/{time}/{cat}/{z}/{x}/{y}.png")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)
