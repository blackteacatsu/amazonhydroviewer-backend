"""
Pyramid-based tile server for HydroViewer
Optimized for Shiny + ipyleaflet integration
"""

from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from pathlib import Path
import hashlib
import xarray as xr
#import shared

app = Flask(__name__)
CORS(app)

# Configuration
TILE_SIZE = 256
PYRAMID_DIR = Path('testbench/data/pyramids')
PYRAMID_CACHE = {}  # Cache loaded pyramids
TILE_IMAGE_CACHE = {}  # Cache rendered tiles


class RegionalTileServer:
    """Serves tiles from pyramids using regional coordinates"""
    
    def __init__(self):
        self.pyramids = {}
        self.data_bounds = None
    
    def load_pyramid(self, variable, profile=0):
        """Load pyramid from disk"""
        cache_key = f"{variable}_lvl_{profile}"
        
        if cache_key in self.pyramids:
            return self.pyramids[cache_key]
        
        pyramid_file = PYRAMID_DIR / f"pyramid_{variable}_lvl_{profile}.pkl"
        
        if not pyramid_file.exists():
            raise FileNotFoundError(f"Pyramid not found: {pyramid_file}")
        
        with open(pyramid_file, 'rb') as f:
            pyramid_data = pickle.load(f)
        
        self.pyramids[cache_key] = pyramid_data
        self.data_bounds = pyramid_data['data_bounds']
        
        print("\n" + "="*60)        
        print(f"Loaded pyramid: {cache_key}, \n"
              f"zoom levels: {list(pyramid_data['pyramid'].keys())}, \n"
              f"bounds: {pyramid_data['data_bounds']}" )
        print("="*60 + "\n")
        
        return pyramid_data
    
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
    
    def get_tile_data(self, data_slice, x, y):
        """
        Interpolate data to tile grid using xarray's interp method
        Matches the working get_tile_data function from subsample_pyramid.py

        Parameters
        ----------
        data_slice : xr.DataArray
            Data slice for a specific time and category
        lon, lat : np.ndarray
            Meshgrid of lon/lat coordinates for the tile

        Returns
        -------
        tile_data : np.ndarray
            Interpolated data for the tile
        """
        # Use xarray's interp method (same as working code)
        tile_data_flat = data_slice.interp(
            lat=xr.DataArray(y.ravel(), dims='points'),
            lon=xr.DataArray(x.ravel(), dims='points'),
            method='nearest'
        ).values

        tile_data = tile_data_flat.reshape(x.shape)

        return tile_data
    
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

        # Make NaN values completely transparent
        rgba_uint8[nan_mask] = np.array([0, 0, 0, 0], dtype=np.uint8)

        return Image.fromarray(rgba_uint8, mode='RGBA')


# Global server instance
tile_server = RegionalTileServer()


@app.route('/tiles/<variable>/<int:time_idx>/<int:category>/<int:z>/<int:x>/<int:y>.png')
def get_tile(variable, time_idx, category, z, x, y):
    """
    Serve a tile
    
    Query params:
    - colormap: matplotlib colormap name
    - vmin, vmax: color scale range
    - profile: depth profile (0-3)
    - cache: enable/disable caching
    """
    colormap = request.args.get('colormap', 'Reds')
    vmin = request.args.get('vmin', type=float)
    vmax = request.args.get('vmax', type=float)
    profile = request.args.get('profile', 0, type=int)
    use_cache = request.args.get('cache', 'true').lower() == 'true'
    mode = request.args.get('mode', 'regional')  # 'regional' or 'global'

    # Check cache
    cache_key = f"{variable}_{time_idx}_{category}_{z}_{x}_{y}_{colormap}_{vmin}_{vmax}_{profile}_{mode}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    if use_cache and cache_hash in TILE_IMAGE_CACHE:
        return send_file(
            BytesIO(TILE_IMAGE_CACHE[cache_hash]),
            mimetype='image/png'
        )

    try:
        # Load pyramid
        pyramid_data = tile_server.load_pyramid(variable, profile)
        pyramid = pyramid_data['pyramid']

        # Select appropriate zoom level
        if z not in pyramid:
            available_zooms = sorted(pyramid.keys())
            z_actual = min(available_zooms, key=lambda k: abs(k - z))
        else:
            z_actual = z

        data = pyramid[z_actual]

        # Select time and category
        data_slice = data.isel(time=time_idx, category=category)

        # Get tile coordinate grids
        grids = tile_server.get_tile_lonlat_grids(z, x, y, TILE_SIZE, mode=mode)

        # If tile is outside data bounds (in global mode), return transparent tile
        if grids is None:
            image = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
            img_io = BytesIO()
            image.save(img_io, 'PNG', optimize=True)
            return send_file(BytesIO(img_io.getvalue()), mimetype='image/png')

        lon, lat = grids

        # Interpolate to tile (using working xarray method)
        tile_data = tile_server.get_tile_data(data_slice, lon, lat)

        # Debug: Check NaN percentage
        nan_pct = np.isnan(tile_data).sum() / tile_data.size * 100
        print(f"Tile {z}/{x}/{y}: {nan_pct:.1f}% NaN values")

        # Create image
        image = tile_server.create_colormap_image(tile_data, colormap, vmin, vmax)
        
        # Save to bytes
        img_io = BytesIO()
        image.save(img_io, 'PNG', optimize=True)
        img_bytes = img_io.getvalue()
        
        # Cache
        if use_cache:
            TILE_IMAGE_CACHE[cache_hash] = img_bytes
            
            # Limit cache size
            if len(TILE_IMAGE_CACHE) > 1000:
                for _ in range(100):
                    TILE_IMAGE_CACHE.pop(next(iter(TILE_IMAGE_CACHE)))
        
        return send_file(BytesIO(img_bytes), mimetype='image/png')
        
    except Exception as e:
        print(f"Tile generation error: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500


@app.route('/pyramid/info/<variable>')
def pyramid_info(variable):
    """Get pyramid information"""
    profile = request.args.get('profile', 0, type=int)

    try:
        pyramid_data = tile_server.load_pyramid(variable, profile)

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
            'zoom_levels': [int(z) for z in pyramid_data['pyramid'].keys()],
            'grain_map': convert_to_native(pyramid_data['grain_map']),
            'data_bounds': convert_to_native(pyramid_data['data_bounds']),
            'levels': {}
        }

        for zoom, data in pyramid_data['pyramid'].items():
            info['levels'][str(zoom)] = {
                'shape': [int(len(data.lat)), int(len(data.lon))],
                'grain': int(pyramid_data['grain_map'][zoom])
            }

        return jsonify(info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/clear')
def clear_cache():
    """Clear tile cache"""
    TILE_IMAGE_CACHE.clear()
    return jsonify({'message': 'Cache cleared'})


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'pyramids_loaded': len(tile_server.pyramids),
        'tiles_cached': len(TILE_IMAGE_CACHE)
    })


if __name__ == '__main__':
    print("="*60)
    print("HydroViewer Pyramid Tile Server")
    print("="*60)
    print("Starting on http://localhost:5000")
    print("Tiles: /tiles/{var}/{time}/{cat}/{z}/{x}/{y}.png")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)