## Summary of the Solution
The Problem:
- RGBA PNG tiles with alpha channel (transparent pixels as [0,0,0,0]) were rendering as white in browsers
- Windows Photos showed transparency correctly, but Leaflet/Plotly browsers showed white backgrounds

### The Solution:
Convert images from RGBA mode to palette mode (indexed color) with a transparency index
This matches how WMS servers serve transparent tiles
Browsers handle indexed transparency much better than RGBA alpha channels

### Key Changes Made:
1. Tile Server (tile_server_pyramid.py): Convert RGBA images to palette mode with transparency index
2. XYZ/TMS Coordinate Support: Added tms parameter to handle both coordinate systems ipyleaflet Integration (ipyleaflet.ipynb): Uses mode=global&tms=false for proper tile alignment
3. vmin/vmax Control: Query parameters for custom colormap ranges

Both ipyleaflet and Plotly should now display tiles correctly with:
- ✅ Proper transparency (ocean/land masks show basemap underneath)
- ✅ Correct geographic location (Amazon region only)
- ✅ Custom vmin/vmax color scaling
- ✅ Efficient pyramid-based multi-resolution rendering