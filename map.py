# British Isles pixel map — generated from real geographic boundaries
# (Historic County Borders Project for the UK, GADM/Tailte Éireann-derived
# data for Ireland), not hand-drawn. Paste this whole thing into one cell
# and run it. Takes a few minutes: it's rasterizing ~120 real polygons at
# high resolution.
#
# Needs historic_counties.geojson and ireland_counties.geojson sitting
# next to this script.

import json
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

SEA  = (247, 245, 240)   # background
LAND = (166, 166, 166)   # grey for all land

SCALE = 3  # pixel density relative to the original 155x236 hand-drawn grid
BASE_WIDTH, BASE_HEIGHT = 155, 236
TARGET_WIDTH = BASE_WIDTH * SCALE

with open("historic_counties.geojson") as f:
    uk_counties = json.load(f)["features"]
with open("ireland_counties.geojson") as f:
    ireland_counties = json.load(f)["features"]

def polygons_of(feat):
    geom = feat["geometry"]
    return geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]

# Project real (lon, lat) onto the pixel grid: equirectangular, fit to the
# combined UK+Ireland extent, with a latitude correction (cos of the
# mid-latitude) so the country isn't stretched east-west.
all_points = [pt for feat in uk_counties + ireland_counties
              for poly in polygons_of(feat) for ring in poly for pt in ring]
lons = [p[0] for p in all_points]
lats = [p[1] for p in all_points]
lon_min, lon_max = min(lons), max(lons)
lat_min, lat_max = min(lats), max(lats)
lon_scale = np.cos(np.radians((lat_min + lat_max) / 2))

PAD = 6 * SCALE
scale = (TARGET_WIDTH - 2 * PAD) / ((lon_max - lon_min) * lon_scale)
width = TARGET_WIDTH
height = int((lat_max - lat_min) * scale) + 2 * PAD

def lonlat_to_pixel(lon, lat):
    col = (lon - lon_min) * lon_scale * scale + PAD
    row = (lat_max - lat) * scale + PAD
    return col, row

# THE COUNTIES. Rasterize every historic county polygon directly onto the
# pixel grid, so every land pixel already knows which real county it's in
# — no separate calibration step needed, since the grid is built from this
# same real geometry in the first place.
county_names = [feat["properties"]["name"] for feat in uk_counties]
county_index = np.full((height, width), -1, dtype=np.int16)

cols_mesh, rows_mesh = np.meshgrid(np.arange(width), np.arange(height))
grid_pts = np.column_stack([cols_mesh.ravel(), rows_mesh.ravel()])

for i, feat in enumerate(uk_counties):
    mask = np.zeros(height * width, dtype=bool)
    for poly in polygons_of(feat):
        pixel_ring = [lonlat_to_pixel(lon, lat) for lon, lat in poly[0]]
        mask |= Path(pixel_ring).contains_points(grid_pts)
    county_index[mask.reshape(height, width)] = i

# Ireland has no historic-county backend data here (out of scope for a UK
# dialect quiz), but its coastline still needs to be drawn.
ireland_mask = np.zeros(height * width, dtype=bool)
for feat in ireland_counties:
    for poly in polygons_of(feat):
        pixel_ring = [lonlat_to_pixel(lon, lat) for lon, lat in poly[0]]
        ireland_mask |= Path(pixel_ring).contains_points(grid_pts)
ireland_mask = ireland_mask.reshape(height, width)

is_land = (county_index >= 0) | ireland_mask

# close stray one-pixel sea holes inside the landmass: flood-fill sea from
# the outer border, then anything left unreached is an enclosed hole, not
# real coastline, so treat it as land
reached = np.zeros((height, width), dtype=bool)
q = deque()
for r in range(height):
    for c in (0, width - 1):
        if not is_land[r, c] and not reached[r, c]:
            reached[r, c] = True
            q.append((r, c))
for c in range(width):
    for r in (0, height - 1):
        if not is_land[r, c] and not reached[r, c]:
            reached[r, c] = True
            q.append((r, c))
while q:
    r, c = q.popleft()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not is_land[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))
enclosed_hole = (~is_land) & (~reached)
total_land = is_land | enclosed_hole

# a handful of remaining UK land pixels (tiny gaps between adjacent county
# polygons, or filled-in holes) get the nearest already-labeled county,
# spreading outward the same way the hole-fill above does
fillable = total_land & (~ireland_mask) & (county_index == -1)
seed_q = deque()
has_county = county_index >= 0
for r in range(height):
    for c in range(width):
        if has_county[r, c]:
            seed_q.append((r, c))
while seed_q:
    r, c = seed_q.popleft()
    v = county_index[r, c]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and fillable[nr, nc] and county_index[nr, nc] == -1:
                county_index[nr, nc] = v
                seed_q.append((nr, nc))

picture = np.full((height, width, 3), SEA, dtype=np.uint8)
picture[total_land] = LAND

# CITIES. A handful of well-known landmarks, projected the same way.
cities = [
    ("London", -0.1278, 51.5074), ("Manchester", -2.2426, 53.4808),
    ("Birmingham", -1.8904, 52.4862), ("Leeds", -1.5491, 53.8008),
    ("Liverpool", -2.9916, 53.4084), ("Bristol", -2.5879, 51.4545),
    ("Edinburgh", -3.1883, 55.9533), ("Glasgow", -4.2518, 55.8642),
    ("Cardiff", -3.1791, 51.4816), ("Belfast", -5.9301, 54.5973),
    ("Newcastle", -1.6178, 54.9783), ("Sheffield", -1.4701, 53.3811),
    ("Nottingham", -1.1581, 52.9548), ("Oxford", -1.2577, 51.7520),
    ("Cambridge", 0.1218, 52.2053), ("York", -1.0827, 53.9600),
    ("Norwich", 1.2974, 52.6309), ("Exeter", -3.5339, 50.7184),
    ("Aberdeen", -2.0943, 57.1497), ("Inverness", -4.2247, 57.4778),
]

# crop tight around the landmass (plus a small margin) so it's centered
# in the frame instead of floating in a lot of empty sea
land_rows, land_cols = np.nonzero(total_land)
margin = 3 * SCALE
r0, r1 = max(land_rows.min() - margin, 0), min(land_rows.max() + margin + 1, height)
c0, c1 = max(land_cols.min() - margin, 0), min(land_cols.max() + margin + 1, width)
picture = picture[r0:r1, c0:c1]
county_index = county_index[r0:r1, c0:c1]
total_land = total_land[r0:r1, c0:c1]
height, width = picture.shape[:2]

cropped_cities = []
for name, lon, lat in cities:
    col, row = lonlat_to_pixel(lon, lat)
    col, row = col - c0, row - r0
    if 0 <= col < width and 0 <= row < height:
        cropped_cities.append((name, col, row))
cities = cropped_cities

plt.figure(figsize=(7, 11))
plt.imshow(picture)
plt.axis("off")
plt.show()

cropped_pixels = ["".join("X" if total_land[r, c] else "." for c in range(width)) for r in range(height)]
county_grid = county_index.tolist()  # -1 = no county data (sea, or Ireland)

CELL = 1  # pixels per grid cell; kept at SCALE original-CELL so the page footprint is unchanged

html = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin: 0; height: 100%%; overflow: hidden; }
  body { display: flex; justify-content: center; align-items: center; background: #f7f5f0; }
  canvas { max-width: 95vw; max-height: 95vh; width: auto; height: auto; display: block; }
</style>
</head>
<body>
<canvas id="c" width="%d" height="%d"></canvas>
<script>
const pixels = %s;
const CELL = %d;
const LAND = "#a6a6a6";
const ctx = document.getElementById("c").getContext("2d");
for (let row = 0; row < pixels.length; row++)
  for (let col = 0; col < pixels[row].length; col++) {
    if (pixels[row][col] === ".") continue;
    ctx.fillStyle = LAND;
    ctx.fillRect(col * CELL, row * CELL, CELL, CELL);
  }

// Backend region data — not shown on screen, but available to any script
// on this page (e.g. your quiz logic) via window.regionData:
//   regionData.countyNames[i]        -> name of county i
//   regionData.countyGrid[row][col]  -> county index at that pixel, or -1
//   regionData.cities                -> [[name, col, row], ...]
//   regionData.getRegionAt(row, col) -> county name at that pixel, or null
//   regionData.getCityNear(col, row) -> nearest city within a few pixels, or null
const countyNames = %s;
const countyGrid = %s;
const cities = %s;
window.regionData = {
  countyNames,
  countyGrid,
  cities,
  getRegionAt(row, col) {
    if (row < 0 || row >= countyGrid.length || col < 0 || col >= countyGrid[0].length) return null;
    const idx = countyGrid[row][col];
    return idx >= 0 ? countyNames[idx] : null;
  },
  getCityNear(col, row, maxDist) {
    maxDist = maxDist || 3;
    for (const [name, ccol, crow] of cities)
      if (Math.hypot(col - ccol, row - crow) <= maxDist) return name;
    return null;
  },
};
</script></body></html>""" % (
    width * CELL, height * CELL, str(cropped_pixels), CELL,
    json.dumps(county_names), json.dumps(county_grid), json.dumps(cities),
)

with open("map.html", "w") as f:
    f.write(html)

print(f"Saved map.html next to this script. Grid is {width}x{height} ({SCALE}x the original density).")
