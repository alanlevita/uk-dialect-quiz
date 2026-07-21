# British Isles / Britain pixel maps — generated from real geographic
# boundaries (Historic County Borders Project for the UK, GADM/Tailte
# Éireann-derived data for Ireland), not hand-drawn. Paste this whole thing
# into one cell and run it. Takes a few minutes: it's rasterizing ~120 real
# polygons at high resolution.
#
# Needs historic_counties.geojson, ireland_counties.geojson, and
# foot_strut_map.png sitting next to this script.
#
# Produces two outputs:
#   map.html / map.png         -> British Isles (Britain + Ireland)
#   britain.html / britain.png -> Great Britain only (no Northern Ireland),
#     shaded by the real foot-strut ("do foot and cut rhyme for you?") Gi*
#     data, sampled per pixel (not averaged per county) by registering
#     foot_strut_map.png — Figure/Map 2 of MacKenzie, Bailey & Turton (2022),
#     "Towards an updated dialect atlas of British English", Journal of
#     Linguistic Geography — against real coordinates. Gi* is a spatial
#     hotspot statistic, not a raw percentage; a handful of counties get a
#     real percentage callout too, quoting the paper's own stated numbers.
#     Not our data — credit that paper if you publish this.
# Orkney is excluded from both, per request.

import base64
import json
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
from PIL import Image
from scipy.spatial import cKDTree

SEA  = (247, 245, 240)   # background
LAND = (166, 166, 166)   # grey for all land

NI_COUNTIES = {"Antrim", "Armagh", "Down", "Fermanagh", "Londonderry", "Tyrone"}

# --- foot-strut reference decode -------------------------------------------
# Calibration derived once by inspecting foot_strut_map.png directly: its
# Gi* colorbar sits at y=350, spanning x=1006..1289, with tick marks at
# x=1057(-4), x=1143(0), x=1229(4) giving a linear color->value scale. The
# image's own pixel grid was registered to real coordinates using four
# identifiable coastline extremes (Shetland/Cornwall/Suffolk/Pembrokeshire).
_REF_BAR_Y = 350
_REF_BAR_X0, _REF_BAR_X1 = 1006, 1290
_REF_TICK0_X, _REF_TICK0_VAL = 1143, 0.0
_REF_TICK_SPACING_PX, _REF_TICK_SPACING_VAL = 86, 4.0
_REF_CALIBRATION = [
    ((557, 41),   (-0.8736, 60.8608)),   # Shetland (north)
    ((375, 1357), (-6.4000, 49.8647)),   # Cornwall, Lizard Point (south)
    ((945, 1002), (1.7631, 52.4811)),    # Suffolk coast (east)
    ((371, 1087), (-5.4822, 51.7303)),   # Pembrokeshire (west)
]

# Real percentages the paper states in prose for specific places (the
# survey question: "do foot and cut rhyme for you?" — this is the %
# who said yes, i.e. no phonemic split).
FOOTSTRUT_CALLOUTS = [
    # label, lon, lat, percent who rhyme
    ("Derbyshire", -1.4746, 52.9225, 79),
    ("Nottinghamshire", -1.1581, 52.9548, 76),
    ("Leicester", -1.1398, 52.6369, 43),
    ("Northamptonshire", -0.8969, 52.2405, 7),
    ("Warwickshire", -1.5849, 52.2823, 24),
    ("Worcestershire", -2.2200, 52.1920, 31),
    ("Berwick-upon-Tweed", -2.0000, 55.7708, 8),
]

def _load_footstrut_reference():
    im = np.array(Image.open("foot_strut_map.png").convert("RGB")).astype(float)
    rh, rw, _ = im.shape

    xs = np.arange(_REF_BAR_X0, _REF_BAR_X1)
    bar_colors = im[_REF_BAR_Y, xs]
    bar_values = (xs - _REF_TICK0_X) / _REF_TICK_SPACING_PX * _REF_TICK_SPACING_VAL

    tree = cKDTree(bar_colors)
    dist, idx = tree.query(im.reshape(-1, 3), k=1)
    values = bar_values[idx]
    values[dist > 12] = np.nan  # not a viridis color (lines, text, boxes, background)
    value_grid = values.reshape(rh, rw)

    A = np.array([[lo, la, 1] for _, (lo, la) in _REF_CALIBRATION])
    col_b = np.array([c for (c, r), _ in _REF_CALIBRATION])
    row_b = np.array([r for (c, r), _ in _REF_CALIBRATION])
    col_coef, *_ = np.linalg.lstsq(A, col_b, rcond=None)
    row_coef, *_ = np.linalg.lstsq(A, row_b, rcond=None)
    return value_grid, col_coef, row_coef

FOOTSTRUT_GRID, FOOTSTRUT_COL_COEF, FOOTSTRUT_ROW_COEF = _load_footstrut_reference()
FOOTSTRUT_VMIN = float(np.nanmin(FOOTSTRUT_GRID))
FOOTSTRUT_VMAX = float(np.nanmax(FOOTSTRUT_GRID))
_viridis = cm.get_cmap("viridis")

def _footstrut_project(lon, lat):
    c = FOOTSTRUT_COL_COEF[0] * lon + FOOTSTRUT_COL_COEF[1] * lat + FOOTSTRUT_COL_COEF[2]
    r = FOOTSTRUT_ROW_COEF[0] * lon + FOOTSTRUT_ROW_COEF[1] * lat + FOOTSTRUT_ROW_COEF[2]
    return c, r

# Rough Gi*->percentage conversion. Gi* is a smoothed spatial statistic,
# not a direct transform of the raw survey percentage, so this is only an
# approximation: fit by least squares from 10 real (Gi*, %) anchor pairs
# (the 7 callout points above, plus Scotland/Edinburgh 3%, South/London
# 5%, North/Newcastle 79%, sampled the same way). Typical error is
# +-15-20 percentage points, worse at documented local exceptions (e.g.
# Berwick-upon-Tweed reads only mildly warm in Gi* despite a very low
# real percentage) — use it as a rough readout, not a precise figure.
_GI_TO_PCT_A, _GI_TO_PCT_B = 9.8972, 31.4619

def gi_to_percent(v):
    return float(np.clip(_GI_TO_PCT_A * v + _GI_TO_PCT_B, 0, 100))

# ----------------------------------------------------------------------------

SCALE = 3  # pixel density relative to the original 155x236 hand-drawn grid
BASE_WIDTH, BASE_HEIGHT = 155, 236
TARGET_WIDTH = BASE_WIDTH * SCALE

EXCLUDED_COUNTIES = {"Orkney"}

with open("historic_counties.geojson") as f:
    uk_counties = [feat for feat in json.load(f)["features"]
                   if feat["properties"]["name"] not in EXCLUDED_COUNTIES]
with open("ireland_counties.geojson") as f:
    ireland_counties = json.load(f)["features"]

def polygons_of(feat):
    geom = feat["geometry"]
    return geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]

# Project real (lon, lat) onto the pixel grid: equirectangular, fit to the
# combined UK+Ireland extent, with a latitude correction (cos of the
# mid-latitude) so the country isn't stretched east-west. Shared by both
# outputs so they're on the same scale.
all_points = [pt for feat in uk_counties + ireland_counties
              for poly in polygons_of(feat) for ring in poly for pt in ring]
lons = [p[0] for p in all_points]
lats = [p[1] for p in all_points]
lon_min, lon_max = min(lons), max(lons)
lat_min, lat_max = min(lats), max(lats)
lon_scale = np.cos(np.radians((lat_min + lat_max) / 2))

PAD = 6 * SCALE
scale = (TARGET_WIDTH - 2 * PAD) / ((lon_max - lon_min) * lon_scale)
full_width = TARGET_WIDTH
full_height = int((lat_max - lat_min) * scale) + 2 * PAD

def lonlat_to_pixel(lon, lat):
    col = (lon - lon_min) * lon_scale * scale + PAD
    row = (lat_max - lat) * scale + PAD
    return col, row

def pixel_to_lonlat(col, row):
    lon = (col - PAD) / (lon_scale * scale) + lon_min
    lat = lat_max - (row - PAD) / scale
    return lon, lat

# THE COUNTIES. Rasterize every historic county polygon directly onto the
# full-size pixel grid once, so every land pixel already knows which real
# county it's in — no separate calibration step needed, since the grid is
# built from this same real geometry in the first place. Both outputs
# below reuse this same rasterization (Britain just skips the Ireland mask).
county_names = [feat["properties"]["name"] for feat in uk_counties]
county_index_full = np.full((full_height, full_width), -1, dtype=np.int16)

cols_mesh, rows_mesh = np.meshgrid(np.arange(full_width), np.arange(full_height))
grid_pts = np.column_stack([cols_mesh.ravel(), rows_mesh.ravel()])

for i, feat in enumerate(uk_counties):
    mask = np.zeros(full_height * full_width, dtype=bool)
    for poly in polygons_of(feat):
        pixel_ring = [lonlat_to_pixel(lon, lat) for lon, lat in poly[0]]
        mask |= Path(pixel_ring).contains_points(grid_pts)
    county_index_full[mask.reshape(full_height, full_width)] = i

ireland_mask_full = np.zeros(full_height * full_width, dtype=bool)
for feat in ireland_counties:
    for poly in polygons_of(feat):
        pixel_ring = [lonlat_to_pixel(lon, lat) for lon, lat in poly[0]]
        ireland_mask_full |= Path(pixel_ring).contains_points(grid_pts)
ireland_mask_full = ireland_mask_full.reshape(full_height, full_width)

CITIES = [
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

def build_map(include_ireland, out_prefix, exclude_extra=None, use_footstrut_heat=False, pixelate_block=None):
    county_index = county_index_full.copy()
    if exclude_extra:
        for name in exclude_extra:
            county_index[county_index == county_names.index(name)] = -1
    is_land = county_index >= 0
    if include_ireland:
        is_land = is_land | ireland_mask_full

    # close stray one-pixel sea holes inside the landmass: flood-fill sea
    # from the outer border, then anything left unreached is an enclosed
    # hole, not real coastline, so treat it as land
    h, w = is_land.shape
    reached = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not is_land[r, c] and not reached[r, c]:
                reached[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
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
                if 0 <= nr < h and 0 <= nc < w and not is_land[nr, nc] and not reached[nr, nc]:
                    reached[nr, nc] = True
                    q.append((nr, nc))
    enclosed_hole = (~is_land) & (~reached)
    total_land = is_land | enclosed_hole

    # remaining UK land pixels with no county yet (tiny gaps between
    # adjacent polygons, or filled-in holes) get the nearest already
    # -labeled county, spreading outward the same way the hole-fill does
    ireland_mask = ireland_mask_full if include_ireland else np.zeros((h, w), dtype=bool)
    fillable = total_land & (~ireland_mask) & (county_index == -1)
    seed_q = deque()
    has_county = county_index >= 0
    for r in range(h):
        for c in range(w):
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
                if 0 <= nr < h and 0 <= nc < w and fillable[nr, nc] and county_index[nr, nc] == -1:
                    county_index[nr, nc] = v
                    seed_q.append((nr, nc))

    picture = np.full((h, w, 3), SEA, dtype=np.uint8)
    picture[total_land] = LAND

    # crop tight around the landmass (plus a small margin) so it's
    # centered in the frame instead of floating in a lot of empty sea
    land_rows, land_cols = np.nonzero(total_land)
    margin = 3 * SCALE
    r0, r1 = max(land_rows.min() - margin, 0), min(land_rows.max() + margin + 1, h)
    c0, c1 = max(land_cols.min() - margin, 0), min(land_cols.max() + margin + 1, w)
    picture = picture[r0:r1, c0:c1]
    county_index = county_index[r0:r1, c0:c1]
    total_land = total_land[r0:r1, c0:c1]
    height, width = picture.shape[:2]

    cropped_cities = []
    for name, lon, lat in CITIES:
        col, row = lonlat_to_pixel(lon, lat)
        col, row = col - c0, row - r0
        if 0 <= col < width and 0 <= row < height:
            cropped_cities.append((name, col, row))

    # HEAT. Real foot-strut Gi* data, sampled per pixel (not averaged per
    # county): every one of our own land pixels gets its own real-world
    # (lon, lat) looked up, projected into the reference figure's pixel
    # space, and bilinear-ish sampled from its decoded value grid — a
    # genuinely granular result instead of flat per-county blocks. Any
    # pixel the source figure has no color data for (thin border lines,
    # callout boxes, out-of-frame) gets the nearest sampled neighbor's
    # value via the same flood-fill technique as the coastline holes.
    callout_points = []
    if use_footstrut_heat:
        rows_idx, cols_idx = np.indices((height, width))
        full_col = cols_idx + c0
        full_row = rows_idx + r0
        lon = (full_col - PAD) / (lon_scale * scale) + lon_min
        lat = lat_max - (full_row - PAD) / scale
        ref_col = FOOTSTRUT_COL_COEF[0] * lon + FOOTSTRUT_COL_COEF[1] * lat + FOOTSTRUT_COL_COEF[2]
        ref_row = FOOTSTRUT_ROW_COEF[0] * lon + FOOTSTRUT_ROW_COEF[1] * lat + FOOTSTRUT_ROW_COEF[2]
        ref_ci = np.clip(np.round(ref_col).astype(int), 0, FOOTSTRUT_GRID.shape[1] - 1)
        ref_ri = np.clip(np.round(ref_row).astype(int), 0, FOOTSTRUT_GRID.shape[0] - 1)
        sampled = FOOTSTRUT_GRID[ref_ri, ref_ci].copy()

        gb_land = county_index >= 0
        sampled[~gb_land] = np.nan

        # nearest-neighbor fill for any GB land pixel the source had no
        # color for, spreading from every already-valid pixel
        missing = gb_land & np.isnan(sampled)
        if missing.any():
            fq = deque()
            filled = ~np.isnan(sampled)
            for r in range(height):
                for c in range(width):
                    if filled[r, c]:
                        fq.append((r, c))
            while fq:
                r, c = fq.popleft()
                v = sampled[r, c]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and gb_land[nr, nc] and not filled[nr, nc]:
                            filled[nr, nc] = True
                            sampled[nr, nc] = v
                            fq.append((nr, nc))

        t = np.clip((sampled - FOOTSTRUT_VMIN) / (FOOTSTRUT_VMAX - FOOTSTRUT_VMIN), 0, 1)
        rgba = _viridis(np.nan_to_num(t))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        picture[gb_land] = rgb[gb_land]

        for label, lon_c, lat_c, pct in FOOTSTRUT_CALLOUTS:
            col, row = lonlat_to_pixel(lon_c, lat_c)
            col, row = col - c0, row - r0
            if 0 <= col < width and 0 <= row < height:
                callout_points.append((label, col, row, pct))

    # PIXELATE. Block-average everything down to a coarse grid so
    # individual square "pixels" are visible again — same real coastline
    # and real data underneath, just rendered chunky/hand-drawn instead
    # of smooth and precise. Every backend lookup (county, Gi*, cities)
    # gets downsampled the same way, so clicks/queries match what's shown.
    if pixelate_block and pixelate_block > 1:
        bs = pixelate_block
        h2, w2 = height // bs, width // bs

        def block_mode(arr, bg):
            out = np.full((h2, w2), bg, dtype=arr.dtype)
            for i in range(h2):
                for j in range(w2):
                    block = arr[i * bs:(i + 1) * bs, j * bs:(j + 1) * bs].ravel()
                    vals, counts = np.unique(block, return_counts=True)
                    out[i, j] = vals[np.argmax(counts)]
            return out

        def block_sample(arr):
            # take the center pixel of each block, not an average — flat,
            # unblended color per block, like real pixel art, instead of
            # every coastline/color edge smearing into a muddy blend
            cropped = arr[:h2 * bs, :w2 * bs]
            return cropped[bs // 2::bs, bs // 2::bs]

        county_index = block_mode(county_index, -1)
        total_land = block_mode(total_land.astype(np.int8), 0).astype(bool)
        picture = np.stack([block_sample(picture[..., k]) for k in range(3)], axis=-1)
        if use_footstrut_heat:
            sampled = block_sample(sampled)
            gb_land = county_index >= 0

        cropped_cities = [(name, col / bs, row / bs) for name, col, row in cropped_cities]
        callout_points = [(label, col / bs, row / bs, pct) for label, col, row, pct in callout_points]
        height, width = h2, w2

    # MOSAIC GRID. For rendering only (the raw data/JSON/CSV stay at the
    # plain low-res grid): blow each pixel up into a small tile with a
    # thin gap of page-background color around it, so tiles read as
    # separate hand-placed mosaic squares instead of one solid 8-bit
    # block image.
    if pixelate_block and pixelate_block > 1:
        TILE, GAP = 6, 1
        inner = TILE - GAP
        off = GAP // 2
        render_picture = np.full((height * TILE, width * TILE, 3), SEA, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                render_picture[i * TILE + off:i * TILE + off + inner,
                                j * TILE + off:j * TILE + off + inner] = picture[i, j]
        render_height, render_width = render_picture.shape[:2]
        render_callouts = [(label, col * TILE + TILE / 2, row * TILE + TILE / 2, pct)
                            for label, col, row, pct in callout_points]
    else:
        render_picture = picture
        render_height, render_width = height, width
        render_callouts = callout_points

    fig, ax = plt.subplots(figsize=(7, 11))
    ax.imshow(render_picture, interpolation="nearest")
    ax.axis("off")

    if use_footstrut_heat:
        sm = cm.ScalarMappable(cmap=_viridis,
                                norm=plt.Normalize(vmin=FOOTSTRUT_VMIN, vmax=FOOTSTRUT_VMAX))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.3, anchor=(0, 0.9))
        cbar.set_label("Gi* (foot–strut rhyme; real data)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        # spread callouts down the right margin with leader lines, similar
        # to the source figure's own style
        render_callouts.sort(key=lambda p: p[2])
        n = len(render_callouts)
        for i, (label, col, row, pct) in enumerate(render_callouts):
            label_x = render_width * 1.28
            label_y = render_height * (0.08 + 0.8 * i / max(n - 1, 1))
            ax.plot([col], [row], "o", color="#888", markersize=4, markeredgecolor="white")
            ax.annotate(f"{label} {pct}%", xy=(col, row), xytext=(label_x, label_y),
                        fontsize=9, va="center",
                        bbox=dict(boxstyle="round", fc="white", ec="black", lw=0.8),
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))
        ax.set_xlim(0, render_width * 1.85)

    plt.savefig(f"{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    char_grid = [["X" if total_land[r, c] else "." for c in range(width)] for r in range(height)]
    cropped_pixels = ["".join(row) for row in char_grid]
    county_grid = county_index.tolist()  # -1 = no county data (sea, or Ireland)

    CELL = 1  # pixels per grid cell; kept at SCALE original-CELL so the page footprint is unchanged

    if use_footstrut_heat:
        # THE ACTUAL DATA. The real per-pixel Gi* grid (not just the
        # rendered colors), saved as a standalone file for use outside
        # this page too, plus embedded in the page's own backend data.
        footstrut_export = [
            [round(float(sampled[r, c]), 2) if gb_land[r, c] else None for c in range(width)]
            for r in range(height)
        ]
        with open(f"{out_prefix}_pixel_data.json", "w") as f:
            json.dump({
                "width": width, "height": height,
                "county_names": county_names, "county_grid": county_grid,
                "footstrut_gi_star": footstrut_export,
                "footstrut_vmin": FOOTSTRUT_VMIN, "footstrut_vmax": FOOTSTRUT_VMAX,
                "footstrut_pct_formula": {
                    "a": _GI_TO_PCT_A, "b": _GI_TO_PCT_B,
                    "note": "pct ~= a*Gi* + b; rough approximation, typically +-15-20 points off",
                },
                "cities": cropped_cities,
                "source": "Gi* from MacKenzie, Bailey & Turton (2022), Journal of Linguistic "
                          "Geography 10(1):46-66 — color-decoded from their Figure/Map 2 and "
                          "registered to real coordinates. Not our data; credit that paper.",
            }, f)

        # same raw grid as a flat CSV (row, col, real lon/lat, county,
        # Gi*, rough estimated %) for anyone who'd rather open it in a
        # spreadsheet than parse the nested JSON
        bs_geo = pixelate_block if (pixelate_block and pixelate_block > 1) else 1
        half_geo = bs_geo // 2
        with open(f"{out_prefix}_pixel_data.csv", "w") as f:
            f.write("row,col,lon,lat,county,gi_star,estimated_pct\n")
            for r in range(height):
                for c in range(width):
                    full_row = r0 + r * bs_geo + half_geo
                    full_col = c0 + c * bs_geo + half_geo
                    lon, lat = pixel_to_lonlat(full_col, full_row)
                    if gb_land[r, c]:
                        county = county_names[county_index[r, c]]
                        gi = round(float(sampled[r, c]), 2)
                        pct = round(gi_to_percent(gi), 1)
                        f.write(f"{r},{c},{lon:.4f},{lat:.4f},{county},{gi},{pct}\n")
                    else:
                        f.write(f"{r},{c},{lon:.4f},{lat:.4f},,,\n")

        # continuous per-pixel color data doesn't suit the per-pixel JS
        # fillRect loop used for the flat-color isles map — embed the
        # already-rendered PNG directly instead.
        with open(f"{out_prefix}.png", "rb") as f:
            png_b64 = base64.b64encode(f.read()).decode("ascii")
        visual_html = f'<img src="data:image/png;base64,{png_b64}" alt="{out_prefix} map">'

        html = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin: 0; height: 100%%; overflow: hidden; }
  body { display: flex; justify-content: center; align-items: center; background: #f7f5f0; }
  img { max-width: 95vw; max-height: 95vh; width: auto; height: auto; display: block;
        image-rendering: pixelated; image-rendering: crisp-edges; }
</style>
</head>
<body>
%s
<script>
// Backend region + real foot-strut data — not shown on screen, available
// to any script on this page (e.g. your quiz logic) via window.regionData:
//   regionData.countyNames[i]            -> name of county i
//   regionData.countyGrid[row][col]      -> county index at that pixel, or -1
//   regionData.footstrutGiStar[row][col] -> real Gi* value at that pixel, or null
//   regionData.getRegionAt(row, col)          -> county name, or null
//   regionData.getFootstrutGiStarAt(row, col) -> real Gi* value, or null
//   regionData.estimatePercentAt(row, col)    -> rough estimated "%% who rhyme"
//     (derived from Gi* by linear fit to the paper's own stated numbers —
//     typically +-15-20 points off; see map.py for the caveat)
const countyNames = %s;
const countyGrid = %s;
const footstrutGiStar = %s;
const cities = %s;
const PCT_A = %r, PCT_B = %r;
window.regionData = {
  countyNames, countyGrid, footstrutGiStar, cities,
  getRegionAt(row, col) {
    if (row < 0 || row >= countyGrid.length || col < 0 || col >= countyGrid[0].length) return null;
    const idx = countyGrid[row][col];
    return idx >= 0 ? countyNames[idx] : null;
  },
  getFootstrutGiStarAt(row, col) {
    if (row < 0 || row >= footstrutGiStar.length || col < 0 || col >= footstrutGiStar[0].length) return null;
    return footstrutGiStar[row][col];
  },
  estimatePercentAt(row, col) {
    const v = this.getFootstrutGiStarAt(row, col);
    if (v === null) return null;
    return Math.max(0, Math.min(100, PCT_A * v + PCT_B));
  },
  getCityNear(col, row, maxDist) {
    maxDist = maxDist || 3;
    for (const [name, ccol, crow] of cities)
      if (Math.hypot(col - ccol, row - crow) <= maxDist) return name;
    return null;
  },
};
</script></body></html>""" % (
            visual_html, json.dumps(county_names), json.dumps(county_grid),
            json.dumps(footstrut_export), json.dumps(cropped_cities),
            _GI_TO_PCT_A, _GI_TO_PCT_B,
        )

    else:
        # THE OFFICIAL MAP. Just the landmass and clickable cities —
        # click a city to see its name; click anywhere else on land to
        # see which county it's in.
        visual_html = f'<canvas id="c" width="{width * CELL}" height="{height * CELL}"></canvas>'

        html = """<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin: 0; height: 100%%; overflow: hidden; }
  body { display: flex; flex-direction: column; align-items: center; justify-content: center;
         gap: 8px; background: #f7f5f0; font-family: -apple-system, sans-serif; }
  canvas { max-width: 95vw; max-height: 88vh; width: auto; height: auto; display: block; cursor: pointer; }
  #label { font-size: 15px; color: #3d3d3a; min-height: 1.2em; }
</style>
</head>
<body>
<div id="label">Click a city, or anywhere on the map</div>
%s
<script>
const pixels = %s;
const CELL = %d;
const LAND = "#a6a6a6";
const CITY_DOT = "#3d3d3a";
const cities = %s;
const countyNames = %s;
const countyGrid = %s;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const label = document.getElementById("label");

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < pixels.length; row++)
    for (let col = 0; col < pixels[row].length; col++) {
      if (pixels[row][col] === ".") continue;
      ctx.fillStyle = LAND;
      ctx.fillRect(col * CELL, row * CELL, CELL, CELL);
    }
  ctx.fillStyle = CITY_DOT;
  for (const [, ccol, crow] of cities) {
    ctx.beginPath();
    ctx.arc(ccol * CELL, crow * CELL, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}
draw();

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
    maxDist = maxDist || 4;
    for (const [name, ccol, crow] of cities)
      if (Math.hypot(col - ccol, row - crow) <= maxDist) return name;
    return null;
  },
};

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const s = canvas.width / rect.width;
  const x = (e.clientX - rect.left) * s;
  const y = (e.clientY - rect.top) * s;
  const col = Math.floor(x / CELL), row = Math.floor(y / CELL);
  const city = window.regionData.getCityNear(col, row);
  if (city) { label.textContent = city; return; }
  const county = window.regionData.getRegionAt(row, col);
  label.textContent = county || "Click a city, or anywhere on the map";
});
</script></body></html>""" % (
            visual_html, str(cropped_pixels), CELL,
            json.dumps(cropped_cities), json.dumps(county_names), json.dumps(county_grid),
        )

    with open(f"{out_prefix}.html", "w") as f:
        f.write(html)

    print(f"Saved {out_prefix}.html — {width}x{height} ({SCALE}x original density).")

build_map(include_ireland=True, out_prefix="map")
build_map(include_ireland=False, out_prefix="britain",
          exclude_extra=NI_COUNTIES, use_footstrut_heat=True, pixelate_block=4)
