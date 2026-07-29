# Rebuild britain.html as a STREAMLINED INTERACTIVE map from the existing
# britain_pixel_data.json — no baked-in labels/callouts, no colorbar clutter.
# Just the mosaic foot-strut map + a dot for each major city that you can
# hover to see its name and estimated % who rhyme foot/strut. Runs in
# seconds (reuses already-computed data; no re-rasterization).

import base64
import io
import json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image

SEA = (247, 245, 240)
LAND = (150, 150, 150)  # land with no foot-strut data (isolated islands) —
                        # same neutral grey as the baseline, so islands read
                        # as "no merger" rather than a separate colour

# Diverging colour scale: saturated BLUE (split) -> light cool-grey midpoint
# -> saturated RED (rhyme). The midpoint is a light grey (not pure white) so
# mid-value tiles stay visible against the warm cream page background instead
# of disappearing into it.
COLORMAP = LinearSegmentedColormap.from_list("blue_white_red", [
    (0.0, (0.06, 0.30, 0.82)),      # saturated blue  ~(15,77,209)
    (0.5, (0.87, 0.87, 0.90)),      # light cool grey (222,222,230)
    (1.0, (0.82, 0.06, 0.11)),      # saturated red   ~(209,15,28)
])

with open("britain_pixel_data.json") as f:
    d = json.load(f)

W, H = d["width"], d["height"]
county_grid = d["county_grid"]
county_names = d["county_names"]
fs = d["footstrut_gi_star"]
vmin, vmax = d["footstrut_vmin"], d["footstrut_vmax"]
a = d["footstrut_pct_formula"]["a"]
b = d["footstrut_pct_formula"]["b"]



def is_val(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


# --- colour the REAL decoded heat surface from the paper's photo ----------
# Every pixel's value is the actual Gi* colour-decoded from foot_strut_map.png
# (the figure you shared) and sampled per pixel — the genuine paper heat map,
# not a reconstruction. Low Gi* -> grey, high Gi* -> red, matching the photo's
# own gradient (just recoloured to grey->red).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

cg_arr = np.array(county_grid)
land_all = cg_arr >= 0
val = np.array([[fs[r][c] if is_val(fs[r][c]) else np.nan
                 for c in range(W)] for r in range(H)])

# Any land pixel the decode missed (a couple of coastal gaps, e.g. east of
# York) takes its NEAREST decoded value, so there are no stray grey holes
# inside the coloured heat.
nanmask = np.isnan(val)
ind = ndimage.distance_transform_edt(nanmask, return_distances=False, return_indices=True)
val_filled = val[tuple(ind)]

# full Gi* range -> diverging blue(split) / white / red(rhyme)
t = np.clip((val_filled - vmin) / (vmax - vmin), 0, 1)
rgb = (np.array(COLORMAP(t))[..., :3] * 255).astype(np.uint8)
picture = np.where(land_all[..., None], rgb, np.array(SEA, dtype=np.uint8)).astype(np.uint8)

# --- isogloss line (demo): the Gi* = 0 contour = merger/split boundary -----
# Smooth the land field a little, then trace the level-0 contour, like the
# black boundary line in the original figure. Purely a demo overlay.
land_val = np.where(land_all, val_filled, np.nan)
m = land_all.astype(float)
sm = ndimage.gaussian_filter(np.where(land_all, val_filled, 0.0), 1.6)
den = ndimage.gaussian_filter(m, 1.6)
sm = np.where(den > 1e-6, sm / den, np.nan)
cs = plt.contour(np.ma.masked_invalid(np.where(land_all, sm, np.nan)), levels=[0.0])
iso_segs = [seg for seg in cs.allsegs[0] if len(seg) >= 12]  # drop tiny specks
plt.close()
# each seg is (x=col, y=row) in grid units -> SVG polyline points
iso_polylines = [" ".join(f"{x:.2f},{y:.2f}" for x, y in seg) for seg in iso_segs]

# --- mosaic-tile render with soft seams (same look as the current map) -----
TILE, GAP = 10, 2
inner = TILE - GAP
off = GAP // 2
cell_mask = np.zeros((TILE, TILE), dtype=float)
cell_mask[off:off + inner, off:off + inner] = 1.0
mask = gaussian_filter(np.tile(cell_mask, (H, W)), sigma=1.0)
color_up = np.repeat(np.repeat(picture, TILE, axis=0), TILE, axis=1).astype(float)
bg = np.array(SEA, dtype=float)
render = np.clip(mask[..., None] * color_up + (1 - mask[..., None]) * bg, 0, 255).astype(np.uint8)

buf = io.BytesIO()
Image.fromarray(render).save(buf, format="PNG")
png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
Image.fromarray(render).save("britain.png")  # also refresh the standalone PNG

# --- major cities: dot position (% of image) + estimated % who rhyme -------
MAJOR = {"London", "Birmingham", "Manchester", "Leeds", "Liverpool", "Sheffield",
         "Bristol", "Newcastle", "Nottingham", "Cardiff", "Edinburgh", "Glasgow",
         "Aberdeen", "York"}


# % who rhyme, read from the ACTUAL decoded figure at each city (not a flat
# regional average): sample the decoded Gi* in a small window around the city
# and convert with the paper's Gi*->% fit. This reflects the real per-city
# variation the image shows (e.g. York/Leeds brightest, Newcastle less so).
def sample_pct(col, row, k=2):
    r0, r1 = max(0, int(row) - k), min(H, int(row) + k + 1)
    c0, c1 = max(0, int(col) - k), min(W, int(col) + k + 1)
    vals = [fs[r][c] for r in range(r0, r1) for c in range(c0, c1) if is_val(fs[r][c])]
    if not vals:
        return None
    return round(float(np.clip(a * float(np.mean(vals)) + b, 0, 100)))


dots = []
for name, col, row in d["cities"]:
    if name not in MAJOR:
        continue
    dots.append({
        "name": name,
        "left": round((col + 0.5) / W * 100, 3),   # cell-center as % of width
        "top": round((row + 0.5) / H * 100, 3),
        "pct": sample_pct(col, row),
    })

# legend gradient (bottom = split/grey, top = rhyme/red)
stops = []
for i in range(11):
    t = i / 10
    R, G, B = [int(round(x * 255)) for x in COLORMAP(t)[:3]]
    stops.append(f"rgb({R},{G},{B}) {int(t*100)}%")
legend_gradient = ", ".join(stops)

# isogloss overlay: white casing under a black line, like the figure's boundary.
# Toggle off for the clean published map; flip to True to bring the demo line back.
DRAW_ISOGLOSS = False
if DRAW_ISOGLOSS:
    iso_white = "".join(
        f'<polyline points="{pl}" fill="none" stroke="#f7f5f0" stroke-width="2.6" '
        'stroke-linejoin="round" stroke-linecap="round"/>' for pl in iso_polylines)
    iso_black = "".join(
        f'<polyline points="{pl}" fill="none" stroke="#141414" stroke-width="1.2" '
        'stroke-linejoin="round" stroke-linecap="round"/>' for pl in iso_polylines)
    iso_svg = (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" '
               'style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;">'
               f'{iso_white}{iso_black}</svg>')
else:
    iso_svg = ""
iso_legend = ('<div class="lab" style="margin-top:6px;"><span style="display:inline-block;'
              'width:22px;border-top:2px solid #141414;vertical-align:middle;"></span>'
              '<br><small>isogloss (demo)</small></div>') if DRAW_ISOGLOSS else ""

html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; height: 100%%; }
  body { display: flex; justify-content: center; align-items: center; gap: 24px;
         background: #f7f5f0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
  #wrap { position: relative; display: inline-block; max-height: 94vh; }
  #wrap img { display: block; max-width: 92vw; max-height: 94vh; width: auto; height: auto; }
  .dot { position: absolute; width: 9px; height: 9px; margin: -4.5px 0 0 -4.5px;
         border-radius: 50%%; background: #fff; border: 2px solid #2b2b2b;
         box-shadow: 0 0 0 1px rgba(255,255,255,.6); pointer-events: none;
         transition: transform .12s ease; }
  .dot.active { transform: scale(1.7); z-index: 5; }
  .tip { position: absolute; bottom: 150%%; left: 50%%; transform: translateX(-50%%);
         background: #2b2b2b; color: #fff; padding: 5px 9px; border-radius: 6px;
         font-size: 12px; line-height: 1.3; white-space: nowrap; text-align: center;
         opacity: 0; pointer-events: none; transition: opacity .12s ease; }
  .tip::after { content: ""; position: absolute; top: 100%%; left: 50%%;
         transform: translateX(-50%%); border: 5px solid transparent;
         border-top-color: #2b2b2b; }
  .tip b { font-weight: 600; }
  .tip small { opacity: .8; font-weight: 400; }
  .dot.active .tip { opacity: 1; }
  #wrap { cursor: crosshair; }
  #legend { display: flex; flex-direction: column; align-items: center; gap: 8px;
            font-size: 12px; color: #3d3d3a; }
  #legend .bar { width: 16px; height: 220px; border-radius: 3px;
                 background: linear-gradient(to top, %s); border: 1px solid #ccc; }
  #legend .lab { text-align: center; max-width: 90px; }
</style>
</head>
<body>
<div id="wrap">
  <img src="data:image/png;base64,%s" alt="Foot-strut dialect map of Great Britain">
  %s
</div>
<div id="legend">
  <div class="lab"><b>rhyme</b><br><small>foot = strut</small></div>
  <div class="bar"></div>
  <div class="lab"><b>split</b><br><small>foot &ne; strut</small></div>
  %s
</div>
<script>
const cities = %s;
const wrap = document.getElementById("wrap");
const dotEls = [];
for (const ct of cities) {
  const dot = document.createElement("div");
  dot.className = "dot";
  dot.style.left = ct.left + "%%";
  dot.style.top = ct.top + "%%";
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.innerHTML = "<b>" + ct.name + "</b>" +
    (ct.pct != null ? "<br><small>~" + ct.pct + "%% rhyme foot/strut</small>" : "");
  dot.appendChild(tip);
  wrap.appendChild(dot);
  dotEls.push({el: dot, city: ct});
}

// Nearest-city hover: instead of each dot owning its own hover (where
// overlapping dots in the dense Manchester/Leeds/Sheffield cluster fight
// each other), a single handler lights up whichever city is closest to
// the cursor — as long as it's within a sensible radius. Clean even when
// dots overlap.
const HOVER_RADIUS = 34;  // px in displayed-image space
let activeEl = null;
function setActive(el) {
  if (activeEl === el) return;
  if (activeEl) activeEl.classList.remove("active");
  activeEl = el;
  if (activeEl) activeEl.classList.add("active");
}
wrap.addEventListener("mousemove", (e) => {
  const rect = wrap.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bestD = Infinity;
  for (const d of dotEls) {
    const dx = d.city.left / 100 * rect.width - mx;
    const dy = d.city.top / 100 * rect.height - my;
    const dist = Math.hypot(dx, dy);
    if (dist < bestD) { bestD = dist; best = d.el; }
  }
  setActive(bestD <= HOVER_RADIUS ? best : null);
});
wrap.addEventListener("mouseleave", () => setActive(null));

// backend data still available to any script on the page
const countyNames = %s, countyGrid = %s, footstrutGiStar = %s;
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
    if (v == null) return null;
    return Math.max(0, Math.min(100, PCT_A * v + PCT_B));
  },
};
</script>
</body>
</html>""" % (
    legend_gradient, png_b64, iso_svg, iso_legend, json.dumps(dots),
    json.dumps(county_names), json.dumps(county_grid), json.dumps(fs),
    a, b,
)

with open("britain.html", "w") as f:
    f.write(html)

print(f"Rebuilt britain.html — {len(dots)} major-city hover dots, no baked labels.")
print("cities:", ", ".join(c["name"] for c in dots))
