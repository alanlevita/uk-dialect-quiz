# Build quiz.html — SIDE-BY-SIDE stepping quiz (work in progress).
# Left column  = the current (nth) question.
# Right column = the PREVIOUS (n-1th) question's single-answer heat map:
#   red = where that answer is common. Binary yes/no -> the "no" map is the
#   flip (1 - yes). Multiple-choice -> each option has its OWN map.
# Cities are shaded by the map, hover shows an APPROXIMATE % (strong regions
# read "90%+"/"under 10%", not fake-precise). Phonological questions let you
# click a city for the expected local IPA. No combined final result yet.
#
# Data:
#  - foot-strut  : real, decoded from foot_strut_map.png
#  - bread roll  : eight single-select options, each surface read closely off
#    the eight panels of bread_isogloss.png (MacKenzie, Bailey & Turton 2022)
#    and cross-checked against the paper's prose (Nottingham 65% cob;
#    Birmingham 41% roll / 20% bap / 17% cob / 14% bun; barm -> Manchester;
#    muffin -> East Manchester). Approximate regionalisations, not a pixel
#    decode of every panel.

import json
import base64
import numpy as np
from scipy import ndimage

# ice-lolly photo, embedded so the page stays self-contained
popsicle_uri = "data:image/avif;base64," + base64.b64encode(open("popsicle-image.avif", "rb").read()).decode()

with open("britain_pixel_data.json") as f:
    d = json.load(f)
W, H = d["width"], d["height"]
fs = d["footstrut_gi_star"]
cg = np.array(d["county_grid"])
names = d["county_names"]
a = d["footstrut_pct_formula"]["a"]
b = d["footstrut_pct_formula"]["b"]
land = cg >= 0


def is_val(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


# ---- foot-strut: real decoded surface -> P(rhyme) ----
gi = np.array([[fs[r][c] if is_val(fs[r][c]) else np.nan for c in range(W)] for r in range(H)])
ind = ndimage.distance_transform_edt(np.isnan(gi), return_distances=False, return_indices=True)
gi = gi[tuple(ind)]
q1 = np.clip(a * gi + b, 0, 100) / 100.0

# ---- bread roll: per-term surfaces read off bread_isogloss.png ----
SCOT = ["Aberdeenshire", "Angus", "Argyllshire", "Ayrshire", "Banffshire", "Berwickshire",
        "Buteshire", "Caithness", "Clackmannanshire", "Cromartyshire", "Dumfriesshire",
        "Dunbartonshire", "East Lothian", "Fife", "Inverness-shire", "Kincardineshire",
        "Kinross-shire", "Kirkcudbrightshire", "Lanarkshire", "Midlothian", "Morayshire",
        "Nairnshire", "Orkney", "Peeblesshire", "Perthshire", "Renfrewshire", "Ross-shire",
        "Roxburghshire", "Selkirkshire", "Stirlingshire", "Sutherland", "West Lothian", "Wigtownshire"]
WALES = ["Anglesey", "Brecknockshire", "Caernarfonshire", "Cardiganshire", "Carmarthenshire",
         "Denbighshire", "Flintshire", "Glamorgan", "Merionethshire", "Monmouthshire",
         "Montgomeryshire", "Pembrokeshire", "Radnorshire"]
NORTH = ["Cheshire", "Cumberland", "Durham", "Lancashire", "Northumberland", "Westmorland", "Yorkshire"]
EMIDS = ["Derbyshire", "Leicestershire", "Lincolnshire", "Northamptonshire", "Nottinghamshire", "Rutland"]
EANG = ["Cambridgeshire", "Huntingdonshire", "Norfolk", "Suffolk"]
SOUTH = ["Bedfordshire", "Berkshire", "Buckinghamshire", "Cornwall", "Devon", "Dorset", "Essex",
         "Gloucestershire", "Hampshire", "Hertfordshire", "Kent", "Middlesex", "Oxfordshire",
         "Somerset", "Surrey", "Sussex", "Wiltshire"]
SE_ENG = ["Bedfordshire", "Berkshire", "Buckinghamshire", "Essex", "Hampshire",
          "Hertfordshire", "Kent", "Middlesex", "Oxfordshire", "Surrey", "Sussex"]
SW_ENG = ["Cornwall", "Devon", "Dorset", "Gloucestershire", "Somerset", "Wiltshire"]
NWALES = ["Anglesey", "Caernarfonshire", "Denbighshire", "Flintshire", "Merionethshire", "Montgomeryshire"]
SWALES = ["Brecknockshire", "Cardiganshire", "Carmarthenshire", "Glamorgan",
          "Monmouthshire", "Pembrokeshire", "Radnorshire"]
NSCOT = ["Aberdeenshire", "Angus", "Argyllshire", "Banffshire", "Caithness", "Cromartyshire",
         "Inverness-shire", "Kincardineshire", "Morayshire", "Nairnshire", "Perthshire",
         "Ross-shire", "Sutherland", "Orkney"]
CSCOT = ["Ayrshire", "Buteshire", "Clackmannanshire", "Dunbartonshire", "East Lothian", "Fife",
         "Kinross-shire", "Lanarkshire", "Midlothian", "Renfrewshire", "Stirlingshire", "West Lothian"]
SSCOT = ["Berwickshire", "Dumfriesshire", "Kirkcudbrightshire", "Peeblesshire",
         "Roxburghshire", "Selkirkshire", "Wigtownshire"]
WMIDS = ["Herefordshire", "Shropshire", "Staffordshire", "Warwickshire", "Worcestershire"]


def mk(base, groups=None, counties=None):
    m = {n: base for n in names}
    for grp, val in (groups or []):
        for n in grp:
            m[n] = val
    if counties:
        m.update(counties)
    return m


BREAD = {
    # peak Manchester/SE Lancashire, greening SW through Cheshire/NE-Wales/Marches; low in Yorkshire
    "barm": mk(6, None, {"Lancashire": 82, "Cheshire": 55, "Flintshire": 35, "Denbighshire": 30,
                         "Merionethshire": 16, "Shropshire": 26, "Staffordshire": 28,
                         "Worcestershire": 20, "Gloucestershire": 12, "Derbyshire": 18,
                         "Cumberland": 15, "Westmorland": 15, "Yorkshire": 12}),
    # peak West Yorkshire + East Lancashire (Pennines), north tongue to Cumbria
    "teacake": mk(5, None, {"Yorkshire": 70, "Lancashire": 48, "Westmorland": 30, "Cumberland": 28,
                            "Cheshire": 18, "Derbyshire": 20, "Durham": 20, "Nottinghamshire": 12}),
    # very tight on Manchester
    "muffin": mk(3, None, {"Lancashire": 58, "Cheshire": 22, "Yorkshire": 12, "Derbyshire": 10,
                           "Warwickshire": 12, "Staffordshire": 10}),
    # East Midlands core (Notts/Derby/Leics), greening W into the Marches; low Lincolnshire
    "cob": mk(6, [(WMIDS, 32)], {"Nottinghamshire": 80, "Leicestershire": 68, "Derbyshire": 62,
                                 "Rutland": 55, "Northamptonshire": 42, "Lincolnshire": 35,
                                 "Staffordshire": 40, "Warwickshire": 22, "Worcestershire": 36,
                                 "Herefordshire": 40, "Shropshire": 38, "Gloucestershire": 22,
                                 "Huntingdonshire": 22}),  # Warwickshire lowered: Birmingham is ~17% cob
    # tight on Coventry/Warwickshire, faint NW + Welsh-border green; ~0 elsewhere
    "batch": mk(5, None, {"Warwickshire": 38, "Worcestershire": 28, "Herefordshire": 24,
                          "Shropshire": 22, "Staffordshire": 18, "Lancashire": 22, "Westmorland": 22,
                          "Cumberland": 18, "Gloucestershire": 18, "Leicestershire": 14}),
    # PEAK NORTH WALES; moderate S. Wales / S. England / N. Scotland; low N-England,
    # E-Mids and central Scotland (this was the wrong one before)
    "bap": mk(12, [(NSCOT, 30), (SW_ENG, 28), (SE_ENG, 22), (EANG, 22), (SWALES, 28),
                   (CSCOT, 9), (SSCOT, 11)],
              {"Anglesey": 55, "Caernarfonshire": 55, "Merionethshire": 52, "Denbighshire": 45,
               "Flintshire": 42, "Montgomeryshire": 40, "Shropshire": 30, "Herefordshire": 30,
               "Cheshire": 18, "Gloucestershire": 24,
               "Northumberland": 6, "Durham": 6, "Yorkshire": 8, "Lancashire": 9,
               "Cumberland": 8, "Westmorland": 8,
               "Nottinghamshire": 8, "Leicestershire": 10, "Derbyshire": 8, "Lincolnshire": 8,
               "Northamptonshire": 12, "Rutland": 10,
               "Warwickshire": 22, "Staffordshire": 10, "Worcestershire": 16}),  # Birmingham ~20% bap
    # peak Yorkshire + Durham (not the far NE)
    "bun": mk(8, None, {"Yorkshire": 75, "Durham": 68, "Lincolnshire": 40, "Northumberland": 40,
                        "Cumberland": 35, "Westmorland": 32, "Nottinghamshire": 25, "Derbyshire": 22,
                        "Warwickshire": 15, "Hampshire": 18, "Berkshire": 16, "Wiltshire": 16}),
    # Scotland + South + East Anglia high; low in the NW/N-Wales; but Birmingham
    # itself is a roll city (~41%), so Warwickshire/Worcestershire keep roll up
    "roll": mk(30, [(NSCOT, 58), (CSCOT, 68), (SSCOT, 66), (SE_ENG, 64), (SW_ENG, 58), (EANG, 58)],
               {"Lancashire": 8, "Cheshire": 8, "Staffordshire": 24, "Shropshire": 8, "Derbyshire": 10,
                "Nottinghamshire": 12, "Warwickshire": 44, "Worcestershire": 34, "Herefordshire": 14,
                "Flintshire": 10, "Denbighshire": 10, "Merionethshire": 12, "Caernarfonshire": 14,
                "Anglesey": 16, "Montgomeryshire": 12,
                "Yorkshire": 22, "Durham": 26, "Northumberland": 32, "Cumberland": 30, "Westmorland": 26,
                "Leicestershire": 18, "Lincolnshire": 26, "Northamptonshire": 32, "Rutland": 22,
                "Glamorgan": 35, "Monmouthshire": 40, "Carmarthenshire": 30, "Pembrokeshire": 30,
                "Cardiganshire": 28, "Brecknockshire": 28, "Radnorshire": 28,
                "Middlesex": 66, "Gloucestershire": 56}),
}

# tea vs dinner (name for the evening meal): P(tea). Binary, and the paper gives
# real proportions, so this one carries a %. North = tea (NW/NE ~67%, Yorks 69%),
# extending into the N Midlands; South East = dinner (London ~5% tea, SE ~16%);
# SW (Cornwall/Devon/Somerset ~45-47%) and Suffolk (43%) are moderate-tea pockets;
# Scotland mixed (Glasgow/central belt low).
TEA = mk(40, [(NSCOT, 46), (CSCOT, 30), (SSCOT, 42), (NWALES, 52), (SWALES, 42),
              (SE_ENG, 16), (SW_ENG, 42), (EANG, 32)],
         {"Lancashire": 67, "Cheshire": 65, "Yorkshire": 69, "Durham": 67, "Northumberland": 65,
          "Cumberland": 62, "Westmorland": 62,
          "Derbyshire": 68, "Staffordshire": 64, "Nottinghamshire": 60, "Lincolnshire": 52,
          "Leicestershire": 50, "Rutland": 45, "Shropshire": 55, "Worcestershire": 46,
          "Warwickshire": 46, "Herefordshire": 44,
          "Cornwall": 45, "Devon": 47, "Somerset": 47, "Dorset": 34, "Wiltshire": 30, "Gloucestershire": 42,
          "Middlesex": 5, "Surrey": 12, "Sussex": 13, "Kent": 14, "Hampshire": 16, "Berkshire": 16,
          "Hertfordshire": 15, "Essex": 20, "Buckinghamshire": 16, "Bedfordshire": 20, "Oxfordshire": 20,
          "Suffolk": 43, "Norfolk": 35, "Cambridgeshire": 28, "Huntingdonshire": 30,
          "Northamptonshire": 30,
          "Fife": 42, "Midlothian": 42, "East Lothian": 42})

# book vs spook: do the -ook words keep the long GOOSE vowel /uː/ (so book = [buːk],
# rhyming with spook)? Yes-strongholds from the map + paper: all of Scotland (no
# foot/goose distinction), the North East (Northumberland/Tyne&Wear ~83-85%), Stoke
# (77%), fading pockets in Merseyside (~25%), North Wales & Bridgend. Low elsewhere.
BOOKSPOOK = mk(7, [(SCOT, 72), (NWALES, 34)],
    {"Northumberland": 83, "Durham": 78, "Cumberland": 32, "Westmorland": 25,
     "Staffordshire": 55, "Shropshire": 22, "Cheshire": 12, "Lancashire": 20,
     "Yorkshire": 10, "Derbyshire": 12,
     "Anglesey": 46, "Caernarfonshire": 46, "Merionethshire": 35, "Denbighshire": 28,
     "Flintshire": 26, "Glamorgan": 30, "Monmouthshire": 15})

# TRAP-BATH split: proportion who use the LONG /ɑː/ in bath/grass/last (= the split).
# South = long a (high, red); North, Wales & Scotland = short a (low, blue).
# BLENDED from the two panels in trap-bath.webp (BBC Future + English Dialect App):
# they agree on the sharp Severn-Wash N/S line — EDA more extreme, BBC Future more
# moderate — so these values sit between the two.
TRAPBATH = mk(88, [(SCOT, 12), (NWALES, 15), (SWALES, 20), (SE_ENG, 90), (SW_ENG, 87), (EANG, 82)],
    {"Northumberland": 8, "Durham": 8, "Cumberland": 10, "Westmorland": 10, "Lancashire": 10,
     "Cheshire": 16, "Yorkshire": 12,
     "Derbyshire": 25, "Nottinghamshire": 32, "Lincolnshire": 48, "Staffordshire": 30,
     "Leicestershire": 48, "Rutland": 58, "Shropshire": 25,
     "Warwickshire": 45, "Worcestershire": 48, "Herefordshire": 48,
     "Northamptonshire": 62, "Huntingdonshire": 68, "Cambridgeshire": 72, "Bedfordshire": 78,
     "Gloucestershire": 80, "Oxfordshire": 82, "Buckinghamshire": 82, "Hertfordshire": 85,
     "Essex": 85, "Middlesex": 92})

# words for a splinter of wood in the skin (each .ppm map = one variant vs splinter):
# spelk = North East + Borders/Cumbria (Old Norse); spell = northern (weak);
# shiver = East Anglian (weak); sliver = scattered; splinter = the standard, common
# everywhere except spelk's NE heartland. Blended from BBC Future + English Dialect App.
SPELK = mk(4, [(["Berwickshire", "Roxburghshire", "Selkirkshire", "Peeblesshire"], 40)],
    {"Northumberland": 75, "Durham": 70, "Cumberland": 42, "Westmorland": 38, "Yorkshire": 14,
     "Dumfriesshire": 26, "Lancashire": 10})
SPELL = mk(4, None,
    {"Lancashire": 32, "Cheshire": 26, "Yorkshire": 24, "Cumberland": 22, "Westmorland": 22,
     "Derbyshire": 18, "Flintshire": 18, "Durham": 16})
# shiver: SOLELY East Anglia, peaked hard on Norfolk (Norwich) so it's the one
# city that reads "shiver"; near-zero everywhere else.
SHIVER = mk(1, None, {"Norfolk": 52, "Suffolk": 24, "Cambridgeshire": 9})
# sliver: the SOUTH EAST / Home Counties word — Essex core, spreading through
# Hertfordshire/Middlesex/Cambridgeshire and down into Surrey/Kent/Sussex
# (per splinter-variant-map.webp, the orange region around & NE of London).
SLIVER = mk(2, None, {"Essex": 54, "Hertfordshire": 42, "Middlesex": 36, "Cambridgeshire": 30,
                      "Surrey": 30, "Kent": 24, "Sussex": 20, "Bedfordshire": 24, "Buckinghamshire": 16})
# "Give it me" (alternative double-object dative): % who accept it. Strongest in the
# North West + N Midlands (Manchester/Cheshire/Staffs/Derbys/S.Yorks), high in the
# West & East Midlands, thinning fast to the NE (York 56, Teesside 41, Newcastle 25)
# and low in the South, Scotland & Wales. From MacKenzie, Bailey & Turton 2022 (Map 12).
GIVEITME = mk(14, [(SCOT, 12), (SWALES, 18), (SE_ENG, 18), (EANG, 20)],
    {"Lancashire": 82, "Cheshire": 82, "Staffordshire": 78, "Derbyshire": 78,
     "Nottinghamshire": 74, "Leicestershire": 72, "Rutland": 58, "Lincolnshire": 48,
     "Northamptonshire": 46, "Warwickshire": 66, "Worcestershire": 66, "Shropshire": 70,
     "Herefordshire": 56, "Gloucestershire": 48, "Yorkshire": 64,
     "Cumberland": 48, "Westmorland": 54, "Durham": 38, "Northumberland": 24,
     "Flintshire": 60, "Denbighshire": 56, "Caernarfonshire": 48, "Anglesey": 42,
     "Merionethshire": 46, "Montgomeryshire": 48, "Radnorshire": 30,
     "Oxfordshire": 32, "Buckinghamshire": 19, "Hertfordshire": 19, "Bedfordshire": 24,
     "Somerset": 44, "Devon": 42, "Cornwall": 30, "Dorset": 26, "Wiltshire": 30,
     "Huntingdonshire": 24, "Cambridgeshire": 22, "Argyllshire": 24, "Dunbartonshire": 22})
# ice lolly (the standard nationwide term) vs lolly ice (the famous Merseyside /
# Liverpool 'Scouse' reversal). No published map to hand -> approximated from the
# well-known Scouse distribution. NB: Liverpool & Manchester are both historic
# Lancashire, so the grid can't fully isolate Merseyside from Greater Manchester.
# ice lolly is the national word (near-universal); lolly ice is a tight Liverpool /
# Merseyside pocket (see LOLLYICE point-blob below). Together they blanket the UK.
ICELOLLY = mk(85, None, {})
SPLINTER = mk(66, None,
    {"Northumberland": 28, "Durham": 32, "Cumberland": 46, "Westmorland": 50,
     "Norfolk": 52, "Suffolk": 52, "Lancashire": 58, "Yorkshire": 60})


def surface(valmap, sigma=2.0):
    v = np.full((H, W), np.nan)
    for r in range(H):
        for c in range(W):
            if land[r, c]:
                v[r, c] = valmap.get(names[cg[r, c]], 0)
    numg = ndimage.gaussian_filter(np.where(land, v, 0.0), sigma)
    deng = ndimage.gaussian_filter(land.astype(float), sigma)
    return np.where(deng > 1e-6, numg / deng, 0.0) / 100.0


def grid_json(p):
    return [[round(float(p[r][c]), 4) if land[r][c] else None for c in range(W)]
            for r in range(H)]


# a localized hotspot centred on grid coordinates (not a whole county) — used for
# lolly ice, which is specific to Liverpool/Merseyside, not all of Lancashire.
def point_blob(points, sigma):
    ys, xs = np.mgrid[0:H, 0:W]
    v = np.zeros((H, W))
    for col, row, peak in points:
        v = np.maximum(v, peak * np.exp(-(((xs - col) ** 2 + (ys - row) ** 2) / (2.0 * sigma ** 2))))
    return np.where(land, v / 100.0, 0.0)


grids_all = {"q1": grid_json(q1), "tvd": grid_json(surface(TEA)),
             "giveitme": grid_json(surface(GIVEITME)),
             "bookspook": grid_json(surface(BOOKSPOOK)),
             # store as P(rhyme = short a) so "yes, they rhyme" -> North (matches the option)
             "trapbath": grid_json(1 - surface(TRAPBATH)),
             "spelk": grid_json(surface(SPELK)), "spell": grid_json(surface(SPELL)),
             "shiver": grid_json(surface(SHIVER)), "sliver": grid_json(surface(SLIVER)),
             "splinter": grid_json(surface(SPLINTER)),
             "icelolly": grid_json(surface(ICELOLLY)),
             # lolly ice: Liverpool/Merseyside + a North Wales coast cluster (Flintshire/
             # Denbighshire), which shares the Merseyside form; NOT Manchester.
             "lollyice": grid_json(point_blob([(50.7, 91.3, 88), (47.8, 95.0, 76)], 3.0))}
for term, vm in BREAD.items():
    grids_all[term] = grid_json(surface(vm))

# "I don't have a word for this" = the NEGATIVE of all the variants combined.
# Combine the listed terms per-pixel (max = "where ANY of these words is common"),
# then invert: 1 - that. So red = the places least covered by any of the words,
# blue = the places where at least one of the words is strongly used.
def negative_union(terms):
    out = []
    for r in range(H):
        row = []
        for c in range(W):
            if not land[r][c]:
                row.append(None)
                continue
            m = 0.0
            for t in terms:
                val = grids_all[t][r][c]
                if val is not None and val > m:
                    m = val
            row.append(round(1.0 - m, 4))
        out.append(row)
    return out


grids_all["none_splinter"] = negative_union(["splinter", "spelk", "spell", "shiver", "sliver"])
grids_all["none_bread"] = negative_union(list(BREAD.keys()))

landj = [[bool(land[r][c]) for c in range(W)] for r in range(H)]

CITY_LL = {"London", "Manchester", "Birmingham", "Leeds", "Liverpool", "Sheffield",
           "Bristol", "Newcastle", "Nottingham", "Cardiff", "Edinburgh", "Glasgow",
           "Aberdeen", "York", "Norwich", "Cambridge", "Exeter"}
cities = [{"name": n, "col": co, "row": ro} for n, co, ro in d["cities"] if n in CITY_LL]

# ---- regions, for the "your answer matches X" readout ----
REGIONS = {
    "Scotland": SCOT,
    "the North East": ["Northumberland", "Durham"],
    "the North West": ["Lancashire", "Cheshire", "Cumberland", "Westmorland"],
    "Yorkshire": ["Yorkshire"],
    "the East Midlands": EMIDS,
    "the West Midlands": ["Warwickshire", "Worcestershire", "Herefordshire", "Shropshire", "Staffordshire"],
    "East Anglia": EANG,
    "the South East": ["Bedfordshire", "Berkshire", "Buckinghamshire", "Essex", "Hampshire",
                       "Hertfordshire", "Kent", "Middlesex", "Oxfordshire", "Surrey", "Sussex"],
    "the South West": ["Cornwall", "Devon", "Dorset", "Gloucestershire", "Somerset", "Wiltshire"],
    "Wales": WALES,
}
region_names = list(REGIONS.keys())
county_region = {}
for ri, (rn, cl) in enumerate(REGIONS.items()):
    for cn in cl:
        county_region[cn] = ri
region_grid = [[(county_region.get(names[cg[r][c]], -1) if land[r][c] else -1)
                for c in range(W)] for r in range(H)]

# ---- major dialect GROUPS (for the landing-page teaser), grouped & coloured to
# match dialect-map.jpg: Scotland (blue), the North (purple), the Midlands
# (orange), East Anglia (yellow), the South (red), the West Country (green),
# Wales (teal-green). Great Britain only (the grid has no Ireland).
# muted, atlas-style colours (less bright than before)
DIALECT = [
    ("Scotland", SCOT, (78, 110, 168)),
    ("The North", NORTH, (146, 100, 156)),
    ("The Midlands", EMIDS + WMIDS, (216, 146, 80)),
    ("East Anglia", EANG + ["Essex"], (212, 192, 110)),
    ("The South", ["Bedfordshire", "Berkshire", "Buckinghamshire", "Hampshire", "Hertfordshire",
                   "Kent", "Middlesex", "Oxfordshire", "Surrey", "Sussex"], (190, 96, 100)),
    ("The West Country", ["Cornwall", "Devon", "Dorset", "Gloucestershire", "Somerset", "Wiltshire"],
     (92, 152, 104)),
    ("Wales", WALES, (60, 128, 116)),
]
county_dialect = {}
for di, (dn, cl, _col) in enumerate(DIALECT):
    for cn in cl:
        county_dialect[cn] = di
dialect_grid = [[(county_dialect.get(names[cg[r][c]], -1) if land[r][c] else -1)
                 for c in range(W)] for r in range(H)]
dialect_colors = [[dn, list(col)] for dn, _cl, col in DIALECT]

# ---- HIGH-RES landing map: rasterize the real historic-county polygons so the
# teaser is detailed like the survey maps (still a pixel raster, just much finer) ----
from matplotlib.path import Path as _MplPath
_gj = json.load(open("historic_counties.geojson"))
_polys = []
for _ft in _gj["features"]:
    _di = county_dialect.get(_ft["properties"].get("name"), -1)
    if _di < 0:
        continue
    _geom = _ft["geometry"]
    _coords = _geom["coordinates"] if _geom["type"] == "MultiPolygon" else [_geom["coordinates"]]
    for _poly in _coords:
        _polys.append((_di, _poly[0]))
_pts = [p for _, ring in _polys for p in ring]
_lon0, _lon1 = min(p[0] for p in _pts), max(p[0] for p in _pts)
_lat0, _lat1 = min(p[1] for p in _pts), max(p[1] for p in _pts)
_kx = float(np.cos(np.radians((_lat0 + _lat1) / 2)))
_Hh = 480
_Wh = int(round((_lon1 - _lon0) * _kx / (_lat1 - _lat0) * _Hh))
_xs, _ys = np.meshgrid(np.arange(_Wh), np.arange(_Hh))
_gp = np.column_stack([_xs.ravel() + 0.5, _ys.ravel() + 0.5])
_hg = np.full(_Hh * _Wh, -1, dtype=int)
for _di, _ring in _polys:
    _proj = [((lo - _lon0) * _kx / ((_lon1 - _lon0) * _kx) * (_Wh - 1),
              (_lat1 - la) / (_lat1 - _lat0) * (_Hh - 1)) for lo, la in _ring]
    _hg[_MplPath(_proj).contains_points(_gp)] = _di
_hg = _hg.reshape(_Hh, _Wh)
hires_dialect = ["".join("." if v < 0 else str(v) for v in row) for row in _hg]

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=1100"><style>
  :root{--bg:#f7f5f0;--ink:#2b2b2b;--muted:#8a857c;--accent:#c0141f;--card:#fff;--line:#e6e1d8;}
  *{box-sizing:border-box;}
  html,body{margin:0;min-height:100%%;}
  body{background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;}
  #app{max-width:1060px;margin:0 auto;padding:20px 20px 16px;position:relative;}
  #qimg{display:block;width:230px;height:230px;object-fit:cover;border-radius:16px;margin:2px auto 16px;box-shadow:0 6px 18px rgba(0,0,0,.16);}
  .sliderbox{margin:20px 0 8px;}
  .sliderbox input[type=range]{width:100%%;accent-color:var(--accent);height:6px;cursor:pointer;}
  .slabels{display:flex;justify-content:space-between;align-items:flex-start;font-size:12px;color:var(--muted);margin-top:10px;line-height:1.35;}
  .slabels .sval{font-size:26px;font-weight:750;color:var(--accent);align-self:center;}
  #restart{position:absolute;top:8px;right:16px;display:inline-flex;align-items:center;gap:7px;
       font-size:13.5px;font-weight:650;color:#fff;background:var(--accent);
       border:none;border-radius:10px;padding:10px 18px;cursor:pointer;z-index:20;box-shadow:0 3px 10px rgba(192,20,31,.22);}
  #restart:hover{background:#a5101a;}
  #restart .ricon{font-size:30px;line-height:1;display:flex;align-items:center;}
  header{text-align:center;margin-bottom:6px;}
  .site-title{font-size:27px;font-weight:750;letter-spacing:-.015em;margin:0 0 3px;}
  .site-sub{font-size:13px;color:var(--muted);margin:0 0 18px;}
  .progress-wrap{max-width:420px;margin:0 auto 7px;height:6px;background:#e9e4da;border-radius:99px;overflow:hidden;}
  .progress-bar{height:100%%;width:0;background:var(--accent);border-radius:99px;transition:width .3s ease;}
  #progress{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);text-align:center;margin-bottom:14px;}
  #stage{display:flex;gap:30px;align-items:flex-start;justify-content:center;flex-wrap:wrap;}
  #left{flex:0 0 340px;max-width:340px;}
  #qtext{font-size:24px;line-height:1.3;margin:0 0 7px;font-weight:650;}
  #qtag{font-size:12px;color:var(--muted);margin-bottom:20px;}
  #qtag.real{color:#0a7a63;} #qtag::before{content:"\\25CF  ";font-size:9px;vertical-align:middle;}
  .opt{display:flex;align-items:center;gap:11px;width:100%%;margin:6px 0;padding:11px 15px;font-size:15px;text-align:left;
       background:var(--card);border:1.5px solid var(--line);border-radius:12px;cursor:pointer;transition:all .12s;color:var(--ink);}
  .opt:not(:disabled):hover{border-color:var(--accent);transform:translateY(-1px);box-shadow:0 4px 14px rgba(0,0,0,.07);}
  .opt.sel{border-color:var(--accent);background:#fdf0f0;}
  .opt:disabled{cursor:default;opacity:.45;} .opt.sel:disabled{opacity:1;}
  .opt .box{flex:0 0 18px;height:18px;border:1.5px solid #c9c3b8;border-radius:5px;display:inline-flex;
            align-items:center;justify-content:center;font-size:12px;line-height:1;color:#fff;}
  .opt.sel .box{background:var(--accent);border-color:var(--accent);}
  .hint{font-size:12px;color:var(--muted);margin:0 0 8px;min-height:16px;}
  #next{margin-top:12px;width:100%%;padding:13px;font-size:15px;font-weight:600;color:#fff;background:var(--accent);
        border:none;border-radius:12px;cursor:pointer;transition:background .12s;}
  #next:hover:not(:disabled){background:#a5101a;} #next:disabled{background:#dcbcbe;cursor:not-allowed;}
  #leftdone{font-size:16px;color:#555;line-height:1.55;} #leftdone b{color:var(--ink);}
  .navbtns{margin-top:14px;display:flex;gap:12px;align-items:center;}
  #back{font-size:14px;font-weight:600;color:var(--ink);background:var(--card);border:1.5px solid var(--line);
        border-radius:10px;padding:11px 18px;cursor:pointer;transition:all .12s;}
  #back:hover{border-color:#bdb7ab;background:#f2ede4;}
  #startover{font-size:14px;color:#fff;background:var(--ink);border:none;border-radius:9px;padding:9px 16px;cursor:pointer;}
  #startover:hover{background:#000;}
  #right{flex:0 0 auto;text-align:center;max-width:360px;}
  #rtitle{font-size:14px;line-height:1.4;margin-bottom:6px;min-height:2em;color:var(--muted);}
  #rtitle b{color:var(--ink);}
  #out{position:relative;display:inline-block;}
  canvas{display:block;cursor:pointer;max-height:52vh;max-width:86vw;width:auto;height:auto;border-radius:6px;}
  #rprompt{color:#bbb;font-size:14px;padding:60px 44px;border:2px dashed #e2ddd3;border-radius:14px;}
  #match{font-size:16px;margin-top:8px;min-height:1.3em;font-weight:600;} #match b{color:var(--accent);}
  #legend{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);justify-content:center;margin-top:6px;}
  #legend .bar{width:150px;height:12px;border-radius:3px;
       background:linear-gradient(to right,rgb(15,77,209),rgb(222,222,230),rgb(209,15,28));border:1px solid #ccc;}
  #infowrap{position:relative;display:inline-block;margin-top:6px;}
  #infobtn{font-size:13px;color:#0a7a63;cursor:help;} #infobtn:hover{text-decoration:underline;}
  #info{display:none;position:absolute;bottom:150%%;left:50%%;transform:translateX(-50%%);width:340px;max-width:82vw;
        font-size:12.5px;line-height:1.5;color:#333;text-align:left;background:var(--card);border:1px solid var(--line);
        border-radius:10px;padding:11px 13px;box-shadow:0 8px 24px rgba(0,0,0,.16);z-index:40;}
  #info::before{content:"";position:absolute;top:100%%;left:50%%;transform:translateX(-50%%);
        border:7px solid transparent;border-top-color:var(--card);}
  #infowrap:hover #info{display:block;}
  #info .src{color:#aaa;font-size:11px;margin-top:8px;} #info .isep{border:none;border-top:1px solid #eee;margin:9px 0;}
  #detail{font-size:13px;color:#444;margin-top:6px;min-height:1.2em;} #detail .ipa{font-size:18px;color:#111;margin:0 6px;letter-spacing:.5px;}
  .tip{position:absolute;background:var(--ink);color:#fff;padding:6px 10px;border-radius:6px;font-size:12px;
       line-height:1.35;white-space:nowrap;pointer-events:none;opacity:0;transform:translate(-50%%,-118%%);
       transition:opacity .1s;} .tip b{font-weight:600;} .tip small{opacity:.82;}
  /* landing page: two panels, vertically centred and filling the viewport */
  #intro{display:flex;flex-direction:column;align-items:center;justify-content:center;max-width:1180px;
       margin:0 auto;min-height:calc(100vh - 40px);}
  .intro-head{text-align:center;margin-bottom:30px;width:100%%;}
  .intro-panels{display:flex;gap:8vw;align-items:center;justify-content:center;flex-wrap:wrap;width:100%%;}
  .intro-left{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;text-align:center;}
  .intro-right{flex:0 1 400px;max-width:400px;display:flex;flex-direction:column;justify-content:center;text-align:left;}
  .intro-title{font-size:clamp(36px,5.4vw,62px);font-weight:800;letter-spacing:-.025em;margin:0 0 8px;line-height:1.02;}
  .intro-sub{font-size:14px;color:var(--muted);margin:0;}
  #introcvwrap{position:relative;display:inline-block;}
  #introcv{display:block;margin:0 auto 10px;image-rendering:auto;cursor:crosshair;max-width:100%%;height:auto;}
  #introtip{position:absolute;pointer-events:none;opacity:0;transform:translate(-50%%,-135%%);white-space:nowrap;
       font-size:12px;font-weight:650;color:#fff;padding:4px 9px;border-radius:6px;transition:opacity .08s;z-index:5;}
  .intro-cap{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0;min-height:1.2em;}
  .intro-lead{font-size:25px;line-height:1.4;font-weight:700;letter-spacing:-.01em;margin:0 0 16px;}
  .intro-body{font-size:15.5px;line-height:1.65;color:#555;margin:0 0 28px;}
  #startbtn{font-size:16px;font-weight:650;color:#fff;background:var(--accent);border:none;border-radius:12px;
       padding:15px 36px;cursor:pointer;transition:all .14s;box-shadow:0 6px 18px rgba(192,20,31,.24);}
  #startbtn:hover{background:#a5101a;transform:translateY(-2px);box-shadow:0 9px 22px rgba(192,20,31,.30);}
  .intro-note{font-size:12px;color:var(--muted);margin:18px 0 0;letter-spacing:.03em;text-align:center;}
  .aboutwrap{position:relative;display:inline-block;vertical-align:middle;}
  .aboutbtn{color:#0a7a63;cursor:help;font-size:15px;line-height:1;}
  .aboutinfo{display:none;position:absolute;bottom:150%%;left:50%%;transform:translateX(-50%%);width:430px;max-width:88vw;
       font-size:12.5px;line-height:1.6;color:#333;text-align:left;background:var(--card);border:1px solid var(--line);
       border-radius:10px;padding:14px 16px;box-shadow:0 10px 28px rgba(0,0,0,.18);z-index:60;letter-spacing:normal;}
  .aboutinfo::after{content:"";position:absolute;top:100%%;left:50%%;transform:translateX(-50%%);
       border:7px solid transparent;border-top-color:var(--card);}
  .aboutwrap:hover .aboutinfo{display:block;}
  /* ---- narrow screens / phones: stop anything running off the edge ---- */
  @media (max-width:640px){
    #app{padding:16px 14px 14px;}
    #left{flex:1 1 auto;max-width:100%%;width:100%%;}
    #right{max-width:100%%;}
    .intro-right{max-width:100%%;}
    #restart{top:10px;right:10px;padding:8px 12px;font-size:12px;}
    #restart .ricon{font-size:22px;}
    /* the (i) pop-ups centre in the viewport so their text never gets cut off */
    #info,.aboutinfo{position:fixed;left:50%%;right:auto;top:auto;bottom:16px;
         transform:translateX(-50%%);width:92vw;max-width:92vw;}
    #info::before,.aboutinfo::after{display:none;}
  }
</style></head><body>
<div id="app">
  <button id="restart" style="display:none"><span class="ricon">&#10227;</span> Restart quiz</button>
  <header>
    <div class="site-title">The British Dialect Quiz</div>
    <div class="site-sub">Answer a few questions &mdash; see where each answer places you on the map.</div>
    <div class="progress-wrap" id="progresswrap"><div class="progress-bar" id="pbar"></div></div>
    <div id="progress"></div>
  </header>
  <div id="intro">
    <div class="intro-head">
      <div class="intro-title">The Great British Dialect Quiz</div>
      <div class="intro-sub">Answer a few questions, see where each answer places you on the map.</div>
    </div>
    <div class="intro-panels">
    <div class="intro-left">
      <div id="introcvwrap"><canvas id="introcv"></canvas><div id="introtip"></div></div>
      <p class="intro-cap" id="introcap">Hover the map to explore the dialect groups</p>
    </div>
    <div class="intro-right">
      <p class="intro-lead">How you say a few everyday words, and what you call bread, your evening meal, or a splinter, quietly gives away where in Britain you&rsquo;re from.</p>
      <p class="intro-body">This short quiz asks how <i>you</i> speak. After each answer a heat map lights up, showing where in Great Britain that feature is common, all drawn from published dialect research. Work through them and see which corner of the map your speech belongs to.</p>
      <button id="startbtn">Start the quiz &rarr;</button>
      <p class="intro-note"><span class="aboutwrap"><span class="aboutbtn">&#9432;</span><span class="aboutinfo">This is a pixel-art version of the British dialect map. It was made by <b>Alan Levita</b> during an internship at the Intellectual Forum at Jesus College, University of Cambridge, drawing on the research of <b>Prof. Bert Vaux</b> of King&rsquo;s College, Cambridge. Bert&rsquo;s work formed the basis for the original <i>New York Times</i> dialect quiz.<br><br>Your answers are used to estimate roughly where you&rsquo;re from. All the maps were redrawn by hand in a pixel-art style, based on isoglosses from published research on British dialects.</span></span> Powered by the Intellectual Forum at Jesus College, University of Cambridge</p>
    </div>
    </div>
  </div>
  <div id="stage">
    <div id="left">
      <h1 id="qtext"></h1>
      <img id="qimg" style="display:none" alt="">
      <div id="qtag"></div>
      <div id="opts"></div>
      <div class="hint" id="hint" style="display:none"></div>
      <button id="next" style="display:none">See my map &rarr;</button>
      <div id="leftdone" style="display:none"></div>
      <div class="navbtns">
        <button id="back" style="display:none">&#8592; back</button>
        <button id="startover" style="display:none">&#8630; start over</button>
      </div>
    </div>
    <div id="right">
      <div id="rtitle">Answer the question &rarr; your map appears here</div>
      <div id="out"><div id="rprompt">your map will appear here</div><canvas id="cv" style="display:none"></canvas>
        <div class="tip" id="tip"></div></div>
      <div id="match"></div>
      <div id="legend" style="display:none"><span>uncommon</span><span class="bar"></span><span>common</span></div>
      <div id="infowrap" style="display:none"><span id="infobtn"></span><div id="info"></div></div>
      <div id="detail"></div>
    </div>
  </div>
</div>
<script>
const W=%d,H_=%d,CELL=5,GAP=1;
const grids=%s;
const land=%s,cities=%s,cg=%s,names=%s;
const regionGrid=%s,regionNames=%s;
const dialectGrid=%s,dialectColors=%s;
const HIRES=%s;
const POPSICLE_IMG=%s;
const QUESTIONS=[
  // metric "pct": a clean binary the paper reports as proportions -> show a percent.
  // ipa:true enables the click-a-city foot-strut IPA readout (foot-strut only).
  {id:"q1",text:"Do <i>foot</i> and <i>cut</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   ipa:true,info:"footstrut",infoLabel:"the foot&ndash;strut split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme"},{label:"No, they sound different",v:0,word:"split"}]},
  {id:"bookspook",text:"Do <i>book</i> and <i>spook</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"bookspook",infoLabel:"book vs spook",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme book/spook"},{label:"No, they sound different",v:0,word:"don&rsquo;t rhyme"}]},
  {id:"trapbath",text:"Do <i>gas</i> and <i>grass</i> rhyme for you?",tag:"blended: BBC Future + English Dialect App",real:true,metric:"pct",
   info:"trapbath",infoLabel:"the trap&ndash;bath split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (short a)"},
         {label:"No, they sound different",v:0,word:"split (long a)"}]},
  // binary + the paper gives real proportions -> metric "pct"
  {id:"tvd",text:"What do you call your evening meal?",tag:"real data",real:true,metric:"pct",
   info:"tvd",infoLabel:"tea vs dinner",
   opts:[{label:"Tea",v:1,word:"say tea"},{label:"Dinner",v:0,word:"say dinner"}]},
  {id:"giveitme",text:"How natural does &ldquo;<i>Give it me</i>&rdquo; sound to you (for <i>give it to me</i>)?",tag:"real data",real:true,metric:"pct",
   slider:true,grid:"giveitme",sliderLabels:["Sounds wrong","Sounds fine"],info:"giveitme",infoLabel:"the &lsquo;give it me&rsquo; dative"},
  {id:"splinter",text:"What do you call a small piece of wood stuck in your skin?",tag:"",real:true,multi:true,metric:"prevalence",
   info:"splinter",infoLabel:"words for a splinter",
   opts:[
     {label:"Splinter",v:"splinter",term:"splinter",grid:"splinter"},
     {label:"Spelk",v:"spelk",term:"spelk",grid:"spelk"},
     {label:"Spell",v:"spell",term:"spell",grid:"spell"},
     {label:"Shiver",v:"shiver",term:"shiver",grid:"shiver"},
     {label:"Sliver",v:"sliver",term:"sliver",grid:"sliver"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_splinter",excl:true,none:true}
   ]},
  // metric "prevalence": lexical variants overlap in a speaker's lexicon and the
  // surface is a relative Gi* hotspot, not a headcount -> show a qualitative band,
  // NOT a fake percentage.
  {id:"bread",text:"What do you call a small bread roll?",tag:"real data (bread-roll survey)",real:true,phon:false,multi:true,metric:"prevalence",
   opts:[
     {label:"Barm / barm cake",v:"barm",term:"barm",grid:"barm"},
     {label:"Tea cake",v:"teacake",term:"tea cake",grid:"teacake"},
     {label:"Muffin",v:"muffin",term:"muffin",grid:"muffin"},
     {label:"Cob",v:"cob",term:"cob",grid:"cob"},
     {label:"Batch",v:"batch",term:"batch",grid:"batch"},
     {label:"Bap",v:"bap",term:"bap",grid:"bap"},
     {label:"Bun",v:"bun",term:"bun",grid:"bun"},
     {label:"Roll",v:"roll",term:"roll",grid:"roll"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_bread",excl:true,none:true}
   ]},
  // single-select lexical with a photo. "ice lolly" is standard; "lolly ice" is the
  // Merseyside reversal. "other"/"no word" don't map to a region (inconclusive).
  {id:"lolly",text:"What would you call this frozen treat?",
   img:POPSICLE_IMG,tag:"",real:true,metric:"prevalence",info:"lolly",infoLabel:"ice lolly vs lolly ice",
   opts:[
     {label:"Ice lolly",v:"icelolly",term:"ice lolly",grid:"icelolly"},
     {label:"Lolly ice",v:"lollyice",term:"lolly ice",grid:"lollyice"},
     {label:"I use both interchangeably",v:"both",term:"both",grid:"lollyice"},
     {label:"Other term (ice pop, popsicle, etc.)",v:"other",term:"another term",none:true},
     {label:"I have no word for this",v:"none",term:"no word for this",none:true}
   ]}
];
// foot-strut rate is still only an estimate -> don't fake precision at the extremes
function fmtPct(p){if(p>=88)return "90%%+";if(p<=12)return "under 10%%";return "~"+(Math.round(p/5)*5)+"%%";}
// etymologies / notes (sourced from Wiktionary), shown only when the user asks
const ETYM={
  barm:"<b>barm</b> &mdash; from Old English <i>beorma</i> \\"yeast, the froth on fermenting liquor\\" (Proto-West-Germanic *bermō). The bread-roll sense is a shortening of <i>barm cake</i>: bread raised with barm.",
  teacake:"<b>tea cake</b> &mdash; a compound of <i>tea</i> + <i>cake</i>: a light bun eaten at tea; in the North, a soft round bread roll.",
  muffin:"<b>muffin</b> &mdash; probably a diminutive of Low German <i>Muffe</i> \\"small cake\\" (Middle Low German <i>muffe</i> \\"small pastry\\"); alternatively from Old French <i>(pain) moflet</i> \\"soft bread, roll\\".",
  cob:"<b>cob</b> &mdash; uncertain. \\"Cob\\" carries many unrelated senses of diverse origin, several meaning a rounded lump or head; the exact source of the bread sense can't be established.",
  batch:"<b>batch</b> &mdash; from Old English <i>bæcc</i> \\"something baked\\", related to <i>bacan</i> \\"to bake\\": literally a quantity baked together.",
  bap:"<b>bap</b> &mdash; originally Scottish English (16th c.), of unknown origin.",
  bun:"<b>bun</b> &mdash; from Middle English <i>bunne</i>, from Anglo-Norman <i>bugne</i> \\"bump, fritter\\" (Old French), ultimately \\"little lump\\".",
  roll:"<b>roll</b> &mdash; named for its rolled, rounded shape; via Old French <i>rol(l)er</i> from Latin <i>rotulus</i> \\"little wheel\\".",
  footstrut:"<b>The FOOT&ndash;STRUT split</b> &mdash; in the 17th century the Middle English short <i>u</i> /&#650;/ split, in southern England, into two vowels: /&#650;/ (FOOT) and a new unrounded /&#652;/ (STRUT). The split never reached most of the North &amp; Midlands, where <i>foot</i> and <i>strut</i> still share /&#650;/ and rhyme.",
  tvd:"<b>tea vs dinner</b> &mdash; the name for the evening meal. <i>Tea</i> is the northern (and traditionally working-class) term; <i>dinner</i> the southern one, and historically the &lsquo;U&rsquo;/upper-class usage (Ross, 1954). So it carries a class edge as well as a regional one.",
  bookspook:"<b>book vs spook</b> &mdash; in some accents the <i>-ook</i> words (book, cook, look) keep the old long vowel /u&#720;/, so <i>book</i> is [bu&#720;k] and rhymes with <i>spook</i> &mdash; putting it in the GOOSE set rather than FOOT. Traditional in the North East and Stoke (and once Liverpool); Scotland has no foot&ndash;goose distinction at all.",
  trapbath:"<b>The trap&ndash;bath split</b> &mdash; in the 18th century southern English lengthened the <i>a</i> in a set of words (<i>bath, grass, last, dance</i>) to /&#593;&#720;/, splitting them from TRAP words (<i>cat, trap</i>). The North, Wales and Scotland kept the short /a/ &mdash; so a northerner says [ba&#952;], a southerner [b&#593;&#720;&#952;]. It&rsquo;s one of the sharpest north&ndash;south markers.",
  splinter:"<b>Words for a splinter</b> of wood in the skin. <b>Splinter</b> is the standard nationwide; <b>spelk</b> (from Old Norse / Old English <i>spelc</i>) belongs to the North East &amp; the Borders; <b>spell</b> is northern; <b>shiver</b> is East Anglian; <b>sliver</b> is a South East word.",
  giveitme:"<b>&lsquo;Give it me&rsquo;</b> &mdash; the &lsquo;alternative double-object&rsquo; dative: the theme (<i>it</i>) comes before the goal (<i>me</i>) with no <i>to</i> &mdash; <i>give it me</i> rather than <i>give it to me</i> or <i>give me it</i>. It&rsquo;s a North West &amp; Midlands feature (strongest around Manchester and the Potteries), thinning towards the North East and the South.",
  lolly:"<b>Ice lolly vs lolly ice</b> &mdash; <i>ice lolly</i> is the standard British term; <i>lolly ice</i> (the words reversed) is the well-known Merseyside / Liverpool (&lsquo;Scouse&rsquo;) form. Further afield you&rsquo;ll hear <i>ice pop</i> (Ireland, Scotland) or <i>popsicle</i> (North America)."
};
// only etymology sources are cited (the maps are our own recreations, not originals)
const ETYM_SRC="Wiktionary";
const SRC={ splinter:"Wiktionary" };
// lexical prevalence: a relative band, no misleading headcount
function band(v){return v>=0.5?"the main word(s) here":v>=0.3?"common here":v>=0.15?"one of several here":"rarely used here";}
// for the "no word" negative map: high v = the words are absent here
function bandNone(v){return v>=0.5?"few people have a word":v>=0.3?"a word is less usual":v>=0.15?"most people have a word":"nearly everyone has a word";}
let idx=0; const answers={}; const revealedSet=new Set();
const cv=document.getElementById("cv"),cx=cv.getContext("2d");cv.width=W*CELL;cv.height=H_*CELL;
let SHOWN=null;
// national average of each map (over land), so hover can spot LOCALLY distinctive words
const gridMean={};
for(const k in grids){let s=0,n=0;const g=grids[k];
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const v=g[r][c];if(v!=null){s+=v;n++;}}
  gridMean[k]=n?s/n:1e-6;}

function clearRight(prompt){
  // no map yet -> hide the whole right panel and let the left frame center on the page
  document.getElementById("right").style.display="none";
  document.getElementById("rprompt").style.display="";cv.style.display="none";
  document.getElementById("legend").style.display="none";document.getElementById("detail").innerHTML="";
  document.getElementById("match").innerHTML="";
  document.getElementById("infowrap").style.display="none";
  document.getElementById("rtitle").innerHTML=prompt;
}
function render(){
  const prog=document.getElementById("progress"),qt=document.getElementById("qtext"),
        tag=document.getElementById("qtag"),box=document.getElementById("opts"),
        done=document.getElementById("leftdone"),back=document.getElementById("back"),
        next=document.getElementById("next"),hint=document.getElementById("hint");
  // progress bar tracks the CURRENT question (idx), so it stays in sync with
  // "Question N of M" and moves with Back/Continue
  const atEnd=idx>=QUESTIONS.length;
  document.getElementById("pbar").style.width=(atEnd?100:(idx+1)/QUESTIONS.length*100)+"%%";
  // back is always available: on the first question it returns to the landing page
  back.style.display="inline";
  back.textContent=(idx===0)?"\\u2190 intro":"\\u2190 back";
  document.getElementById("startover").style.display="none";   // redundant with the top "Restart quiz"
  next.style.display="none"; hint.style.display="none"; tag.style.display="none";
  if(atEnd){
    prog.textContent="All questions answered";
    qt.style.display="none";box.style.display="none";
    done.style.display="";done.innerHTML="<b>That&rsquo;s every question for now.</b><br>"+
      "More coming &mdash; and a combined &ldquo;place me&rdquo; result down the line.";
    clearRight("&nbsp;"); return;
  }
  const q=QUESTIONS[idx];
  prog.textContent="Question "+(idx+1)+" of "+QUESTIONS.length;
  qt.style.display="";qt.innerHTML=q.text; box.style.display="";box.innerHTML="";done.style.display="none";
  const qimg=document.getElementById("qimg");
  if(q.img){ qimg.src=q.img; qimg.style.display="block"; } else { qimg.style.display="none"; }
  const ans=answers[q.id];
  const contLabel=(idx===QUESTIONS.length-1)?"Finish →":"Continue →";
  const answered=q.multi?(Array.isArray(ans)&&ans.length>0):(ans!==undefined);
  const isRevealed=revealedSet.has(q.id) && answered;
  // ---- options: ALWAYS editable (so Back lets you change answers), BUT changing the
  // selection hides the map — you only ever see it by pressing "See map". ----
  if(q.slider){
    // 1-5 acceptability slider
    hint.innerHTML=isRevealed?"":"Drag the slider, then press &ldquo;See map&rdquo;.";
    const v=(ans!==undefined)?ans:3;
    const wrap=document.createElement("div"); wrap.className="sliderbox";
    wrap.innerHTML="<input type='range' id='slider' min='1' max='5' step='1' value='"+v+"'>"+
      "<div class='slabels'><span>1<br>"+q.sliderLabels[0]+"</span>"+
      "<span class='sval'>"+(ans!==undefined?v:"")+"</span>"+
      "<span style='text-align:right'>5<br>"+q.sliderLabels[1]+"</span></div>";
    box.appendChild(wrap);
    const sl=wrap.querySelector("#slider"), sval=wrap.querySelector(".sval");
    sl.oninput=()=>{ sval.textContent=sl.value; };
    sl.onchange=()=>{ answers[q.id]=+sl.value; revealedSet.delete(q.id); render(); };
  } else if(q.multi){
    const selected=new Set(Array.isArray(ans)?ans:[]);
    hint.innerHTML=isRevealed?"":"Select all that apply, then press &ldquo;See map&rdquo;.";
    q.opts.forEach(o=>{const bt=document.createElement("button");
      bt.className="opt"+(selected.has(o.v)?" sel":"");
      bt.innerHTML="<span class='box'>"+(selected.has(o.v)?"&#10003;":"")+"</span>"+o.label;
      bt.onclick=()=>{ let s=new Set(Array.isArray(answers[q.id])?answers[q.id]:[]);
        if(o.excl){ if(s.has(o.v)) s.clear(); else s=new Set([o.v]); }   // "no word" is exclusive
        else { if(s.has(o.v))s.delete(o.v); else s.add(o.v);
          q.opts.filter(x=>x.excl).forEach(x=>s.delete(x.v)); }
        if(s.size) answers[q.id]=q.opts.filter(x=>s.has(x.v)).map(x=>x.v); else delete answers[q.id];
        revealedSet.delete(q.id);   // selection changed -> hide map until "See map" pressed
        render(); };
      box.appendChild(bt);});
  } else {
    hint.textContent="";
    q.opts.forEach(o=>{const bt=document.createElement("button");
      bt.className="opt"+(ans===o.v?" sel":"");
      bt.innerHTML=o.label;
      bt.onclick=()=>{ answers[q.id]=o.v; revealedSet.delete(q.id); render(); };   // change -> hide map until "See map"
      box.appendChild(bt);});
  }
  hint.style.display="block";   // always reserve this row so the buttons never jump
  // ---- primary button: "See map" first, then (once revealed) "Continue" ----
  next.style.display="block";
  if(isRevealed){
    drawMap(q,ans);
    next.disabled=false; next.textContent=contLabel; next.onclick=()=>{idx++;render();};
  } else {
    clearRight("Choose your answer"+(q.multi?"(s)":"")+", then press &ldquo;See map&rdquo;");
    next.disabled=!answered; next.textContent="See map →";
    next.onclick=()=>{ if(answered){revealedSet.add(q.id);render();} };
  }
}
document.getElementById("back").onclick=()=>{
  if(idx===0){ showIntro(); return; }   // first question -> landing page
  idx--; render();                      // go back one and SHOW that question's last map (keep its answer)
};
function restartQuiz(){ idx=0; revealedSet.clear(); for(const k in answers)delete answers[k]; showIntro(); }
document.getElementById("restart").onclick=restartQuiz;
document.getElementById("startover").onclick=restartQuiz;

function heat(t){t=Math.max(0,Math.min(1,t));
  const blue=[15,77,209],white=[222,222,230],red=[209,15,28];
  const mix=(A,B,k)=>[Math.round(A[0]+(B[0]-A[0])*k),Math.round(A[1]+(B[1]-A[1])*k),Math.round(A[2]+(B[2]-A[2])*k)];
  return t<0.5?mix(blue,white,t/0.5):mix(white,red,(t-0.5)/0.5);}

// Where does this answer concentrate? Mean surface per region; if one/a few
// regions clearly stand out, name them (grouping adjacent ones); if it's
// scattered or nothing stands out, say "multiple regions".
function joinRegions(a){return a.length<=1?a[0]:a.slice(0,-1).join(", ")+" &amp; "+a[a.length-1];}
function matchRegion(surf){
  const sum=regionNames.map(()=>[0,0]);
  let peakVal=-1,peakReg=-1;
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const g=regionGrid[r][c];if(g<0)continue;
    const v=surf[r][c];if(v==null)continue;sum[g][0]+=v;sum[g][1]++;
    if(v>peakVal){peakVal=v;peakReg=g;}}
  const mean=sum.map(([s,n])=>n?s/n:0);
  const order=mean.map((v,i)=>i).sort((a,b)=>mean[b]-mean[a]);
  const topMean=mean[order[0]];
  const NORTHSET=["Yorkshire","the North West","the North East"];
  const SOUTHSET=["the South East","the South West","East Anglia"];
  const hot=[];for(let i=0;i<mean.length;i++) if(mean[i]>=0.6*topMean && mean[i]>=0.32) hot.push([regionNames[i],mean[i]]);
  hot.sort((a,b)=>b[1]-a[1]);
  // a genuinely strong regional signal -> name it (or the grouping)
  if(topMean>=0.40 && hot.length){
    const nm=hot.map(h=>h[0]);
    // spread across both north and south and most regions -> it's basically nationwide
    if(nm.length>=6 && NORTHSET.some(n=>nm.includes(n)) && SOUTHSET.some(n=>nm.includes(n)))
      return "much of Britain";
    // if all three northern regions dominate -> "the North of England"
    if(NORTHSET.every(n=>nm.includes(n))) return "the North of England";
    if(nm.length<=3) return joinRegions(nm);
    return "multiple regions ("+nm.slice(0,3).join(", ")+"&hellip;)";
  }
  // a sharp LOCAL peak in an otherwise-cool region (muffin=Manchester, batch=Coventry)
  if(peakVal>=0.45 && mean[peakReg]<0.35) return regionNames[peakReg];
  // otherwise the signal is weak / spread across places (e.g. bap) -> multiple regions
  return "multiple regions ("+order.slice(0,3).map(i=>regionNames[i]).join(", ")+"&hellip;)";
}

// for the slider: place the user by matching their rating (0-1) to the region whose
// average acceptance is closest to it (rate it high -> the high-acceptance regions).
function matchByLevel(surf,target){
  const sum=regionNames.map(()=>[0,0]);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const g=regionGrid[r][c];if(g<0)continue;
    const v=surf[r][c];if(v==null)continue;sum[g][0]+=v;sum[g][1]++;}
  const mean=sum.map(([s,n])=>n?s/n:0);
  const order=mean.map((v,i)=>i).sort((a,b)=>Math.abs(mean[a]-target)-Math.abs(mean[b]-target));
  return regionNames[order[0]];
}

function drawMap(q,ans){
  document.getElementById("right").style.display="";   // map revealed -> show the right panel
  document.getElementById("rprompt").style.display="none";cv.style.display="block";
  document.getElementById("legend").style.display="flex";
  const multi=Array.isArray(ans);
  const isSlider=!!q.slider;
  let sel, surf=[], incon=false;
  if(isSlider){
    // 1-5 rating -> show where people FEEL AS YOU DO: high rating leans to the
    // acceptance map (NW/Midlands red); low rating leans to its inverse (red across
    // the many places that reject it). The raw acceptance rate is kept for the hover.
    sel=[{term:q.id,word:"acceptance"}];
    const base=grids[q.grid]; const w=(ans-1)/4;   // 0=sounds wrong .. 1=sounds fine
    for(let r=0;r<H_;r++){surf.push([]);for(let c=0;c<W;c++){
      const b=base[r][c]; surf[r].push(b==null?null:(w*b+(1-w)*(1-b)));}}
  } else if(multi){
    // MULTIPLE SELECT: layer the chosen maps (per-pixel max = "where ANY are common")
    sel=ans.map(v=>q.opts.find(o=>o.v===v));
    for(let r=0;r<H_;r++){surf.push([]);for(let c=0;c<W;c++){
      if(!land[r][c]){surf[r].push(null);continue;}
      let m=0,any=false;
      for(const o of sel){ if(!o.grid)continue; const val=grids[o.grid][r][c]; if(val!=null){any=true; if(val>m)m=val;}}
      surf[r].push(any?m:null);}}
    incon = sel.length===1 && sel[0].none;
  } else {
    const opt=q.opts.find(o=>o.v===ans); sel=[opt];
    if(opt.grid){
      const base=grids[opt.grid];
      for(let r=0;r<H_;r++){surf.push([]);for(let c=0;c<W;c++){surf[r].push(base[r][c]);}}
    } else if(grids[q.id]){   // binary pct question: flip for the "no" option
      const base=grids[q.id];
      for(let r=0;r<H_;r++){surf.push([]);for(let c=0;c<W;c++){
        surf[r].push(base[r][c]==null?null:(ans?base[r][c]:1-base[r][c]));}}
    } else {                  // option with no map (e.g. "other" / "no word") -> inconclusive
      incon=true;
      for(let r=0;r<H_;r++){surf.push([]);for(let c=0;c<W;c++){surf[r].push(null);}}
    }
  }
  SHOWN={q,ans,sel,surf,multi,noneOnly:incon,incon,slider:isSlider,raw:isSlider?grids[q.grid]:null};
  const rtitle=document.getElementById("rtitle");
  if(isSlider){
    rtitle.innerHTML="You rated it <b>"+ans+" / 5</b><br><span style='color:#8a857c'>Red = where people tend to feel the same &middot; hover for local acceptance</span>";
  } else if(incon){
    rtitle.innerHTML="You chose <b>&ldquo;"+(sel[0].term||"this")+"&rdquo;</b><br><span style='color:#8a857c'>This one doesn&rsquo;t map to a particular region</span>";
  } else {
    const answerLabel = multi ? sel.map(o=>"&ldquo;"+o.term+"&rdquo;").join(", ") : "&ldquo;"+sel[0].label+"&rdquo;";
    rtitle.innerHTML="You chose <b>"+answerLabel+"</b><br><span style='color:#8a857c'>"+
      (q.metric==="pct"?"Hover a city to see the approximate percentage"
                       :"Hover a city to see the word most likely used there")+"</span>";
  }
  cx.clearRect(0,0,cv.width,cv.height);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){
    if(!land[r][c])continue;
    cx.fillStyle="#c9c9d2";cx.fillRect(c*CELL,r*CELL,CELL,CELL);
    const v=surf[r][c];
    const [rr,gg,bb]=(incon||v==null)?[214,214,220]:heat(v);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fillRect(c*CELL,r*CELL,CELL-GAP,CELL-GAP);
  }
  for(const ct of cities){const v=surf[ct.row|0]?surf[ct.row|0][ct.col|0]:null;
    const [rr,gg,bb]=(incon||v==null)?[200,200,205]:heat(v);
    cx.beginPath();cx.arc((ct.col+0.5)*CELL,(ct.row+0.5)*CELL,5,0,7);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fill();cx.lineWidth=2;cx.strokeStyle="#2b2b2b";cx.stroke();}
  document.getElementById("match").innerHTML=
    (incon || (isSlider && ans===3))
    ? "<span style='color:#8a857c;font-weight:500'>Inconclusive &mdash; this doesn&rsquo;t point to a particular region.</span>"
    : "&#9873; closest to: <b>"+matchRegion(surf)+"</b>";
  // (i) more info — resolved per question so it's never "undefined"
  let infoHTML="", infoLabel="", infoSrc="";
  if(q.info){ infoHTML=ETYM[q.info]||""; infoLabel=q.infoLabel||""; infoSrc=SRC[q.info]||""; }
  else if(q.multi){ const parts=sel.map(o=>ETYM[o.grid]).filter(Boolean);
    infoHTML=parts.join("<hr class='isep'>"); infoLabel="your word"+(sel.length>1?"s":""); infoSrc=ETYM_SRC; }
  const infowrap=document.getElementById("infowrap"), infobtn=document.getElementById("infobtn"), info=document.getElementById("info");
  if(infoHTML){ infowrap.style.display="inline-block"; infobtn.innerHTML="&#9432; about "+infoLabel;
    info.innerHTML=infoHTML+(infoSrc?"<div class='src'>Source: "+infoSrc+"</div>":"");   // shows on hover via CSS
  } else { infowrap.style.display="none"; }
  document.getElementById("detail").innerHTML=q.ipa?
    "Click a city to see the expected local IPA for <i>foot</i> vs <i>cut</i>":"";
}

const tip=document.getElementById("tip");
cv.addEventListener("mousemove",(e)=>{
  if(!SHOWN){tip.style.opacity=0;return;}
  const rect=cv.getBoundingClientRect(),sx=cv.width/rect.width,sy=cv.height/rect.height;
  const x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy;
  let best=null,bd=1e9;for(const ct of cities){const dd=Math.hypot((ct.col+0.5)*CELL-x,(ct.row+0.5)*CELL-y);if(dd<bd){bd=dd;best=ct;}}
  if(best&&bd<=24){const v=SHOWN.surf[best.row|0]?SHOWN.surf[best.row|0][best.col|0]:null;
    const br=best.row|0,bc=best.col|0;
    let line;
    if(SHOWN.incon){ line="&mdash;"; }
    else if(SHOWN.slider){ const rv=SHOWN.raw[br]?SHOWN.raw[br][bc]:null; line=(rv==null)?"&mdash;":fmtPct(rv*100)+" acceptance"; }
    else if(SHOWN.q.metric==="pct"){ line=(v==null)?"&mdash;":fmtPct(v*100)+" "+SHOWN.sel[0].word; }
    else{
      // lexical: show the MAIN word here (top absolute) AND, separately, the local/
      // regional word — the variant that is unusually common here relative to its own
      // national average (its "lift"). That surfaces low-frequency regional words like
      // shiver (East Anglia) or spelk (NE) even when splinter dominates in raw numbers.
      const seen=new Set();
      const realOpts=SHOWN.q.opts.filter(o=>{ if(!o.grid||o.none||seen.has(o.grid))return false; seen.add(o.grid); return true; });
      let main=null,mainV=-1, reg=null,bestLift=1.35;
      for(const o of realOpts){const gr=grids[o.grid][br]; const gv=gr?gr[bc]:null;
        if(gv==null)continue;
        if(gv>mainV){mainV=gv;main=o;}
        const lift=gv/(gridMean[o.grid]||1e-6);
        if(gv>=0.10 && lift>bestLift){bestLift=lift;reg=o;}}
      if(!main){ line="&mdash;"; }
      else{ line="<b>&ldquo;"+main.term+"&rdquo;</b>";
        if(reg && reg!==main) line+=" &middot; local variant(s): <b>&ldquo;"+reg.term+"&rdquo;</b>"; }
    }
    tip.innerHTML="<b>"+best.name+"</b><br><small>"+line+"</small>";
    tip.style.left=((best.col+0.5)*CELL/sx)+"px";tip.style.top=((best.row+0.5)*CELL/sy)+"px";tip.style.opacity=1;}
  else tip.style.opacity=0;});
cv.addEventListener("mouseleave",()=>tip.style.opacity=0);

// click a city (foot-strut only) -> expected local IPA
const detail=document.getElementById("detail");
cv.addEventListener("click",(e)=>{
  if(!SHOWN||!SHOWN.q.ipa)return;
  const rect=cv.getBoundingClientRect(),sx=cv.width/rect.width,sy=cv.height/rect.height;
  const x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy;
  let best=null,bd=1e9;for(const ct of cities){const dd=Math.hypot((ct.col+0.5)*CELL-x,(ct.row+0.5)*CELL-y);if(dd<bd){bd=dd;best=ct;}}
  if(best&&bd<=26){const rhymes=(grids.q1[best.row|0][best.col|0]||0)>=0.5;
    const cut=rhymes?"k&#650;t":"k&#652;t";
    detail.innerHTML="<b>"+best.name+"</b> &mdash; "+
      (rhymes?"foot &amp; cut <b>rhyme</b> (both /&#650;/)":"foot &amp; cut are <b>distinct</b> (/&#650;/ vs /&#652;/)")+
      "<br><span class='ipa'>/f&#650;t/ &middot; /"+cut+"/</span>";}
});

// ---- landing page: pixel map of Great Britain's major dialect groups ----
function drawMini(){
  const rows=HIRES, Hh=rows.length, Wh=rows[0].length, M=2;   // fine county-shaped raster
  const mc=document.getElementById("introcv"); mc.width=Wh*M; mc.height=Hh*M;
  mc.style.width=(Wh*M/2)+"px";   // height is auto (CSS) -> scales proportionally, stays responsive
  const mx=mc.getContext("2d");
  for(let r=0;r<Hh;r++){const row=rows[r];for(let c=0;c<Wh;c++){
    const ch=row[c]; if(ch===".")continue;
    const col=dialectColors[+ch][1];
    mx.fillStyle="rgb("+col[0]+","+col[1]+","+col[2]+")";
    mx.fillRect(c*M,r*M,M,M);
  }}
  // hover -> name the dialect group under the cursor (replaces the old static key)
  const tip=document.getElementById("introtip"), cap=document.getElementById("introcap");
  mc.onmousemove=(e)=>{const rect=mc.getBoundingClientRect();
    const c=Math.floor((e.clientX-rect.left)/(rect.width/Wh)), r=Math.floor((e.clientY-rect.top)/(rect.height/Hh));
    const ch=(r>=0&&r<Hh&&c>=0&&c<Wh)?rows[r][c]:".";
    const di=(ch===".")?-1:+ch;
    if(di>=0){const [nm,col]=dialectColors[di];
      tip.textContent=nm; tip.style.background="rgb("+col[0]+","+col[1]+","+col[2]+")";
      tip.style.left=(e.clientX-rect.left)+"px"; tip.style.top=(e.clientY-rect.top)+"px"; tip.style.opacity=1;
      cap.textContent=nm; cap.style.color="rgb("+col[0]+","+col[1]+","+col[2]+")";
    } else { tip.style.opacity=0; }};
  mc.onmouseleave=()=>{tip.style.opacity=0; cap.textContent="Hover the map to explore the dialect groups"; cap.style.color="";};
}
function startQuiz(){
  document.getElementById("intro").style.display="none";
  document.querySelector("header").style.display="";     // show the title bar during the quiz
  document.getElementById("progresswrap").style.display="";
  document.getElementById("progress").style.display="";
  document.getElementById("stage").style.display="flex";
  document.getElementById("restart").style.display="inline-flex";
  render();
}
function showIntro(){
  document.getElementById("intro").style.display="flex";
  document.querySelector("header").style.display="none"; // landing has its own title in the left panel
  document.getElementById("progresswrap").style.display="none";
  document.getElementById("progress").style.display="none";
  document.getElementById("stage").style.display="none";
  document.getElementById("restart").style.display="none";
}
document.getElementById("startbtn").onclick=startQuiz;
drawMini();
showIntro();
</script></body></html>""" % (
    W, H, json.dumps(grids_all), json.dumps(landj),
    json.dumps(cities), json.dumps(cg.tolist()), json.dumps(names),
    json.dumps(region_grid), json.dumps(region_names),
    json.dumps(dialect_grid), json.dumps(dialect_colors),
    json.dumps(hires_dialect),
    json.dumps(popsicle_uri),
)

with open("index.html", "w") as f:
    f.write(html)
print("wrote index.html — done. Now just: git add index.html build_quiz.py && git commit -m 'update' && git push")
