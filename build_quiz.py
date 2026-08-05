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

# question photos, embedded so the page stays self-contained
def _img_uri(path, mime):
    return "data:" + mime + ";base64," + base64.b64encode(open(path, "rb").read()).decode()
icelolly_uri = _img_uri("ice_lolly-pic.webp", "image/webp")
bread_uri = _img_uri("bread-pic.jpg", "image/jpeg")

with open("britain_pixel_data.json") as f:
    d = json.load(f)
W, H = d["width"], d["height"]
fs = d["footstrut_gi_star"]
cg = np.array(d["county_grid"])
names = d["county_names"]
a = d["footstrut_pct_formula"]["a"]
b = d["footstrut_pct_formula"]["b"]
land = cg >= 0

# The coarse county raster leaves a few sea cells fully enclosed by land (mostly in
# Scotland), which show up as holes in the pixel maps. Fill each such hole with its
# nearest county so every map (landing + quiz surfaces) is gap-free.
_filled = ndimage.binary_fill_holes(land)
_holes = _filled & ~land
if _holes.any():
    _ind = ndimage.distance_transform_edt(~land, return_distances=False, return_indices=True)
    cg[_holes] = cg[tuple(_ind)][_holes]
    land = cg >= 0

# Per-cell surfaces decoded from the source Gi* PNGs (pre-computed into decoded_maps.json so
# the build itself needs no image libraries). These give foot-strut-level detail. Each stored
# array is the viridis position (0-1, low->high Gi*); we stretch it to a sensible acceptance
# range [lo,hi] so it plugs into the pct pipeline (0.5 = the split; hover shows ~a percentage).
_decoded = json.load(open("decoded_maps.json"))


def decoded_surface(key, lo, hi):
    a = np.array([[v if v is not None else np.nan for v in row] for row in _decoded[key]], dtype=float)
    vals = a[~np.isnan(a)]
    pmin, pmax = float(vals.min()), float(vals.max())
    p = (a - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(a)
    return np.where(land, lo + np.nan_to_num(p) * (hi - lo), 0.0)


def decoded_pct_surface(key):
    # like decoded_surface, but the decoded value already IS the true 0-1 percentage
    # (matched against the source legend's own %-bin colours), so no min-max rescale.
    a = np.array([[v if v is not None else np.nan for v in row] for row in _decoded[key]], dtype=float)
    return np.where(land, np.nan_to_num(a), 0.0)


def is_val(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


# ---- foot-strut: real decoded surface -> P(rhyme) ----
gi = np.array([[fs[r][c] if is_val(fs[r][c]) else np.nan for c in range(W)] for r in range(H)])
ind = ndimage.distance_transform_edt(np.isnan(gi), return_distances=False, return_indices=True)
gi = gi[tuple(ind)]
q1 = np.clip(a * gi + b, 0, 100) / 100.0
# smooth out the granular noise so it reads as clean regional patterns
_num = ndimage.gaussian_filter(np.where(land, q1, 0.0), 1.8)
_den = ndimage.gaussian_filter(land.astype(float), 1.8)
q1 = np.where(land & (_den > 1e-6), _num / _den, 0.0)

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
    # peak West Yorkshire + East Lancashire (Pennines), north tongue to Cumbria. A LOCAL,
    # MODERATE word (competes with 'bread cake'/'bun') -> not the dominant term anywhere.
    "teacake": mk(4, None, {"Yorkshire": 40, "Lancashire": 28, "Westmorland": 22, "Cumberland": 20,
                            "Cheshire": 12, "Derbyshire": 14, "Durham": 14}),
    # tight on Manchester, but a LOCAL MINORITY (competes with barm) -> modest values
    "muffin": mk(3, None, {"Lancashire": 34, "Cheshire": 16, "Yorkshire": 9, "Derbyshire": 8,
                           "Warwickshire": 9, "Staffordshire": 8}),
    # East Midlands core (Notts/Derby/Leics), greening W into the Marches; low Lincolnshire
    "cob": mk(6, [(WMIDS, 32)], {"Nottinghamshire": 80, "Leicestershire": 68, "Derbyshire": 62,
                                 "Rutland": 55, "Northamptonshire": 42, "Lincolnshire": 35,
                                 "Staffordshire": 40, "Warwickshire": 22, "Worcestershire": 36,
                                 "Herefordshire": 40, "Shropshire": 38, "Gloucestershire": 22,
                                 "Huntingdonshire": 22}),  # Warwickshire lowered: Birmingham is ~17% cob
    # tight on Coventry/Warwickshire, faint NW + Welsh-border green; ~0 elsewhere
    # batch: a WEST MIDLANDS word centred on Coventry/Birmingham (Warwickshire), greening
    # west into Worcestershire/Herefordshire and south-west toward Gloucestershire/Somerset.
    # NOT a northern word (the earlier Lancashire/Cumbria values were wrong).
    "batch": mk(5, None, {"Warwickshire": 44, "Worcestershire": 32, "Herefordshire": 26,
                          "Gloucestershire": 26, "Staffordshire": 24, "Shropshire": 22,
                          "Somerset": 18, "Oxfordshire": 15, "Leicestershire": 15, "Northamptonshire": 14}),
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
    # peak Yorkshire + Durham (not the far NE); a moderate word, not locally dominant
    "bun": mk(7, None, {"Yorkshire": 48, "Durham": 50, "Lincolnshire": 26, "Northumberland": 40,
                        "Cumberland": 28, "Westmorland": 24, "Nottinghamshire": 18, "Derbyshire": 16,
                        "Hampshire": 14, "Berkshire": 14, "Wiltshire": 14}),
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

# NURSE-SQUARE merger: do NURSE (stir, fur) and SQUARE (stare, fair) rhyme? Two present-day
# cores (square-nurse.png + MacKenzie/Bailey/Turton description): (1) the North West /
# Merseyside — long-established, best-documented, highest rates in the West; (2) the EAST
# COAST — Hull & the East Riding and Teesside/Middlesbrough — now among the highest rates,
# an apparently newer change in progress. The older East-Midlands / West-Midlands merger
# (Notts, Derbys, Leics, Staffs) has largely been stamped out and is now LOW; of Lincolnshire
# only the north-east coast (Grimsby) persists. Low across Scotland, the South, SW and Wales.
NURSESQUARE = mk(9, [(SCOT, 11)],
    {"Lancashire": 76, "Cheshire": 40, "Yorkshire": 44, "Durham": 34, "Northumberland": 18,
     "Cumberland": 15, "Westmorland": 14, "Lincolnshire": 30,
     "Nottinghamshire": 12, "Derbyshire": 12, "Leicestershire": 10, "Rutland": 10,
     "Staffordshire": 12, "Shropshire": 10, "Worcestershire": 10, "Warwickshire": 10,
     "Glamorgan": 14, "Monmouthshire": 12, "Flintshire": 16, "Denbighshire": 14})

# words for a splinter of wood in the skin (each .ppm map = one variant vs splinter):
# spelk = North East + Borders/Cumbria (Old Norse); spell = northern (weak);
# shiver = East Anglian (weak); sliver = scattered; splinter = the standard, common
# everywhere except spelk's NE heartland. Blended from BBC Future + English Dialect App.
# By 2016 splinter has taken over almost everywhere, so the other variants are localized
# MINORITIES (residual), not dominant. spelk is the strongest survivor (the NE).
SPELK = mk(3, [(["Berwickshire", "Roxburghshire", "Selkirkshire", "Peeblesshire"], 26)],
    {"Northumberland": 50, "Durham": 44, "Cumberland": 28, "Westmorland": 24, "Yorkshire": 10,
     "Dumfriesshire": 18, "Lancashire": 8})
SPELL = mk(3, None,
    {"Lancashire": 18, "Cheshire": 14, "Yorkshire": 14, "Cumberland": 14, "Westmorland": 12,
     "Derbyshire": 10, "Flintshire": 10, "Durham": 10})
# shiver: an East Anglian minority, peaked on Norfolk (Norwich); near-zero elsewhere.
SHIVER = mk(1, None, {"Norfolk": 24, "Suffolk": 12, "Cambridgeshire": 6})
# sliver: a South East / Home Counties minority — Essex core, faint through the Home
# Counties (per splinter-variant-map.webp, the orange region around & NE of London).
SLIVER = mk(2, None, {"Essex": 26, "Hertfordshire": 20, "Middlesex": 18, "Cambridgeshire": 14,
                      "Surrey": 16, "Kent": 13, "Sussex": 11, "Bedfordshire": 12, "Buckinghamshire": 9})
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


grids_all = {"q1": grid_json(q1), "tvd": grid_json(decoded_surface("tvd", 0.05, 0.72)),
             "giveitme": grid_json(decoded_surface("giveitme", 0.12, 0.82)),
             "bookspook": grid_json(decoded_surface("bookspook", 0.10, 0.85)),
             "stirstare": grid_json(surface(NURSESQUARE)),
             "scone": grid_json(decoded_pct_surface("scone")),
             "northforce": grid_json(decoded_surface("northforce", 0.10, 0.85)),
             "forcecure": grid_json(decoded_surface("forcecure", 0.10, 0.85)),
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

# names for the playground chasing game (tag-game.jpg, Starkey Comics): each region of
# the source map is a flat colour naming one variant, decoded by nearest-colour match per
# cell (not a Gi* gradient), using nearest-VALID-pixel lookup rather than a windowed
# majority vote so small pockets (tip, dobby, hit, had) don't get washed out by bigger
# neighbours. "catch/chase" turned out to be an Ireland-only term with ~0 presence in GB,
# so it's dropped. Two small pockets (Caithness/Orkney = "tag"; Anglesey = "tip") are
# hand-corrected in the decode script: the comic draws the far north and NW Wales with
# enough distortion that the whole-map linear registration lands a few grid cells off,
# confirmed by direct pixel inspection of the source image.
TAG_TERMS = ["tag", "tick", "tip", "tig", "tiggy", "tuggy", "it", "hit", "had", "touch", "dobby"]
for term in TAG_TERMS:
    grids_all["tag_" + term] = grid_json(decoded_pct_surface("tag_" + term))

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
grids_all["none_tag"] = negative_union(["tag_" + t for t in TAG_TERMS])

landj = [[bool(land[r][c]) for c in range(W)] for r in range(H)]

CITY_LL = {"London", "Manchester", "Birmingham", "Leeds", "Liverpool", "Sheffield",
           "Bristol", "Newcastle", "Nottingham", "Cardiff", "Edinburgh", "Glasgow",
           "Aberdeen", "York", "Norwich", "Cambridge", "Exeter"}
cities = [{"name": n, "col": co, "row": ro} for n, co, ro in d["cities"] if n in CITY_LL]

# representative dialect "places" (the kind the source heat maps label), each with an
# evocative name. The result names the best-matching place(s); 1-3 are shown as dots.
PLACES = [
    ("Liverpool", "Scouse", 51, 91), ("Manchester", "", 57, 90),
    ("Newcastle", "Geordie", 63, 68), ("Middlesbrough", "Teesside", 66, 74),
    ("Leeds", "Yorkshire", 63, 85), ("Sheffield", "", 64, 92), ("Hull", "", 73, 86),
    ("Birmingham", "Brummie", 60, 105), ("Stoke", "the Potteries", 57, 97),
    ("Nottingham", "the East Midlands", 67, 98),
    ("London", "", 75, 120), ("Norwich", "East Anglia", 88, 103),
    ("Margate", "Kent", 86, 122),
    ("Bristol", "the West Country", 54, 120), ("Exeter", "the West Country", 46, 131),
    ("Edinburgh", "Scotland", 49, 53), ("Glasgow", "Scotland", 40, 55),
    ("Aberdeen", "", 58, 35), ("Cardiff", "Wales", 49, 120),
]
places = [{"name": n, "tag": t, "col": co, "row": ro} for n, t, co, ro in PLACES]

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

# (The landing map now uses the same chunky low-res dialect_grid as the quiz maps,
# so the old high-res county-polygon rasterization is no longer needed.)

# Towns/cities of GREAT BRITAIN (England, Scotland, Wales only — no Northern Ireland,
# since the question asks about growing up in GB), each with its county / council area.
# Used for the hometown type-ahead. England = ceremonial counties; Scotland = council
# areas; Wales = principal/preserved counties. Curated but broad; expandable.
GB_TOWNS = sorted(set([
    # ---- England ----
    ("London", "Greater London"), ("Croydon", "Greater London"), ("Bromley", "Greater London"),
    ("Enfield", "Greater London"), ("Harrow", "Greater London"), ("Hounslow", "Greater London"),
    ("Manchester", "Greater Manchester"), ("Salford", "Greater Manchester"), ("Bolton", "Greater Manchester"),
    ("Stockport", "Greater Manchester"), ("Oldham", "Greater Manchester"), ("Rochdale", "Greater Manchester"),
    ("Bury", "Greater Manchester"), ("Wigan", "Greater Manchester"), ("Altrincham", "Greater Manchester"),
    ("Sale", "Greater Manchester"), ("Ashton-under-Lyne", "Greater Manchester"), ("Stretford", "Greater Manchester"),
    ("Liverpool", "Merseyside"), ("Birkenhead", "Merseyside"), ("Bootle", "Merseyside"),
    ("St Helens", "Merseyside"), ("Southport", "Merseyside"), ("Wallasey", "Merseyside"),
    ("Newton-le-Willows", "Merseyside"),
    ("Birmingham", "West Midlands"), ("Coventry", "West Midlands"), ("Wolverhampton", "West Midlands"),
    ("Dudley", "West Midlands"), ("Walsall", "West Midlands"), ("West Bromwich", "West Midlands"),
    ("Solihull", "West Midlands"), ("Sutton Coldfield", "West Midlands"), ("Stourbridge", "West Midlands"),
    ("Halesowen", "West Midlands"), ("Smethwick", "West Midlands"),
    ("Leeds", "West Yorkshire"), ("Bradford", "West Yorkshire"), ("Huddersfield", "West Yorkshire"),
    ("Wakefield", "West Yorkshire"), ("Halifax", "West Yorkshire"), ("Dewsbury", "West Yorkshire"),
    ("Keighley", "West Yorkshire"), ("Batley", "West Yorkshire"), ("Castleford", "West Yorkshire"),
    ("Pontefract", "West Yorkshire"), ("Brighouse", "West Yorkshire"),
    ("Sheffield", "South Yorkshire"), ("Rotherham", "South Yorkshire"), ("Barnsley", "South Yorkshire"),
    ("Doncaster", "South Yorkshire"),
    ("York", "North Yorkshire"), ("Harrogate", "North Yorkshire"), ("Scarborough", "North Yorkshire"),
    ("Middlesbrough", "North Yorkshire"), ("Redcar", "North Yorkshire"), ("Northallerton", "North Yorkshire"),
    ("Ripon", "North Yorkshire"), ("Selby", "North Yorkshire"), ("Skipton", "North Yorkshire"),
    ("Whitby", "North Yorkshire"),
    ("Hull", "East Riding of Yorkshire"), ("Beverley", "East Riding of Yorkshire"),
    ("Bridlington", "East Riding of Yorkshire"), ("Goole", "East Riding of Yorkshire"),
    ("Newcastle upon Tyne", "Tyne and Wear"), ("Sunderland", "Tyne and Wear"), ("Gateshead", "Tyne and Wear"),
    ("South Shields", "Tyne and Wear"), ("Washington", "Tyne and Wear"), ("Whitley Bay", "Tyne and Wear"),
    ("Jarrow", "Tyne and Wear"), ("Wallsend", "Tyne and Wear"),
    ("Durham", "County Durham"), ("Darlington", "County Durham"), ("Hartlepool", "County Durham"),
    ("Stockton-on-Tees", "County Durham"), ("Bishop Auckland", "County Durham"),
    ("Chester-le-Street", "County Durham"), ("Consett", "County Durham"), ("Peterlee", "County Durham"),
    ("Morpeth", "Northumberland"), ("Blyth", "Northumberland"), ("Ashington", "Northumberland"),
    ("Hexham", "Northumberland"), ("Berwick-upon-Tweed", "Northumberland"), ("Cramlington", "Northumberland"),
    ("Preston", "Lancashire"), ("Blackburn", "Lancashire"), ("Blackpool", "Lancashire"),
    ("Burnley", "Lancashire"), ("Lancaster", "Lancashire"), ("Chorley", "Lancashire"),
    ("Accrington", "Lancashire"), ("Nelson", "Lancashire"), ("Morecambe", "Lancashire"),
    ("Leyland", "Lancashire"), ("Fleetwood", "Lancashire"), ("Ormskirk", "Lancashire"),
    ("Skelmersdale", "Lancashire"), ("Lytham St Annes", "Lancashire"),
    ("Chester", "Cheshire"), ("Crewe", "Cheshire"), ("Warrington", "Cheshire"),
    ("Ellesmere Port", "Cheshire"), ("Macclesfield", "Cheshire"), ("Runcorn", "Cheshire"),
    ("Widnes", "Cheshire"), ("Northwich", "Cheshire"), ("Wilmslow", "Cheshire"),
    ("Congleton", "Cheshire"), ("Nantwich", "Cheshire"), ("Winsford", "Cheshire"),
    ("Carlisle", "Cumbria"), ("Barrow-in-Furness", "Cumbria"), ("Kendal", "Cumbria"),
    ("Workington", "Cumbria"), ("Whitehaven", "Cumbria"), ("Penrith", "Cumbria"), ("Ulverston", "Cumbria"),
    ("Lincoln", "Lincolnshire"), ("Grimsby", "Lincolnshire"), ("Scunthorpe", "Lincolnshire"),
    ("Boston", "Lincolnshire"), ("Grantham", "Lincolnshire"), ("Skegness", "Lincolnshire"),
    ("Stamford", "Lincolnshire"), ("Spalding", "Lincolnshire"), ("Gainsborough", "Lincolnshire"),
    ("Nottingham", "Nottinghamshire"), ("Mansfield", "Nottinghamshire"), ("Newark-on-Trent", "Nottinghamshire"),
    ("Worksop", "Nottinghamshire"), ("Retford", "Nottinghamshire"), ("Sutton-in-Ashfield", "Nottinghamshire"),
    ("Derby", "Derbyshire"), ("Chesterfield", "Derbyshire"), ("Ilkeston", "Derbyshire"),
    ("Swadlincote", "Derbyshire"), ("Buxton", "Derbyshire"), ("Glossop", "Derbyshire"),
    ("Long Eaton", "Derbyshire"), ("Belper", "Derbyshire"),
    ("Leicester", "Leicestershire"), ("Loughborough", "Leicestershire"), ("Hinckley", "Leicestershire"),
    ("Coalville", "Leicestershire"), ("Melton Mowbray", "Leicestershire"), ("Wigston", "Leicestershire"),
    ("Oakham", "Rutland"),
    ("Stoke-on-Trent", "Staffordshire"), ("Stafford", "Staffordshire"), ("Burton upon Trent", "Staffordshire"),
    ("Newcastle-under-Lyme", "Staffordshire"), ("Cannock", "Staffordshire"), ("Tamworth", "Staffordshire"),
    ("Lichfield", "Staffordshire"), ("Leek", "Staffordshire"), ("Kidsgrove", "Staffordshire"),
    ("Shrewsbury", "Shropshire"), ("Telford", "Shropshire"), ("Oswestry", "Shropshire"),
    ("Bridgnorth", "Shropshire"), ("Ludlow", "Shropshire"),
    ("Hereford", "Herefordshire"), ("Leominster", "Herefordshire"), ("Ross-on-Wye", "Herefordshire"),
    ("Worcester", "Worcestershire"), ("Redditch", "Worcestershire"), ("Kidderminster", "Worcestershire"),
    ("Bromsgrove", "Worcestershire"), ("Malvern", "Worcestershire"), ("Evesham", "Worcestershire"),
    ("Warwick", "Warwickshire"), ("Nuneaton", "Warwickshire"), ("Rugby", "Warwickshire"),
    ("Leamington Spa", "Warwickshire"), ("Stratford-upon-Avon", "Warwickshire"), ("Bedworth", "Warwickshire"),
    ("Northampton", "Northamptonshire"), ("Kettering", "Northamptonshire"), ("Corby", "Northamptonshire"),
    ("Wellingborough", "Northamptonshire"), ("Rushden", "Northamptonshire"), ("Daventry", "Northamptonshire"),
    ("Norwich", "Norfolk"), ("Great Yarmouth", "Norfolk"), ("King's Lynn", "Norfolk"),
    ("Thetford", "Norfolk"), ("Dereham", "Norfolk"),
    ("Ipswich", "Suffolk"), ("Lowestoft", "Suffolk"), ("Bury St Edmunds", "Suffolk"),
    ("Felixstowe", "Suffolk"), ("Haverhill", "Suffolk"), ("Newmarket", "Suffolk"), ("Sudbury", "Suffolk"),
    ("Cambridge", "Cambridgeshire"), ("Peterborough", "Cambridgeshire"), ("Huntingdon", "Cambridgeshire"),
    ("Ely", "Cambridgeshire"), ("Wisbech", "Cambridgeshire"), ("St Neots", "Cambridgeshire"),
    ("March", "Cambridgeshire"),
    ("Bedford", "Bedfordshire"), ("Luton", "Bedfordshire"), ("Dunstable", "Bedfordshire"),
    ("Leighton Buzzard", "Bedfordshire"), ("Biggleswade", "Bedfordshire"),
    ("Watford", "Hertfordshire"), ("St Albans", "Hertfordshire"), ("Hemel Hempstead", "Hertfordshire"),
    ("Stevenage", "Hertfordshire"), ("Welwyn Garden City", "Hertfordshire"), ("Hitchin", "Hertfordshire"),
    ("Hertford", "Hertfordshire"), ("Bishop's Stortford", "Hertfordshire"), ("Cheshunt", "Hertfordshire"),
    ("Chelmsford", "Essex"), ("Colchester", "Essex"), ("Southend-on-Sea", "Essex"),
    ("Basildon", "Essex"), ("Harlow", "Essex"), ("Braintree", "Essex"), ("Brentwood", "Essex"),
    ("Clacton-on-Sea", "Essex"), ("Grays", "Essex"), ("Loughton", "Essex"), ("Witham", "Essex"),
    ("Milton Keynes", "Buckinghamshire"), ("High Wycombe", "Buckinghamshire"), ("Aylesbury", "Buckinghamshire"),
    ("Amersham", "Buckinghamshire"), ("Marlow", "Buckinghamshire"),
    ("Oxford", "Oxfordshire"), ("Banbury", "Oxfordshire"), ("Bicester", "Oxfordshire"),
    ("Abingdon", "Oxfordshire"), ("Witney", "Oxfordshire"), ("Didcot", "Oxfordshire"),
    ("Reading", "Berkshire"), ("Slough", "Berkshire"), ("Bracknell", "Berkshire"),
    ("Maidenhead", "Berkshire"), ("Newbury", "Berkshire"), ("Windsor", "Berkshire"), ("Wokingham", "Berkshire"),
    ("Guildford", "Surrey"), ("Woking", "Surrey"), ("Epsom", "Surrey"), ("Camberley", "Surrey"),
    ("Farnham", "Surrey"), ("Redhill", "Surrey"), ("Staines", "Surrey"), ("Leatherhead", "Surrey"),
    ("Reigate", "Surrey"), ("Esher", "Surrey"),
    ("Maidstone", "Kent"), ("Canterbury", "Kent"), ("Chatham", "Kent"), ("Gillingham", "Kent"),
    ("Dover", "Kent"), ("Folkestone", "Kent"), ("Margate", "Kent"), ("Ashford", "Kent"),
    ("Royal Tunbridge Wells", "Kent"), ("Gravesend", "Kent"), ("Dartford", "Kent"),
    ("Rochester", "Kent"), ("Sittingbourne", "Kent"), ("Ramsgate", "Kent"), ("Tonbridge", "Kent"),
    ("Brighton", "East Sussex"), ("Hastings", "East Sussex"), ("Eastbourne", "East Sussex"),
    ("Bexhill-on-Sea", "East Sussex"), ("Lewes", "East Sussex"), ("Crowborough", "East Sussex"),
    ("Crawley", "West Sussex"), ("Worthing", "West Sussex"), ("Chichester", "West Sussex"),
    ("Horsham", "West Sussex"), ("Bognor Regis", "West Sussex"), ("Littlehampton", "West Sussex"),
    ("Haywards Heath", "West Sussex"), ("East Grinstead", "West Sussex"),
    ("Southampton", "Hampshire"), ("Portsmouth", "Hampshire"), ("Basingstoke", "Hampshire"),
    ("Gosport", "Hampshire"), ("Fareham", "Hampshire"), ("Eastleigh", "Hampshire"),
    ("Aldershot", "Hampshire"), ("Farnborough", "Hampshire"), ("Winchester", "Hampshire"),
    ("Andover", "Hampshire"), ("Havant", "Hampshire"), ("Waterlooville", "Hampshire"),
    ("Newport", "Isle of Wight"), ("Ryde", "Isle of Wight"), ("Cowes", "Isle of Wight"),
    ("Bournemouth", "Dorset"), ("Poole", "Dorset"), ("Weymouth", "Dorset"),
    ("Dorchester", "Dorset"), ("Christchurch", "Dorset"),
    ("Swindon", "Wiltshire"), ("Salisbury", "Wiltshire"), ("Chippenham", "Wiltshire"),
    ("Trowbridge", "Wiltshire"), ("Devizes", "Wiltshire"),
    ("Taunton", "Somerset"), ("Bath", "Somerset"), ("Yeovil", "Somerset"), ("Bridgwater", "Somerset"),
    ("Weston-super-Mare", "Somerset"), ("Wells", "Somerset"), ("Frome", "Somerset"), ("Glastonbury", "Somerset"),
    ("Bristol", "Bristol"),
    ("Gloucester", "Gloucestershire"), ("Cheltenham", "Gloucestershire"), ("Stroud", "Gloucestershire"),
    ("Cirencester", "Gloucestershire"), ("Tewkesbury", "Gloucestershire"),
    ("Exeter", "Devon"), ("Plymouth", "Devon"), ("Torquay", "Devon"), ("Paignton", "Devon"),
    ("Exmouth", "Devon"), ("Barnstaple", "Devon"), ("Newton Abbot", "Devon"), ("Tiverton", "Devon"),
    ("Bideford", "Devon"), ("Tavistock", "Devon"),
    ("Truro", "Cornwall"), ("Falmouth", "Cornwall"), ("Newquay", "Cornwall"), ("Penzance", "Cornwall"),
    ("St Austell", "Cornwall"), ("Camborne", "Cornwall"), ("Redruth", "Cornwall"), ("Bodmin", "Cornwall"),
    ("Bude", "Cornwall"), ("St Ives", "Cornwall"),
    # ---- Scotland ----
    ("Aberdeen", "Aberdeen City"), ("Peterhead", "Aberdeenshire"), ("Fraserburgh", "Aberdeenshire"),
    ("Inverurie", "Aberdeenshire"),
    ("Dundee", "Dundee City"), ("Arbroath", "Angus"), ("Forfar", "Angus"), ("Montrose", "Angus"),
    ("Edinburgh", "City of Edinburgh"), ("Glasgow", "Glasgow City"),
    ("Inverness", "Highland"), ("Fort William", "Highland"), ("Nairn", "Highland"),
    ("Stirling", "Stirling"), ("Perth", "Perth and Kinross"),
    ("Paisley", "Renfrewshire"), ("Renfrew", "Renfrewshire"),
    ("Falkirk", "Falkirk"), ("Grangemouth", "Falkirk"),
    ("Livingston", "West Lothian"), ("Bathgate", "West Lothian"),
    ("Kilmarnock", "East Ayrshire"), ("Ayr", "South Ayrshire"), ("Irvine", "North Ayrshire"),
    ("Kilwinning", "North Ayrshire"),
    ("Dunfermline", "Fife"), ("Kirkcaldy", "Fife"), ("St Andrews", "Fife"), ("Glenrothes", "Fife"),
    ("Greenock", "Inverclyde"), ("Port Glasgow", "Inverclyde"),
    ("Motherwell", "North Lanarkshire"), ("Coatbridge", "North Lanarkshire"), ("Airdrie", "North Lanarkshire"),
    ("Cumbernauld", "North Lanarkshire"), ("Wishaw", "North Lanarkshire"),
    ("Hamilton", "South Lanarkshire"), ("East Kilbride", "South Lanarkshire"), ("Rutherglen", "South Lanarkshire"),
    ("Dumfries", "Dumfries and Galloway"), ("Stranraer", "Dumfries and Galloway"),
    ("Kirkintilloch", "East Dunbartonshire"), ("Bearsden", "East Dunbartonshire"),
    ("Clydebank", "West Dunbartonshire"), ("Dumbarton", "West Dunbartonshire"),
    ("Elgin", "Moray"), ("Oban", "Argyll and Bute"), ("Helensburgh", "Argyll and Bute"),
    ("Galashiels", "Scottish Borders"), ("Hawick", "Scottish Borders"),
    ("Musselburgh", "East Lothian"), ("Dalkeith", "Midlothian"),
    # ---- Wales ----
    ("Cardiff", "Cardiff"), ("Swansea", "Swansea"), ("Newport", "Gwent"), ("Wrexham", "Wrexham"),
    ("Barry", "Vale of Glamorgan"),
    ("Neath", "Neath Port Talbot"), ("Port Talbot", "Neath Port Talbot"),
    ("Cwmbran", "Torfaen"), ("Pontypool", "Torfaen"),
    ("Bridgend", "Bridgend"),
    ("Llanelli", "Carmarthenshire"), ("Carmarthen", "Carmarthenshire"),
    ("Merthyr Tydfil", "Merthyr Tydfil"),
    ("Pontypridd", "Rhondda Cynon Taf"), ("Aberdare", "Rhondda Cynon Taf"),
    ("Aberystwyth", "Ceredigion"), ("Cardigan", "Ceredigion"),
    ("Bangor", "Gwynedd"), ("Caernarfon", "Gwynedd"),
    ("Llandudno", "Conwy"), ("Colwyn Bay", "Conwy"),
    ("Rhyl", "Denbighshire"), ("Prestatyn", "Denbighshire"),
    ("Ebbw Vale", "Blaenau Gwent"), ("Caerphilly", "Caerphilly"),
    ("Haverfordwest", "Pembrokeshire"), ("Pembroke", "Pembrokeshire"), ("Milford Haven", "Pembrokeshire"),
    ("Newtown", "Powys"), ("Brecon", "Powys"),
]), key=lambda t: t[0])

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><style>
  :root{--bg:#f7f5f0;--ink:#2b2b2b;--muted:#8a857c;--accent:#c0141f;--card:#fff;--line:#e6e1d8;}
  *{box-sizing:border-box;}
  html,body{margin:0;min-height:100%%;}
  body{background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;}
  #app{max-width:1060px;margin:0 auto;padding:20px 20px 16px;position:relative;}
  #qimg{display:block;width:230px;height:230px;object-fit:cover;border-radius:16px;margin:48px auto 0;box-shadow:0 6px 18px rgba(0,0,0,.16);}
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
       background:linear-gradient(to right,rgb(18,86,222),rgb(232,232,236),rgb(214,16,32));border:1px solid #ccc;}
  #infowrap{position:relative;display:inline-block;margin-top:6px;}
  #infobtn{font-size:13px;color:#0a7a63;cursor:help;} #infobtn:hover{text-decoration:underline;}
  #info{display:none;position:absolute;bottom:150%%;left:50%%;transform:translateX(-50%%);width:340px;max-width:82vw;
        font-size:12.5px;line-height:1.5;color:#333;text-align:left;background:var(--card);border:1px solid var(--line);
        border-radius:10px;padding:11px 13px;box-shadow:0 8px 24px rgba(0,0,0,.16);z-index:40;}
  #info::before{content:"";position:absolute;top:100%%;left:50%%;transform:translateX(-50%%);
        border:7px solid transparent;border-top-color:var(--card);}
  #infowrap:hover #info,#infowrap.open #info{display:block;}
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
  #startbtn:hover:not(:disabled){background:#a5101a;transform:translateY(-2px);box-shadow:0 9px 22px rgba(192,20,31,.30);}
  #startbtn:disabled{background:#dcbcbe;cursor:not-allowed;box-shadow:none;transform:none;}
  .intro-note{font-size:12px;color:var(--muted);margin:18px 0 0;letter-spacing:.03em;text-align:center;}
  .hometown-box{margin:2px 0 20px;}
  .hometown-label{display:block;font-size:13px;color:var(--ink);font-weight:600;margin-bottom:7px;}
  .hometown-label .opt-tag{color:var(--muted);font-weight:400;}
  #hometown{width:100%%;padding:11px 14px;font-size:15px;border:1.5px solid var(--line);border-radius:10px;
       background:var(--card);color:var(--ink);outline:none;transition:border-color .12s;}
  #hometown:focus{border-color:var(--accent);}
  .hometown-note{font-size:11.5px;color:var(--muted);line-height:1.45;margin:8px 0 0;}
  .combo{position:relative;}
  .combo-list{list-style:none;margin:5px 0 0;padding:4px;position:absolute;left:0;right:0;z-index:50;
       background:var(--card);border:1.5px solid var(--line);border-radius:10px;box-shadow:0 12px 28px rgba(0,0,0,.16);
       max-height:230px;overflow-y:auto;display:none;text-align:left;}
  .combo-item{padding:9px 12px;font-size:14px;border-radius:7px;cursor:pointer;color:var(--ink);}
  .combo-item .cc{color:var(--muted);font-weight:400;}
  .combo-item:hover,.combo-item.active{background:#fdf0f0;color:var(--accent);}
  .combo-item:hover .cc,.combo-item.active .cc{color:var(--accent);}
  /* first (hometown) question: a distinct tinted card so it doesn't look like a dialect Q */
  .htq{background:#f1ece2;border:1.5px solid var(--line);border-radius:14px;padding:16px;}
  .htq-note{font-size:12px;color:var(--muted);margin:0 0 12px;line-height:1.45;}
  .or-div{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:11px;
       text-transform:uppercase;letter-spacing:.09em;margin:14px 0;}
  .or-div::before,.or-div::after{content:"";flex:1;height:1px;background:#dad3c6;}
  .altbtn{width:100%%;padding:12px 14px;font-size:13.5px;line-height:1.35;background:transparent;
       border:1.5px dashed #c7c0b1;border-radius:10px;color:var(--muted);cursor:pointer;transition:all .12s;}
  .altbtn:hover{border-color:var(--accent);color:var(--accent);background:#fff;}
  .altbtn.chosen{border-style:solid;border-color:var(--accent);color:var(--accent);background:#fdf0f0;}
  .consent{display:flex;align-items:flex-start;gap:8px;font-size:12px;color:var(--muted);
       margin:0 0 18px;text-align:left;line-height:1.45;cursor:pointer;}
  .consent input{margin:1px 0 0;flex:0 0 auto;width:15px;height:15px;accent-color:var(--accent);cursor:pointer;}
  .tlink{color:#0a7a63;text-decoration:underline;cursor:help;position:relative;}
  .terms-pop{display:none;position:absolute;bottom:150%%;left:0;width:330px;max-width:80vw;font-size:11.5px;
       line-height:1.55;color:#333;text-align:left;font-weight:400;background:var(--card);border:1px solid var(--line);
       border-radius:10px;padding:12px 14px;box-shadow:0 10px 26px rgba(0,0,0,.18);z-index:60;}
  .tlink:hover .terms-pop,.tlink.open .terms-pop{display:block;}
  .aboutwrap{position:relative;display:inline-block;vertical-align:middle;}
  .aboutbtn{color:#0a7a63;cursor:help;font-size:15px;line-height:1;}
  .aboutinfo{display:none;position:absolute;bottom:150%%;left:50%%;transform:translateX(-50%%);width:430px;max-width:88vw;
       font-size:12.5px;line-height:1.6;color:#333;text-align:left;background:var(--card);border:1px solid var(--line);
       border-radius:10px;padding:14px 16px;box-shadow:0 10px 28px rgba(0,0,0,.18);z-index:60;letter-spacing:normal;}
  .aboutinfo::after{content:"";position:absolute;top:100%%;left:50%%;transform:translateX(-50%%);
       border:7px solid transparent;border-top-color:var(--card);}
  .aboutwrap:hover .aboutinfo,.aboutwrap.open .aboutinfo{display:block;}
  button,.opt,#infobtn,.aboutbtn{touch-action:manipulation;}   /* removes tap delay on phones */
  /* ---- narrow screens / phones: stop anything running off the edge ---- */
  @media (max-width:640px){
    #app{padding:16px 14px 14px;}
    #left{flex:1 1 auto;max-width:100%%;width:100%%;}
    #right{max-width:100%%;}
    .intro-right{max-width:100%%;}
    /* icon-only restart button on phones so it never covers the title */
    #restart{top:4px;right:10px;padding:7px 9px;gap:0;}
    #restart .rtext{display:none;}
    #restart .ricon{font-size:22px;}
    .site-title{font-size:20px;}
    .site-sub{font-size:12px;}
    /* the (i) pop-ups centre in the viewport so their text never gets cut off */
    #info,.aboutinfo,.terms-pop{position:fixed;left:50%%;right:auto;top:auto;bottom:16px;
         transform:translateX(-50%%);width:92vw;max-width:92vw;}
    #info::before,.aboutinfo::after{display:none;}
  }
</style></head><body>
<div id="app">
  <button id="restart" style="display:none"><span class="ricon">&#10227;</span><span class="rtext"> Restart quiz</span></button>
  <header>
    <div class="site-title">The Great British Dialect Quiz</div>
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
      <label class="consent"><input type="checkbox" id="consent"><span>I agree to the <span class="tlink" id="termsbtn">terms of data collection<span class="terms-pop" id="termspop">By ticking this box, you consent to the collection and storage of your quiz answers and, if you provide it, your hometown. This information is stored <b>anonymously</b>: no name, email address, or IP address is recorded, and it cannot be traced back to you. It is used solely to study regional language variation and to improve future versions of the quiz. Participation is voluntary, and you may request deletion of your data at any time by contacting the Intellectual Forum at Jesus College, Cambridge. Your data will not be sold or shared with third parties.</span></span></span></label>
      <button id="startbtn">Start the quiz &rarr;</button>
      <p class="intro-note"><span class="aboutwrap"><span class="aboutbtn">&#9432;</span><span class="aboutinfo">This is a pixel-art version of the British dialect map. It was made by <b>Alan Levita</b> during a research internship at the Intellectual Forum, drawing on the research of <b>Prof. Bert Vaux</b> of King&rsquo;s College, Cambridge. Bert&rsquo;s work formed the basis for the original <i>New York Times</i> dialect quiz.<br><br>Your answers are used to estimate roughly where you&rsquo;re from. All the maps were redrawn by hand in a pixel-art style, based on isoglosses from published research on British dialects.</span></span> Powered by the Intellectual Forum at Jesus College, Cambridge</p>
    </div>
    </div>
  </div>
  <div id="stage">
    <div id="left">
      <h1 id="qtext"></h1>
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
      <img id="qimg" style="display:none" alt="">
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
// The heat-map surfaces are ~1.9 MB. Kept as a STRING and JSON.parse()d lazily so
// the landing page becomes interactive instantly instead of blocking on a giant
// object-literal parse. (JSON.parse of a string is far faster than the JS engine
// evaluating an equivalent inline object literal, and here it's deferred entirely.)
const GRIDS_JSON=%s; let grids=null;
const land=%s,cities=%s,cg=%s,names=%s;
const PLACES=%s;
const regionGrid=%s,regionNames=%s;
const dialectGrid=%s,dialectColors=%s;
const TOWNS=%s;   // GB towns/cities for the hometown type-ahead
const ICELOLLY_IMG=%s;
const BREAD_IMG=%s;
const QUESTIONS=[
  // first question: where did you grow up? (a GB town type-ahead, or "not from GB").
  // Not a heat-map question — collected (with consent) for future training, not scored.
  {id:"hometown",text:"Where in Great Britain did you grow up?",hometownq:true},
  // metric "pct": a clean binary the paper reports as proportions -> show a percent.
  // ipa:true enables the click-a-city foot-strut IPA readout (foot-strut only).
  {id:"q1",text:"Do <i>foot</i> and <i>cut</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   ipa:true,info:"footstrut",infoLabel:"the foot&ndash;strut split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme"},{label:"No, they sound different",v:0,word:"split"}]},
  {id:"bookspook",text:"Do <i>book</i> and <i>spook</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"bookspook",infoLabel:"book vs spook",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme book/spook"},{label:"No, they sound different",v:0,word:"don&rsquo;t rhyme"}]},
  {id:"stirstare",text:"Do <i>stir</i> and <i>stare</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"nursesquare",infoLabel:"the nurse&ndash;square merger",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme stir/stare"},{label:"No, they sound different",v:0,word:"keep them distinct"}]},
  {id:"northforce",text:"Do <i>horse</i> and <i>hoarse</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"northforce",infoLabel:"the north&ndash;force merger",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (merged)"},{label:"No, they sound different",v:0,word:"keep them distinct"}]},
  {id:"forcecure",text:"Do <i>poor</i> and <i>pour</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"forcecure",infoLabel:"the cure&ndash;force merger",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (merged)"},{label:"No, they sound different",v:0,word:"keep them distinct"}]},
  {id:"trapbath",text:"Do <i>gas</i> and <i>grass</i> rhyme for you?",tag:"blended: BBC Future + English Dialect App",real:true,metric:"pct",
   info:"trapbath",infoLabel:"the trap&ndash;bath split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (short a)"},
         {label:"No, they sound different",v:0,word:"split (long a)"}]},
  {id:"scone",text:"Does <i>scone</i> rhyme with <i>gone</i> or <i>bone</i>?",tag:"real data",real:true,metric:"pct",
   info:"scone",infoLabel:"how you say &lsquo;scone&rsquo;",
   opts:[{label:"Gone (&ldquo;skon&rdquo;)",v:1,word:"rhyme it with gone"},
         {label:"Bone / cone (&ldquo;skohn&rdquo;)",v:0,word:"rhyme it with bone"}]},
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
  {id:"bread",text:"This is called a &#95;&#95;&#95;&#95;",img:BREAD_IMG,tag:"real data (bread-roll survey)",real:true,phon:false,multi:true,metric:"prevalence",
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
   img:ICELOLLY_IMG,tag:"",real:true,metric:"prevalence",info:"lolly",infoLabel:"ice lolly vs lolly ice",
   opts:[
     {label:"Ice lolly",v:"icelolly",term:"ice lolly",grid:"icelolly"},
     {label:"Lolly ice",v:"lollyice",term:"lolly ice",grid:"lollyice"},
     {label:"I use both interchangeably",v:"both",term:"both",grid:"lollyice"},
     {label:"Other term (ice pop, popsicle, etc.)",v:"other",term:"another term",none:true},
     {label:"I have no word for this",v:"none",term:"no word for this",none:true}
   ]},
  {id:"tag",text:"This children&rsquo;s chasing game is called &#95;&#95;&#95;&#95;",tag:"real data (Starkey Comics dialect survey)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"tag",infoLabel:"names for the chasing game",
   opts:[
     {label:"Tag",v:"tag",term:"tag",grid:"tag_tag"},
     {label:"Tick",v:"tick",term:"tick",grid:"tag_tick"},
     {label:"Tip",v:"tip",term:"tip",grid:"tag_tip"},
     {label:"Tig",v:"tig",term:"tig",grid:"tag_tig"},
     {label:"Tiggy",v:"tiggy",term:"tiggy",grid:"tag_tiggy"},
     {label:"Tuggy",v:"tuggy",term:"tuggy",grid:"tag_tuggy"},
     {label:"It",v:"it",term:"it",grid:"tag_it"},
     {label:"Hit",v:"hit",term:"hit",grid:"tag_hit"},
     {label:"Had",v:"had",term:"had",grid:"tag_had"},
     {label:"Touch",v:"touch",term:"touch",grid:"tag_touch"},
     {label:"Dobby",v:"dobby",term:"dobby",grid:"tag_dobby"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_tag",excl:true,none:true}
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
  nursesquare:"<b>The NURSE&ndash;SQUARE merger</b> &mdash; in some accents the vowels of NURSE (<i>stir, fur, her</i>) and SQUARE (<i>stare, fair, hair</i>) fall together, so <i>stir</i> and <i>stare</i> (or <i>fur</i> and <i>fair</i>) rhyme. It is long-established and best-documented in Liverpool / Merseyside and the North West, and is now also strong &mdash; and apparently spreading &mdash; along the east coast (Hull, Teesside). An older East-Midlands merger has largely faded, though north-east Lincolnshire still has it.",
  scone:"<b>scone</b> &mdash; the great teatime shibboleth: does it rhyme with <i>gone</i> (/sk&#594;n/) or with <i>bone</i>/<i>cone</i> (/sko&#650;n/)? Most of Britain &mdash; Scotland and the North especially &mdash; rhymes it with <i>gone</i>. The <i>bone</i> pronunciation is the local norm in the <b>East Midlands</b> (Nottingham, Derby, Leicester), with the far South West leaning that way a little too.",
  northforce:"<b>The NORTH&ndash;FORCE merger</b> &mdash; whether <i>horse</i> and <i>hoarse</i> (or <i>for</i> and <i>four</i>, <i>war</i> and <i>wore</i>) sound identical. Most of England and Wales merged them long ago, so they rhyme; <b>Scotland</b> keeps them clearly distinct, as do pockets around <b>Manchester</b> and Merseyside.",
  forcecure:"<b>The CURE&ndash;FORCE merger</b> &mdash; whether <i>poor</i> and <i>pour</i> (or <i>sure</i> and <i>shore</i>, <i>tour</i> and <i>tore</i>) sound identical. Across most of England they have merged, so they rhyme; the older distinct <i>poor</i>/<i>sure</i> vowel /&#650;&#601;/ survives in <b>Scotland</b>, the <b>North East</b>, and <b>West Yorkshire</b>.",
  trapbath:"<b>The trap&ndash;bath split</b> &mdash; in the 18th century southern English lengthened the <i>a</i> in a set of words (<i>bath, grass, last, dance</i>) to /&#593;&#720;/, splitting them from TRAP words (<i>cat, trap</i>). The North, Wales and Scotland kept the short /a/ &mdash; so a northerner says [ba&#952;], a southerner [b&#593;&#720;&#952;]. It&rsquo;s one of the sharpest north&ndash;south markers.",
  splinter:"<b>Words for a splinter</b> of wood in the skin. <b>Splinter</b> is the standard nationwide; <b>spelk</b> (from Old Norse / Old English <i>spelc</i>) belongs to the North East &amp; the Borders; <b>spell</b> is northern; <b>shiver</b> is East Anglian; <b>sliver</b> is a South East word.",
  giveitme:"<b>&lsquo;Give it me&rsquo;</b> &mdash; the &lsquo;alternative double-object&rsquo; dative: the theme (<i>it</i>) comes before the goal (<i>me</i>) with no <i>to</i> &mdash; <i>give it me</i> rather than <i>give it to me</i> or <i>give me it</i>. It&rsquo;s a North West &amp; Midlands feature (strongest around Manchester and the Potteries), thinning towards the North East and the South.",
  lolly:"<b>Ice lolly vs lolly ice</b> &mdash; <i>ice lolly</i> is the standard British term; <i>lolly ice</i> (the words reversed) is the well-known Merseyside / Liverpool (&lsquo;Scouse&rsquo;) form. Further afield you&rsquo;ll hear <i>ice pop</i> (Ireland, Scotland) or <i>popsicle</i> (North America)."
,
  tag:"<b>Names for tag/it</b> &mdash; <i>tig</i> covers most of England, Scotland &amp; Wales; <i>it</i> is the South East&rsquo;s word instead of <i>tig</i>. Distinct local pockets survive within that: <i>tiggy</i> and <i>tuggy</i> side by side around Durham &amp; North Yorkshire, <i>tick</i> and a tiny <i>tip</i> pocket in North Wales, <i>touch</i> around Birmingham and in the South West, <i>had</i> on the Suffolk/Essex coast, <i>hit</i> on the South Devon coast, and <i>dobby</i> &mdash; a well-known Nottinghamshire/South Yorkshire term &mdash; in a tight pocket around Sheffield."
};
// only etymology sources are cited (the maps are our own recreations, not originals)
const ETYM_SRC="Wiktionary";
const SRC={ splinter:"Wiktionary", tag:"Starkey Comics" };
// lexical prevalence: a relative band, no misleading headcount
function band(v){return v>=0.5?"the main word(s) here":v>=0.3?"common here":v>=0.15?"one of several here":"rarely used here";}
// for the "no word" negative map: high v = the words are absent here
function bandNone(v){return v>=0.5?"few people have a word":v>=0.3?"a word is less usual":v>=0.15?"most people have a word":"nearly everyone has a word";}
let idx=0; const answers={}; const revealedSet=new Set();
const cv=document.getElementById("cv"),cx=cv.getContext("2d");cv.width=W*CELL;cv.height=H_*CELL;
let SHOWN=null;
// national average of each map (over land), so hover can spot LOCALLY distinctive words
const gridMean={};
// parse the big heat-map data + compute means on first use (quiz start), not on page load
function ensureGrids(){
  if(grids) return;
  grids=JSON.parse(GRIDS_JSON);
  for(const k in grids){let s=0,n=0;const g=grids[k];
    for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const v=g[r][c];if(v!=null){s+=v;n++;}}
    gridMean[k]=n?s/n:1e-6;}
}

function clearRight(prompt){
  // no map yet -> hide the whole right panel and let the left frame center on the page
  document.getElementById("right").style.display="none";
  document.getElementById("qimg").style.display="none";
  document.getElementById("rprompt").style.display="";cv.style.display="none";
  document.getElementById("legend").style.display="none";document.getElementById("detail").innerHTML="";
  document.getElementById("match").innerHTML="";
  document.getElementById("infowrap").style.display="none";
  document.getElementById("rtitle").innerHTML=prompt;
}
// photo questions before reveal: show the photo on the right, keep the left compact
function showRightImage(q){
  document.getElementById("right").style.display="";
  const im=document.getElementById("qimg"); im.src=q.img; im.style.display="block";
  document.getElementById("rprompt").style.display="none";cv.style.display="none";
  document.getElementById("legend").style.display="none";document.getElementById("detail").innerHTML="";
  document.getElementById("match").innerHTML="";
  document.getElementById("infowrap").style.display="none";
  document.getElementById("rtitle").innerHTML="";
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
    prog.textContent="Your result";
    qt.style.display="none";box.style.display="none";next.style.display="none";hint.style.display="none";
    ensureGrids();
    const cs=combinedSurface();
    const scored=drawCombined(cs);
    done.style.display="";
    if(!cs.place || cs.count===0){
      done.innerHTML="<b>Answer a few questions to see where your speech fits.</b>";
    } else {
      const top=placeName(cs.place);   // nearest named place to the smoothed peak
      const runners=scored.filter(s=>s.p!==cs.place && s.v>=0.7*scored[0].v).slice(0,2).map(s=>s.p.name);
      done.innerHTML="<div style='font-size:13px;color:#8a857c;letter-spacing:.04em;text-transform:uppercase;margin-bottom:8px'>Based on your "+cs.count+" answers</div>"+
        "<div style='font-size:15px;color:#555;margin-bottom:2px'>You sound most like</div>"+
        "<div style='font-size:30px;font-weight:750;line-height:1.15;color:var(--accent)'>"+top+"</div>"+
        (runners.length?"<div style='font-size:13px;color:#8a857c;margin-top:12px'>Also close: "+runners.join(", ")+"</div>":"")+
        "<div style='font-size:11.5px;color:#8a857c;margin-top:18px;line-height:1.5'>&#9733; marks your closest match. This is a preliminary result &mdash; a combined &ldquo;place me&rdquo; map built from every answer.</div>";
    }
    return;
  }
  const q=QUESTIONS[idx];
  prog.textContent="Question "+(idx+1)+" of "+QUESTIONS.length;
  qt.style.display="";qt.innerHTML=q.text; box.style.display="";box.innerHTML="";done.style.display="none";
  const ans=answers[q.id];
  const contLabel=(idx===QUESTIONS.length-1)?"Finish →":"Continue →";
  // ---- first question: hometown (town type-ahead OR "not from Great Britain") ----
  if(q.hometownq){
    clearRight("&nbsp;");                                  // no map for this question
    hint.style.display="none";
    const prefill=(ans&&ans!=="notgb")?ans:"";
    box.innerHTML=
      "<div class='htq'>"+
        "<p class='htq-note'>This is for our research &mdash; it won&rsquo;t affect your result.</p>"+
        "<div class='combo'><input type='text' id='hometown' autocomplete='off' role='combobox' aria-autocomplete='list' aria-expanded='false' placeholder='Start typing your town&hellip;' value=\\""+prefill+"\\"><ul id='townlist' class='combo-list' role='listbox'></ul></div>"+
        "<div class='or-div'>or</div>"+
        "<button type='button' class='altbtn"+(ans==="notgb"?" chosen":"")+"' id='notgbbtn'>I didn&rsquo;t grow up in Great Britain, but I&rsquo;d still like to play</button>"+
      "</div>";
    attachCombo(document.getElementById("hometown"), document.getElementById("townlist"));
    // toggle: click to choose "not from GB", click again to deselect (was un-deselectable)
    document.getElementById("notgbbtn").onclick=()=>{
      if(answers.hometown==="notgb") delete answers.hometown; else answers.hometown="notgb";
      render();
    };
    next.style.display="block"; next.disabled=false; next.textContent=contLabel;
    next.onclick=()=>{ leaveHometown(); idx++; render(); };
    return;
  }
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
    hint.innerHTML=isRevealed?"":"Select all that apply";
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
    if(q.img) showRightImage(q);           // photo questions: show the photo on the right side
    else clearRight("Choose your answer"+(q.multi?"(s)":"")+", then press &ldquo;See map&rdquo;");
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
  // stretch contrast around the midpoint so the maps read boldly instead of washing out
  t=Math.max(0,Math.min(1, 0.5+(t-0.5)*1.55));
  const blue=[18,86,222],white=[232,232,236],red=[214,16,32];
  const mix=(A,B,k)=>[Math.round(A[0]+(B[0]-A[0])*k),Math.round(A[1]+(B[1]-A[1])*k),Math.round(A[2]+(B[2]-A[2])*k)];
  return t<0.5?mix(blue,white,t/0.5):mix(white,red,(t-0.5)/0.5);}

// Where does this answer concentrate? Mean surface per region; if one/a few
// regions clearly stand out, name them (grouping adjacent ones); if it's
// scattered or nothing stands out, say "multiple regions".
function joinRegions(a){return a.length<=1?a[0]:a.slice(0,-1).join(", ")+" &amp; "+a[a.length-1];}
// score each representative place by the surface value around it, most-representative first
function matchPlaces(surf){
  return PLACES.map(p=>{let s=0,n=0;
    for(let dr=-2;dr<=2;dr++)for(let dc=-2;dc<=2;dc++){const r=p.row+dr,c=p.col+dc;
      if(r<0||r>=H_||c<0||c>=W)continue; const v=surf[r]?surf[r][c]:null; if(v!=null){s+=v;n++;}}
    return {p:p, v:n?s/n:0};}).sort((a,b)=>b.v-a.v);
}
function placeName(p){return p.tag?p.name+" ("+p.tag+")":p.name;}
function matchRegion(surf){
  const sum=regionNames.map(()=>[0,0]);
  let peakVal=-1,peakReg=-1;
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const g=regionGrid[r][c];if(g<0)continue;
    const v=surf[r][c];if(v==null)continue;sum[g][0]+=v;sum[g][1]++;
    if(v>peakVal){peakVal=v;peakReg=g;}}
  const meanA=sum.map(([s,n])=>n?s/n:0);
  const mean={}; regionNames.forEach((n,i)=>mean[n]=meanA[i]);
  const topMean=Math.max.apply(null,meanA);
  const NORTHSET=["Yorkshire","the North West","the North East"];
  const SOUTHSET=["the South East","the South West","East Anglia"];
  const MIDS=["the East Midlands","the West Midlands"];
  // a genuinely strong regional signal -> name the top zone(s). Fine regions collapse into big
  // zones (North/South of England, the Midlands) when >=2 of a set are hot; Scotland & Wales
  // stay named. Zones are ranked by strength, so the strongest are always kept.
  if(topMean>=0.40){
    const hot=regionNames.filter(n=>mean[n]>=0.6*topMean && mean[n]>=0.32);
    const zones=[]; const used=new Set();
    const grp=(set,label)=>{const inh=set.filter(n=>hot.includes(n));
      if(inh.length>=2){ zones.push([label, Math.max.apply(null,inh.map(n=>mean[n]))]); set.forEach(n=>used.add(n)); }};
    grp(NORTHSET,"the North of England"); grp(SOUTHSET,"the South of England"); grp(MIDS,"the Midlands");
    hot.filter(n=>!used.has(n)).forEach(n=>zones.push([n,mean[n]]));
    zones.sort((a,b)=>b[1]-a[1]);
    const nm=zones.map(z=>z[0]);
    if(nm.includes("the North of England") && nm.includes("the South of England")) return "much of Britain";
    return nm.length?joinRegions(nm.slice(0,3)):"several regions";
  }
  // a sharp LOCAL peak in an otherwise-cool region (muffin=Manchester, batch=Coventry)
  if(peakVal>=0.45 && meanA[peakReg]<0.35) return regionNames[peakReg];
  return "several regions";
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

// a filled 5-point star (used to mark the main representative place(s) for the answer)
function drawStar(x,y,r,fill,strokeW){
  cx.beginPath();
  for(let i=0;i<10;i++){const ang=Math.PI/5*i-Math.PI/2;const rad=(i%%2===0)?r:r*0.45;
    const px=x+Math.cos(ang)*rad,py=y+Math.sin(ang)*rad; i===0?cx.moveTo(px,py):cx.lineTo(px,py);}
  cx.closePath();cx.fillStyle=fill;cx.fill();
  if(strokeW){cx.lineWidth=strokeW;cx.strokeStyle="#2b2b2b";cx.stroke();}
}
function drawMap(q,ans){
  document.getElementById("right").style.display="";   // map revealed -> show the right panel
  document.getElementById("qimg").style.display="none";  // hide the photo once the map appears
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
  // prevalence (lexical) maps show ACTUAL prevalence, not a normalized peak: a localized
  // minority word (e.g. shiver, muffin) reads as a small, LIGHT hotspot; a common word reads
  // deep-red and widespread. Value maps to the white->red half of the scale (0 = pale "not
  // used here", 1 = deep red). pct questions keep the full blue-white-red diverging scale.
  const seq = (q.metric!=="pct" && !isSlider);
  const hcol = (v)=> seq ? heat(0.5 + Math.max(0,Math.min(1,v))*0.5) : heat(v);
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
    const [rr,gg,bb]=(incon||v==null)?[214,214,220]:hcol(v);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fillRect(c*CELL,r*CELL,CELL-GAP,CELL-GAP);
  }
  // matchPlaces is used only to NAME the result (e.g. "Liverpool (Scouse)"); the map keeps
  // all its city dots (hover to read them) — no on-map labels.
  const noResult = incon || (isSlider && ans===3);
  const scored = noResult ? [] : matchPlaces(surf);
  const topV = scored.length ? scored[0].v : 0;
  const meanV = scored.length ? scored.reduce((a,s)=>a+s.v,0)/scored.length : 0;
  const near = scored.filter(s=>s.v>=0.6*topV).length;
  const widespread = topV>0.4 && near>=PLACES.length*0.8;   // common almost everywhere -> "much of Britain"
  // stars ONLY for LOCALISED answers: a sharp, isolated peak. Broad phonological splits
  // (foot-cut, book-spook, trap-bath, north-force, force-cure, tea/dinner, give-it-me) never
  // star; that leaves the lexical words + the Scouse nurse-square merger, gated on peakedness.
  const starAllowed = (q.metric!=="pct" && !isSlider) || q.id==="stirstare";
  const localised = starAllowed && topV>0.18 && meanV>0 && (topV/meanV)>=2.4;
  const starPlaces=[];
  if(localised) for(const s of scored){ if(starPlaces.length>=3)break;
    if(s.v<0.8*topV||s.v<=0.2)continue;
    if(starPlaces.some(p=>Math.hypot(p.col-s.p.col,p.row-s.p.row)<9))continue;
    starPlaces.push(s.p);}
  SHOWN.shownPlaces=starPlaces;   // so the stars are hoverable too
  const starKey=new Set(starPlaces.map(p=>p.col+","+p.row));
  // ordinary cities as circles (skip any spot that will get a star)
  for(const ct of cities){ if(starKey.has(ct.col+","+ct.row))continue;
    const v=surf[ct.row|0]?surf[ct.row|0][ct.col|0]:null;
    const [rr,gg,bb]=(incon||v==null)?[200,200,205]:hcol(v);
    cx.beginPath();cx.arc((ct.col+0.5)*CELL,(ct.row+0.5)*CELL,5,0,7);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fill();cx.lineWidth=2;cx.strokeStyle="#2b2b2b";cx.stroke();}
  // representative places as bold, red-shaded stars with a white halo (stand out clearly)
  for(const p of starPlaces){const v=surf[p.row]?surf[p.row][p.col]:null;
    const [rr,gg,bb]=(v==null)?[200,200,205]:hcol(v);
    const x=(p.col+0.5)*CELL,y=(p.row+0.5)*CELL;
    drawStar(x,y,16,"#fff",0);                              // white halo
    drawStar(x,y,12,"rgb("+rr+","+gg+","+bb+")",3);}        // the star
  // legend reflects the scale in use: word maps run pale->red; yes/no maps run blue->red
  {const lg=document.getElementById("legend"); const sp=lg.querySelectorAll("span");
   if(seq){ sp[1].style.background="linear-gradient(to right,rgb(232,232,236),rgb(214,16,32))";
     sp[0].textContent="not used here"; sp[2].textContent="common"; }
   else { sp[1].style.background="linear-gradient(to right,rgb(18,86,222),rgb(232,232,236),rgb(214,16,32))";
     sp[0].textContent="uncommon"; sp[2].textContent="common"; }}
  // no more "closest to" — the STAR is the result. Localised answers name their place(s);
  // broad/widespread answers just describe the pattern (the map shows it).
  let matchHTML;
  if(noResult || !scored.length){
    matchHTML="<span style='color:#8a857c;font-weight:500'>Inconclusive &mdash; this doesn&rsquo;t point to a particular place.</span>";
  } else if(starPlaces.length){
    matchHTML="&#9733; <b>"+starPlaces.map(placeName).join(" &amp; ")+"</b>"+
      "<div style='font-size:11px;color:#8a857c;margin-top:2px;font-weight:400'>the &#9733; marks where this answer stands out</div>";
  } else {
    // broad answers: describe the REGION (e.g. "the North of England"), or "much of Britain"
    const rn=matchRegion(surf);
    matchHTML = (rn==="much of Britain")
      ? "<span style='color:#8a857c;font-weight:500'>Used across much of Britain</span>"
      : "&#9873; most common in <b>"+rn+"</b>";
  }
  document.getElementById("match").innerHTML=matchHTML;
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

// ---- combined "place me" result: the surface for one answer ----
function answerSurface(q,ans){
  if(q.slider){ if(ans===undefined||ans===3)return null; const base=grids[q.grid]; if(!base)return null; const w=(ans-1)/4;
    const s=[];for(let r=0;r<H_;r++){s.push([]);for(let c=0;c<W;c++){const b=base[r]?base[r][c]:null; s[r].push(b==null?null:(w*b+(1-w)*(1-b)));}} return s; }
  if(q.multi){ if(!Array.isArray(ans)||!ans.length)return null;
    const sel=ans.map(v=>q.opts.find(o=>o.v===v)).filter(Boolean); if(!sel.length||(sel.length===1&&sel[0].none))return null;
    const s=[];for(let r=0;r<H_;r++){s.push([]);for(let c=0;c<W;c++){ if(!land[r][c]){s[r].push(null);continue;}
      let m=0,any=false; for(const o of sel){ if(!o.grid||o.none)continue; const gr=grids[o.grid]; const val=gr&&gr[r]?gr[r][c]:null; if(val!=null){any=true; if(val>m)m=val;}} s[r].push(any?m:null);}} return s; }
  if(ans===undefined)return null;
  const opt=q.opts?q.opts.find(o=>o.v===ans):null;
  if(opt&&opt.grid){ const base=grids[opt.grid]; if(!base)return null; const s=[];for(let r=0;r<H_;r++){s.push([]);for(let c=0;c<W;c++)s[r].push(base[r]?base[r][c]:null);} return s; }
  if(grids[q.id]){ const base=grids[q.id]; const s=[];for(let r=0;r<H_;r++){s.push([]);for(let c=0;c<W;c++){const b=base[r]?base[r][c]:null; s[r].push(b==null?null:(ans?b:1-b));}} return s; }
  return null;
}
// a light Gaussian-ish blur over land (repeated 3x3 box passes) — smooths the overlaid maps
function blurLand(surf, passes){
  let s=surf;
  for(let p=0;p<passes;p++){ const o=[];
    for(let r=0;r<H_;r++){o.push([]);for(let c=0;c<W;c++){ if(!land[r][c]){o[r].push(null);continue;}
      let sum=0,n=0; for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){const rr=r+dr,cc=c+dc;
        if(rr<0||rr>=H_||cc<0||cc>=W)continue; const v=s[rr]?s[rr][cc]:null; if(v!=null){sum+=v;n++;}}
      o[r].push(n?sum/n:null);}}
    s=o; }
  return s;
}
// overlay every answered surface (average), Gaussian-smooth, min-max stretch, and find the peak
function combinedSurface(){
  const surfs=[];
  for(const q of QUESTIONS){ if(q.hometownq)continue; const a=answers[q.id]; if(a===undefined)continue; const s=answerSurface(q,a); if(s)surfs.push(s); }
  let comb=[];
  for(let r=0;r<H_;r++){comb.push([]);for(let c=0;c<W;c++){ if(!land[r][c]){comb[r].push(null);continue;}
    let sum=0,n=0; for(const s of surfs){const v=s[r]?s[r][c]:null; if(v!=null){sum+=v;n++;}} comb[r].push(n?sum/n:null);}}
  comb=blurLand(comb,3);   // Gaussian smoothing over the overlaid maps
  let mx=-1e9,mn=1e9,pr=-1,pc=-1;
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const m=comb[r][c]; if(m!=null){if(m>mx){mx=m;pr=r;pc=c;} if(m<mn)mn=m;}}
  if(mx>mn) for(let r=0;r<H_;r++)for(let c=0;c<W;c++){ if(comb[r][c]!=null) comb[r][c]=(comb[r][c]-mn)/(mx-mn); }
  // nearest named place to the geographic peak
  let best=null,bd=1e9; for(const p of PLACES){const d=Math.hypot(p.col-pc,p.row-pr); if(d<bd){bd=d;best=p;}}
  return {surf:comb, count:surfs.length, peak:[pr,pc], place:best};
}
function drawCombined(cs){
  const comb=cs.surf;
  document.getElementById("right").style.display="";
  document.getElementById("qimg").style.display="none";
  document.getElementById("rprompt").style.display="none"; cv.style.display="block";
  document.getElementById("legend").style.display="flex";
  document.getElementById("detail").innerHTML=""; document.getElementById("infowrap").style.display="none";
  document.getElementById("rtitle").innerHTML="<b>Where your speech fits</b>";
  const col=v=>heat(v);   // full blue -> white -> red diverging = least .. most like you
  cx.clearRect(0,0,cv.width,cv.height);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){ if(!land[r][c])continue;
    cx.fillStyle="#c9c9d2";cx.fillRect(c*CELL,r*CELL,CELL,CELL);
    const v=comb[r][c]; const [rr,gg,bb]=(v==null)?[214,214,220]:col(v);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fillRect(c*CELL,r*CELL,CELL-GAP,CELL-GAP);}
  const star=cs.place;   // the named place nearest the smoothed peak
  for(const ct of cities){ if(star&&star.col===ct.col&&star.row===ct.row)continue;
    const v=comb[ct.row|0]?comb[ct.row|0][ct.col|0]:null; const [rr,gg,bb]=(v==null)?[200,200,205]:col(v);
    cx.beginPath();cx.arc((ct.col+0.5)*CELL,(ct.row+0.5)*CELL,5,0,7);cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fill();cx.lineWidth=2;cx.strokeStyle="#2b2b2b";cx.stroke();}
  if(star){const x=(star.col+0.5)*CELL,y=(star.row+0.5)*CELL; drawStar(x,y,17,"#fff",0); drawStar(x,y,13,"rgb(214,16,32)",3);}
  {const lg=document.getElementById("legend"); const sp=lg.querySelectorAll("span");
   sp[1].style.background="linear-gradient(to right,rgb(18,86,222),rgb(232,232,236),rgb(214,16,32))"; sp[0].textContent="least like you"; sp[2].textContent="most like you";}
  SHOWN=null;   // no hover on the summary map (preliminary)
  document.getElementById("match").innerHTML="";
  return matchPlaces(comb);
}

const tip=document.getElementById("tip");
function cvTip(clientX,clientY){
  if(!SHOWN){tip.style.opacity=0;return;}
  const rect=cv.getBoundingClientRect(),sx=cv.width/rect.width,sy=cv.height/rect.height;
  const x=(clientX-rect.left)*sx,y=(clientY-rect.top)*sy;
  let best=null,bd=1e9;for(const ct of [...cities,...(SHOWN.shownPlaces||[])]){const dd=Math.hypot((ct.col+0.5)*CELL-x,(ct.row+0.5)*CELL-y);if(dd<bd){bd=dd;best=ct;}}
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
  else tip.style.opacity=0;
}
cv.addEventListener("mousemove",(e)=>cvTip(e.clientX,e.clientY));
cv.addEventListener("mouseleave",()=>tip.style.opacity=0);
// touch: tap or drag on the map to read a city (phones have no hover)
cv.addEventListener("touchstart",(e)=>{if(e.touches[0])cvTip(e.touches[0].clientX,e.touches[0].clientY);},{passive:true});
cv.addEventListener("touchmove",(e)=>{if(e.touches[0])cvTip(e.touches[0].clientX,e.touches[0].clientY);},{passive:true});

// click a city (foot-strut only) -> expected local IPA
const detail=document.getElementById("detail");
cv.addEventListener("click",(e)=>{
  if(!SHOWN||!SHOWN.q.ipa)return;
  const rect=cv.getBoundingClientRect(),sx=cv.width/rect.width,sy=cv.height/rect.height;
  const x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy;
  let best=null,bd=1e9;for(const ct of [...cities,...(SHOWN.shownPlaces||[])]){const dd=Math.hypot((ct.col+0.5)*CELL-x,(ct.row+0.5)*CELL-y);if(dd<bd){bd=dd;best=ct;}}
  if(best&&bd<=26){const rhymes=(grids.q1[best.row|0][best.col|0]||0)>=0.5;
    const cut=rhymes?"k&#650;t":"k&#652;t";
    detail.innerHTML="<b>"+best.name+"</b> &mdash; "+
      (rhymes?"foot &amp; cut <b>rhyme</b> (both /&#650;/)":"foot &amp; cut are <b>distinct</b> (/&#650;/ vs /&#652;/)")+
      "<br><span class='ipa'>/f&#650;t/ &middot; /"+cut+"/</span>";}
});

// ---- landing page: chunky 8-bit pixel map, same style as the quiz maps ----
function drawMini(){
  const M=5, GAP2=1;   // same chunky pixel cells + subtle grid as the quiz heat maps
  const mc=document.getElementById("introcv"); mc.width=W*M; mc.height=H_*M;
  mc.style.width="300px";   // height is auto (CSS) -> scales proportionally, stays responsive
  const mx=mc.getContext("2d");
  mx.clearRect(0,0,mc.width,mc.height);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){
    const di=dialectGrid[r][c]; if(di<0) continue;
    mx.fillStyle="#c9c9d2"; mx.fillRect(c*M,r*M,M,M);            // grey pixel grid, like the quiz
    const col=dialectColors[di][1];
    mx.fillStyle="rgb("+col[0]+","+col[1]+","+col[2]+")";
    mx.fillRect(c*M,r*M,M-GAP2,M-GAP2);
  }
  // hover/touch -> name the dialect group under the cursor (replaces the old static key)
  const tip=document.getElementById("introtip"), cap=document.getElementById("introcap");
  function miniTip(clientX,clientY){const rect=mc.getBoundingClientRect();
    const c=Math.floor((clientX-rect.left)/(rect.width/W)), r=Math.floor((clientY-rect.top)/(rect.height/H_));
    const di=(r>=0&&r<H_&&c>=0&&c<W)?dialectGrid[r][c]:-1;
    if(di>=0){const [nm,col]=dialectColors[di];
      tip.textContent=nm; tip.style.background="rgb("+col[0]+","+col[1]+","+col[2]+")";
      tip.style.left=(clientX-rect.left)+"px"; tip.style.top=(clientY-rect.top)+"px"; tip.style.opacity=1;
      cap.textContent=nm; cap.style.color="rgb("+col[0]+","+col[1]+","+col[2]+")";
    } else { tip.style.opacity=0; }}
  mc.onmousemove=(e)=>miniTip(e.clientX,e.clientY);
  mc.onmouseleave=()=>{tip.style.opacity=0; cap.textContent="Hover the map to explore the dialect groups"; cap.style.color="";};
  mc.ontouchstart=(e)=>{if(e.touches[0])miniTip(e.touches[0].clientX,e.touches[0].clientY);};
  mc.ontouchmove=(e)=>{if(e.touches[0]){e.preventDefault();miniTip(e.touches[0].clientX,e.touches[0].clientY);}};
}
// ---- hometown: captured for FUTURE model training only, never used for the result ----
// Right now it's just saved in the visitor's own browser (localStorage). To actually
// collect it, wire the fetch() below up to a backend endpoint (Supabase/Formspree/etc.).
let hometown="", consented=false;
// consent is captured on the intro (required to start)
function recordConsent(){
  const cb=document.getElementById("consent");
  consented=!!(cb&&cb.checked);
  try{ localStorage.setItem("gbdq_consent", consented?"1":"0"); }catch(e){}
}
// hometown is captured when leaving the first question; only stored/sent if consented
function leaveHometown(){
  const el=document.getElementById("hometown");
  const v=((el&&el.value)||"").trim();
  if(v) answers.hometown=v;                               // typed/selected a town (else keep "notgb" or unset)
  hometown=(answers.hometown&&answers.hometown!=="notgb")?answers.hometown:"";
  if(consented && hometown){
    try{ localStorage.setItem("gbdq_hometown", hometown); }catch(e){}
    // TODO(backend): with consent given, POST {hometown, answers, ts} to your endpoint here
    // fetch("https://YOUR-ENDPOINT", {method:"POST", headers:{"Content-Type":"application/json"},
    //   body:JSON.stringify({hometown, answers, ts:Date.now()})}).catch(()=>{});
  }
}
function startQuiz(){
  recordConsent();                                       // record agreement before the quiz
  ensureGrids();                                          // make sure the heat-map data is parsed
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
// ---- hometown type-ahead: attached to the input built for the first question ----
function attachCombo(inp, list){
  if(!inp||!list) return;
  let matches=[], active=-1;
  function close(){ list.style.display="none"; list.innerHTML=""; active=-1; inp.setAttribute("aria-expanded","false"); }
  function paint(){ [...list.children].forEach((li,i)=>li.classList.toggle("active",i===active)); }
  function choose(t){ inp.value=t[0]+", "+t[1]; if(answers.hometown==="notgb") delete answers.hometown; close(); }
  function filter(){
    const q=inp.value.trim().toLowerCase();
    if(!q){ close(); return; }
    matches=TOWNS.filter(t=>t[0].toLowerCase().includes(q)||t[1].toLowerCase().includes(q))
                 .sort((a,b)=>(a[0].toLowerCase().startsWith(q)?0:1)-(b[0].toLowerCase().startsWith(q)?0:1))
                 .slice(0,8);
    if(!matches.length){ close(); return; }
    list.innerHTML=matches.map((t,i)=>"<li class='combo-item' role='option' data-i='"+i+"'><b>"+t[0]+"</b><span class='cc'>, "+t[1]+"</span></li>").join("");
    list.style.display="block"; inp.setAttribute("aria-expanded","true"); active=-1;
  }
  inp.addEventListener("input", ()=>{
    // typing a town clears a previously-chosen "not from GB"
    if(answers.hometown==="notgb"){ delete answers.hometown; const b=document.getElementById("notgbbtn"); if(b)b.classList.remove("chosen"); }
    filter();
  });
  inp.addEventListener("keydown",(e)=>{
    if(list.style.display!=="block") return;
    if(e.key==="ArrowDown"){e.preventDefault();active=Math.min(active+1,matches.length-1);paint();}
    else if(e.key==="ArrowUp"){e.preventDefault();active=Math.max(active-1,0);paint();}
    else if(e.key==="Enter"&&active>=0){e.preventDefault();choose(matches[active]);}
    else if(e.key==="Escape"){close();}
  });
  // mousedown (not click) so it fires before the input blurs; also covers touch taps
  list.addEventListener("mousedown",(e)=>{const li=e.target.closest(".combo-item"); if(li){e.preventDefault();choose(matches[+li.dataset.i]);}});
  inp.addEventListener("blur",()=>setTimeout(close,150));
}
// touch: tap the (i) to open its popup (phones have no hover); tap elsewhere to close
document.getElementById("infobtn").addEventListener("click",function(e){e.stopPropagation();document.getElementById("infowrap").classList.toggle("open");});
var _ab=document.querySelector(".aboutbtn");
if(_ab)_ab.addEventListener("click",function(e){e.stopPropagation();this.closest(".aboutwrap").classList.toggle("open");});
// consent is REQUIRED: the Start button stays disabled until the box is ticked
var _cb=document.getElementById("consent"), _sb=document.getElementById("startbtn");
function syncStart(){ if(_sb) _sb.disabled=!(_cb&&_cb.checked); }
if(_cb) _cb.addEventListener("change", syncStart);
syncStart();
// terms-of-data-collection link: tap to open its popup, without toggling the checkbox
var _tb=document.getElementById("termsbtn");
if(_tb)_tb.addEventListener("click",function(e){e.preventDefault();e.stopPropagation();this.classList.toggle("open");});
document.addEventListener("click",function(){document.getElementById("infowrap").classList.remove("open");var aw=document.querySelector(".aboutwrap");if(aw)aw.classList.remove("open");if(_tb)_tb.classList.remove("open");});
drawMini();
showIntro();
// landing is now interactive; parse the heavy heat-map data in the background so it's
// ready by the time the user taps "Start" (startQuiz also calls this, just in case)
setTimeout(ensureGrids, 60);
</script></body></html>""" % (
    W, H, json.dumps(json.dumps(grids_all)), json.dumps(landj),
    json.dumps(cities), json.dumps(cg.tolist()), json.dumps(names), json.dumps(places),
    json.dumps(region_grid), json.dumps(region_names),
    json.dumps(dialect_grid), json.dumps(dialect_colors),
    json.dumps(GB_TOWNS),
    json.dumps(icelolly_uri), json.dumps(bread_uri),
)

with open("index.html", "w") as f:
    f.write(html)
print("wrote index.html — done. Now just: git add index.html build_quiz.py && git commit -m 'update' && git push")
