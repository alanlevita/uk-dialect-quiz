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
# CC0 (Wikimedia Commons, from Unsplash) -- public-domain dedication, so there
# is no attribution obligation on the page, unlike the bread and lolly photos
gum_uri = _img_uri("gum-pic.jpg", "image/jpeg")

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
    # a strongly NE-Scotland (Doric) word, peaking on the Aberdeenshire coast and
    # fading south through Angus/Perthshire; negligible anywhere south of the Highlands
    "softie": mk(0, [(NSCOT, 4)],
                 {"Aberdeenshire": 78, "Banffshire": 58, "Kincardineshire": 52,
                  "Morayshire": 30, "Angus": 22, "Perthshire": 10}),
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
# skelf: the Scots term, absent from this question until now -- Scotland had no
# option of its own here. Dense across the Central Belt, strong through the
# Highlands and Islands, thinning to mixed usage around Aberdeen and yellow
# (splinter) in the far north. Read off the Tweetolectology survey map.
SKELF = mk(1, None,
           {"Lanarkshire": 60, "Renfrewshire": 58, "Dunbartonshire": 58,
            "Midlothian": 55, "West Lothian": 55, "East Lothian": 50,
            "Stirlingshire": 55, "Clackmannanshire": 55, "Argyllshire": 55,
            "Buteshire": 55, "Ayrshire": 50, "Inverness-shire": 50,
            "Ross-shire": 50, "Cromartyshire": 48, "Sutherland": 45,
            "Fife": 45, "Perthshire": 45, "Kinross-shire": 45,
            "Morayshire": 40, "Nairnshire": 40, "Angus": 35,
            "Peeblesshire": 42, "Selkirkshire": 40, "Roxburghshire": 38,
            "Berwickshire": 38, "Dumfriesshire": 40, "Kirkcudbrightshire": 40,
            "Wigtownshire": 40, "Aberdeenshire": 25, "Banffshire": 25,
            "Kincardineshire": 25, "Caithness": 15, "Orkney": 12,
            "Northumberland": 5, "Cumberland": 5, "Westmorland": 3})

SPLINTER = mk(66, None,
    {"Northumberland": 28, "Durham": 32, "Cumberland": 46, "Westmorland": 50,
     "Norfolk": 52, "Suffolk": 52, "Lancashire": 58, "Yorkshire": 60})

# ---- evening meal: tea / dinner / supper -------------------------------------
# Rebuilt from Our Dialects (MacKenzie, Bailey & Turton, CC BY-SA 4.0), n=7,732,
# replacing a surface decoded off a two-colour picture. Doing it from the raw
# responses is what makes SUPPER possible: it is only 4%% nationally, so it never
# appeared on a tea-vs-dinner map at all, but it is a real southern minority --
# 23%% in Oxfordshire, 13%% in Sussex, 11-12%% through Hampshire and Somerset,
# against 1-2%% across Yorkshire, Lancashire and Cheshire.
# Counties need 30+ respondents; Scotland (n=152) is too thin to split, so every
# Scottish county takes the Scotland-wide figure. Left per-county, Lanarkshire's
# ~35 respondents were producing a 91%% dinner hotspot over Glasgow.
TEA = mk(52, None, {
    "Westmorland": 80, "Yorkshire": 77, "Northumberland": 76, "Lancashire": 73,
    "Cheshire": 72, "Staffordshire": 71, "Caernarfonshire": 70, "Durham": 68,
    "Worcestershire": 56, "Cumberland": 56, "Shropshire": 52, "Derbyshire": 51,
    "Glamorgan": 48, "Carmarthenshire": 45, "Somerset": 42, "Suffolk": 36,
    "Cambridgeshire": 35, "Warwickshire": 34, "Devon": 33, "Gloucestershire": 33,
    "Nottinghamshire": 32, "Norfolk": 31, "Aberdeenshire": 29, "Angus": 29,
    "Argyllshire": 29, "Ayrshire": 29, "Banffshire": 29, "Berwickshire": 29,
    "Buteshire": 29, "Caithness": 29, "Clackmannanshire": 29, "Cromartyshire": 29,
    "Dumfriesshire": 29, "Dunbartonshire": 29, "East Lothian": 29, "Fife": 29,
    "Inverness-shire": 29, "Kincardineshire": 29, "Kinross-shire": 29,
    "Kirkcudbrightshire": 29, "Lanarkshire": 29, "Midlothian": 29, "Morayshire": 29,
    "Nairnshire": 29, "Orkney": 29, "Peeblesshire": 29, "Perthshire": 29,
    "Renfrewshire": 29, "Ross-shire": 29, "Roxburghshire": 29, "Selkirkshire": 29,
    "Shetland": 29, "Stirlingshire": 29, "Sutherland": 29, "West Lothian": 29,
    "Wigtownshire": 29, "Oxfordshire": 28, "Hampshire": 21, "Wiltshire": 21,
    "Northamptonshire": 20, "Leicestershire": 19, "Berkshire": 18, "Buckinghamshire": 17,
    "Dorset": 15, "Sussex": 13, "Kent": 11, "Hertfordshire": 11, "Surrey": 10, "Essex": 9,
    "Middlesex": 4
})

DINNER = mk(44, None, {
    "Essex": 89, "Middlesex": 88, "Hertfordshire": 88, "Kent": 86, "Surrey": 82,
    "Dorset": 79, "Buckinghamshire": 78, "Northamptonshire": 77, "Leicestershire": 77,
    "Sussex": 74, "Berkshire": 74, "Wiltshire": 70, "Hampshire": 68, "Nottinghamshire": 66,
    "Aberdeenshire": 65, "Angus": 65, "Argyllshire": 65, "Ayrshire": 65, "Banffshire": 65,
    "Berwickshire": 65, "Buteshire": 65, "Caithness": 65, "Clackmannanshire": 65,
    "Cromartyshire": 65, "Dumfriesshire": 65, "Dunbartonshire": 65, "East Lothian": 65,
    "Fife": 65, "Inverness-shire": 65, "Kincardineshire": 65, "Kinross-shire": 65,
    "Kirkcudbrightshire": 65, "Lanarkshire": 65, "Midlothian": 65, "Morayshire": 65,
    "Nairnshire": 65, "Orkney": 65, "Peeblesshire": 65, "Perthshire": 65,
    "Renfrewshire": 65, "Ross-shire": 65, "Roxburghshire": 65, "Selkirkshire": 65,
    "Shetland": 65, "Stirlingshire": 65, "Sutherland": 65, "West Lothian": 65,
    "Wigtownshire": 65, "Warwickshire": 62, "Devon": 61, "Norfolk": 61,
    "Gloucestershire": 59, "Suffolk": 59, "Cambridgeshire": 58, "Glamorgan": 51,
    "Oxfordshire": 49, "Somerset": 46, "Shropshire": 46, "Derbyshire": 46,
    "Carmarthenshire": 45, "Worcestershire": 43, "Cumberland": 36, "Durham": 30,
    "Cheshire": 27, "Staffordshire": 26, "Caernarfonshire": 26, "Lancashire": 26,
    "Yorkshire": 22, "Northumberland": 21, "Westmorland": 12
})

SUPPER = mk(4, None, {
    "Oxfordshire": 23, "Sussex": 13, "Somerset": 12, "Hampshire": 11, "Wiltshire": 9,
    "Berkshire": 9, "Carmarthenshire": 9, "Surrey": 8, "Middlesex": 8,
    "Gloucestershire": 8, "Cambridgeshire": 8, "Norfolk": 8, "Westmorland": 8,
    "Cumberland": 8, "Devon": 6, "Aberdeenshire": 6, "Angus": 6, "Argyllshire": 6,
    "Ayrshire": 6, "Banffshire": 6, "Berwickshire": 6, "Buteshire": 6, "Caithness": 6,
    "Clackmannanshire": 6, "Cromartyshire": 6, "Dumfriesshire": 6, "Dunbartonshire": 6,
    "East Lothian": 6, "Fife": 6, "Inverness-shire": 6, "Kincardineshire": 6,
    "Kinross-shire": 6, "Kirkcudbrightshire": 6, "Lanarkshire": 6, "Midlothian": 6,
    "Morayshire": 6, "Nairnshire": 6, "Orkney": 6, "Peeblesshire": 6, "Perthshire": 6,
    "Renfrewshire": 6, "Ross-shire": 6, "Roxburghshire": 6, "Selkirkshire": 6,
    "Shetland": 6, "Stirlingshire": 6, "Sutherland": 6, "West Lothian": 6,
    "Wigtownshire": 6, "Dorset": 5, "Buckinghamshire": 5, "Suffolk": 5,
    "Caernarfonshire": 5, "Warwickshire": 4, "Leicestershire": 4, "Kent": 3,
    "Northamptonshire": 3, "Staffordshire": 3, "Derbyshire": 3, "Northumberland": 3,
    "Glamorgan": 2, "Essex": 2, "Hertfordshire": 2, "Shropshire": 2, "Nottinghamshire": 2,
    "Lancashire": 2, "Durham": 2, "Worcestershire": 1, "Cheshire": 1, "Yorkshire": 1
})

# ---- sofa / settee / couch ---------------------------------------------------
# Our Dialects (MacKenzie, Bailey & Turton, CC BY-SA 4.0), n=6,302.
# Nationally sofa 58%%, settee 25%%, couch 17%%. The county values below carry the
# broad structure, but Lancashire is where this variable turns over fastest and
# the county figure is useless on its own: Blackburn is 57%% settee / 7%% couch
# while Wigan, thirty miles away, is 7%% settee / 76%% couch. Both are "Lancashire
# 28%% settee". The blobs put those cities back.
# Couch is also strong in Scotland (41%%, n=131), which the county values cover.
SETTEE = mk(25, None, {
    "Northumberland": 42, "Nottinghamshire": 38, "Glamorgan": 36, "Yorkshire": 36,
    "Durham": 35, "Warwickshire": 34, "Westmorland": 34, "Staffordshire": 33,
    "Northamptonshire": 31, "Caernarfonshire": 29, "Derbyshire": 28, "Lancashire": 28,
    "Shropshire": 27, "Cheshire": 27, "Worcestershire": 22, "Gloucestershire": 22,
    "Leicestershire": 21, "Buckinghamshire": 17, "Cumberland": 15, "Suffolk": 15,
    "Berkshire": 14, "Hampshire": 14, "Essex": 14, "Somerset": 13, "Hertfordshire": 13,
    "Norfolk": 13, "Devon": 12, "Sussex": 12, "Kent": 12, "Aberdeenshire": 12, "Angus": 12,
    "Argyllshire": 12, "Ayrshire": 12, "Banffshire": 12, "Berwickshire": 12,
    "Buteshire": 12, "Caithness": 12, "Clackmannanshire": 12, "Cromartyshire": 12,
    "Dumfriesshire": 12, "Dunbartonshire": 12, "East Lothian": 12, "Fife": 12,
    "Inverness-shire": 12, "Kincardineshire": 12, "Kinross-shire": 12,
    "Kirkcudbrightshire": 12, "Lanarkshire": 12, "Midlothian": 12, "Morayshire": 12,
    "Nairnshire": 12, "Orkney": 12, "Peeblesshire": 12, "Perthshire": 12,
    "Renfrewshire": 12, "Ross-shire": 12, "Roxburghshire": 12, "Selkirkshire": 12,
    "Shetland": 12, "Stirlingshire": 12, "Sutherland": 12, "West Lothian": 12,
    "Wigtownshire": 12, "Middlesex": 11, "Cambridgeshire": 10, "Oxfordshire": 9,
    "Dorset": 8, "Surrey": 8, "Wiltshire": 6
})

COUCH = mk(17, None, {
    "Aberdeenshire": 41, "Angus": 41, "Argyllshire": 41, "Ayrshire": 41, "Banffshire": 41,
    "Berwickshire": 41, "Buteshire": 41, "Caithness": 41, "Clackmannanshire": 41,
    "Cromartyshire": 41, "Dumfriesshire": 41, "Dunbartonshire": 41, "East Lothian": 41,
    "Fife": 41, "Inverness-shire": 41, "Kincardineshire": 41, "Kinross-shire": 41,
    "Kirkcudbrightshire": 41, "Lanarkshire": 41, "Midlothian": 41, "Morayshire": 41,
    "Nairnshire": 41, "Orkney": 41, "Peeblesshire": 41, "Perthshire": 41,
    "Renfrewshire": 41, "Ross-shire": 41, "Roxburghshire": 41, "Selkirkshire": 41,
    "Shetland": 41, "Stirlingshire": 41, "Sutherland": 41, "West Lothian": 41,
    "Wigtownshire": 41, "Lancashire": 35, "Durham": 27, "Cheshire": 20,
    "Caernarfonshire": 18, "Westmorland": 15, "Sussex": 10, "Staffordshire": 8,
    "Surrey": 8, "Middlesex": 8, "Warwickshire": 7, "Devon": 7, "Hertfordshire": 7,
    "Hampshire": 7, "Oxfordshire": 7, "Yorkshire": 7, "Norfolk": 7, "Northamptonshire": 6,
    "Glamorgan": 6, "Dorset": 6, "Kent": 6, "Buckinghamshire": 6, "Cumberland": 6,
    "Somerset": 5, "Gloucestershire": 5, "Nottinghamshire": 5, "Suffolk": 5,
    "Shropshire": 4, "Derbyshire": 4, "Wiltshire": 4, "Worcestershire": 3, "Essex": 3,
    "Northumberland": 3, "Berkshire": 1, "Leicestershire": 1, "Cambridgeshire": 0
})

SOFA = mk(58, None, {
    "Wiltshire": 90, "Cambridgeshire": 90, "Dorset": 86, "Surrey": 84, "Berkshire": 84,
    "Oxfordshire": 84, "Essex": 83, "Somerset": 82, "Kent": 82, "Devon": 81,
    "Middlesex": 81, "Hertfordshire": 80, "Hampshire": 80, "Norfolk": 80, "Suffolk": 80,
    "Cumberland": 79, "Sussex": 78, "Leicestershire": 78, "Buckinghamshire": 77,
    "Worcestershire": 75, "Gloucestershire": 73, "Shropshire": 70, "Derbyshire": 68,
    "Northamptonshire": 62, "Warwickshire": 59, "Staffordshire": 59, "Glamorgan": 58,
    "Yorkshire": 57, "Nottinghamshire": 57, "Northumberland": 55, "Caernarfonshire": 53,
    "Cheshire": 53, "Westmorland": 51, "Aberdeenshire": 47, "Angus": 47, "Argyllshire": 47,
    "Ayrshire": 47, "Banffshire": 47, "Berwickshire": 47, "Buteshire": 47, "Caithness": 47,
    "Clackmannanshire": 47, "Cromartyshire": 47, "Dumfriesshire": 47, "Dunbartonshire": 47,
    "East Lothian": 47, "Fife": 47, "Inverness-shire": 47, "Kincardineshire": 47,
    "Kinross-shire": 47, "Kirkcudbrightshire": 47, "Lanarkshire": 47, "Midlothian": 47,
    "Morayshire": 47, "Nairnshire": 47, "Orkney": 47, "Peeblesshire": 47, "Perthshire": 47,
    "Renfrewshire": 47, "Ross-shire": 47, "Roxburghshire": 47, "Selkirkshire": 47,
    "Shetland": 47, "Stirlingshire": 47, "Sutherland": 47, "West Lothian": 47,
    "Wigtownshire": 47, "Durham": 38, "Lancashire": 37
})

# measured city rates the county grid cannot express (col, row, peak%%)
SOFA_BLOBS = {
    "couch":  [(53.8, 88.9, 76), (51.0, 91.1, 66), (55.5, 88.4, 45), (54.1, 91.4, 39),
               (51.7, 94.5, 39), (50.4, 84.7, 34), (57.0, 90.0, 28)],   # WN,L,BL,WA,CH,FY,M
    "settee": [(55.1, 85.8, 57), (63.2, 91.6, 54)],                     # Blackburn, Sheffield
}
# where a county value over-claims for a city that does not share it
SOFA_CUTS = {
    "settee": [(53.8, 88.9, 22), (51.0, 91.1, 16)],                     # Wigan 7%%, Liverpool 14%%
}

# ---- chewing gum -------------------------------------------------------------
# Our Dialects (MacKenzie, Bailey & Turton, CC BY-SA 4.0), n=3,524. "Chewing gum"
# is the national default at 79%%; the interest is in three tight local terms.
# Built as blobs at each city rather than county values, because the county step
# destroys this variable outright: Liverpool's 78%% chewy averaged with
# Manchester's 8%% gives a Lancashire figure of 30%%, which describes neither.
# Only postcode areas with n>=40 are used. Sunderland, Darlington and Teesside
# all read high for chewy on samples of 13-26, but Durham (0%%) and Newcastle
# (10%%) sit between them, so that north-eastern signal is incoherent at those
# sample sizes and is left out. Merseyside is the solid cluster: n=150 at
# Liverpool with a clean gradient out through Warrington, Wigan and Chester.
GUM_BLOBS = {
    "chewy":  [(51.0, 91.1, 78), (54.1, 91.4, 57), (53.8, 88.9, 53), (51.7, 94.5, 43),
               (53.3, 85.6, 18), (58.0, 89.1, 17)],          # L, WA, WN, CH, PR, OL
    "chuddy": [(57.6, 91.1, 25), (63.2, 91.6, 22), (62.6, 85.0, 21), (55.5, 88.4, 21),
               (55.1, 85.8, 17), (58.0, 89.1, 15), (57.0, 90.0, 14)],  # SK,S,LS,BL,BB,OL,M
    "chud":   [(62.1, 66.6, 17)],                             # Newcastle only
}

# ---- alleyway ---------------------------------------------------------------
# Straight from the Our Dialects survey (MacKenzie, Bailey & Turton -- the same
# team as the 2022 dialect atlas already cited here), CC BY-SA 4.0. Their map
# embeds RESPONDENT-LEVEL records, so these are counted percentages from 2,087
# answers rather than colours read off a picture.
# Counties with fewer than 20 respondents are left at the base rate and anything
# under 8%% is treated as sampling noise -- with n=9 for Dorset, one person saying
# "snicket" becomes 11%% and would paint a phantom pocket across the south coast.
# The county grid is too coarse for the best part of this variable, though: the
# sharpest divide in the whole dataset is Bradford (snicket 84%%) against Leeds
# (ginnel 66%%), about ten miles apart and both inside "Yorkshire". Those, plus
# Sheffield's gennel and Liverpool's entry, are added as point blobs below.
ALLEY = {
    # "Lancashire 46%%" was an average across postcode areas that disagree wildly
    # -- Blackburn 88, Oldham 84, Bolton 73, Manchester 44, Liverpool 4. A flat
    # county value described none of them, and painted a plateau around
    # Manchester that outranked Leeds (66%%) on neighbourhood mean even though
    # Manchester measures 44%%. The measured towns are blobs below instead.
    "ginnel":  mk(1, None, {"Lancashire": 22, "Cheshire": 39, "Yorkshire": 30,
                            "Westmorland": 30, "Derbyshire": 10}),
    "snicket": mk(1, None, {"Yorkshire": 24}),
    "gennel":  mk(0, None, {"Yorkshire": 4}),
    # 84%% is the DE postcode figure, i.e. Derby itself -- carried by the blob.
    # Historic Derbyshire stretches north to Glossop and Buxton, where the word
    # is not used (Chesterfield answers gennel/ginnel), so the county-wide value
    # has to be much lower than the city rate.
    "jitty":   mk(0, None, {"Derbyshire": 32, "Nottinghamshire": 26,
                            "Leicestershire": 18, "Northamptonshire": 11}),
    "entry":   mk(1, None, {"Warwickshire": 12, "Staffordshire": 10,
                            "Lancashire": 9, "Derbyshire": 8, "Worcestershire": 10}),
    "cut":     mk(1, None, {"Northumberland": 56, "Durham": 22}),
    "passage": mk(2, None, {"Warwickshire": 9, "Shropshire": 12, "Gloucestershire": 20}),
}
# city peaks the county grid cannot express (col, row, peak%%)
ALLEY_BLOBS = {
    "snicket": [(61.0, 85.0, 84), (67.0, 83.0, 41)],   # Bradford, York
    # Leeds, Manchester + the east-Lancashire/West-Yorkshire towns where the
    # word actually peaks. Rates are the survey's own per-postcode figures.
    "ginnel":  [(63.0, 85.0, 66), (57.0, 90.0, 44), (55.1, 85.8, 88),
                (58.0, 89.1, 84), (55.5, 88.4, 73), (53.8, 88.9, 50),
                (53.3, 85.6, 42), (63.0, 86.9, 79)],
    "gennel":  [(64.0, 92.0, 44)],                     # Sheffield
    "entry":   [(51.0, 91.0, 35)],                     # Liverpool
    # Derby is 52.92N 1.48W -> col 63.2, row 98.7. This blob was at (60,101),
    # which is 52.78N 1.87W -- Staffordshire, near Burton, three cells west and
    # two south of the city whose 84%% it was carrying.
    "jitty":   [(63.2, 98.7, 84)],                     # Derby
    "cut":     [(63.0, 68.0, 56)],                     # Newcastle
}

# ---- knocking on a door and running away --------------------------------------
# TWO sources, each used where it is strong.
#
# England and Wales come from Our Dialects (CC BY-SA 4.0), respondent-level:
# 1,469 answers, counted rather than read off a picture. Nationally run family
# 75.8%%, ginger family 17.9%%, cherry knocking 3.1%%, ding dong ditch 2.0%%,
# nicky nicky nine doors 1.0%%, chicky melly 0.1%%.
#
# SCOTLAND comes from YouGov (12 Feb 2025, n>12,000), because Our Dialects has
# seven Scottish respondents in the whole country -- not enough to draw, and
# leaving it blank made a quarter of the map dead. YouGov found Scotland has its
# own word entirely: "chap door run" 27%% and the clipped "chappie" 23%%, i.e.
# half of Scots, plus "chicken mellie" at 7%% (which appears nowhere else in the
# UK -- and which Our Dialects independently caught twice as "chicky melly").
#
# Where the two overlap they agree, which is the reason to trust the join: the
# North East is 56%% "knocky nine doors" in YouGov against 60%% for the same word
# in Our Dialects, and the South West is 13%% cherry knocking regionally against
# a tight 68%% in the Gloucestershire cluster inside it. YouGov lists each name
# separately where Our Dialects groups families, so its per-name shares are
# systematically the lower of the two; the two scales are never mixed inside one
# surface.
#
# EVERY name is its own answer in the quiz. Some of them necessarily SHARE a
# surface, because the evidence for them is the same evidence:
#   * knock a door run / knock and run / knock knock run -- Our Dialects, the only
#     source with county-level geography, records them as one "run family"; YouGov
#     separates the names (21/13/2 nationally) but places all of them in the same
#     Midlands-Yorkshire-North West block, so there is no split to draw.
#   * chap door run / chappie / chicken mellie -- all three are Scotland-wide in
#     YouGov and it publishes no sub-Scottish breakdown.
#   * bobby knocking / rat a tat ginger -- both 11%% of Wales, both Wales-only.
# The map panel dedupes by grid, so a shared surface draws once. Splitting these
# further would mean inventing the difference.
#
# YouGov national shares (Feb 2025): knock down ginger 25, knock a door run 21,
# knock and run 13, ding dong ditch 6, knock knock ginger 4, knocking nine doors 3,
# chap door run 3, cherry knocking 3, knock knock run 2, chappie 2, others 18.
#
# Thresholds, so a handful of people don't paint a region:
#   * counties with n >= 12 carry their measured rate;
#   * counties with n < 12 are left at the base rate and filled by smoothing --
#     EXCEPT where one variant is an outright majority and agrees with the
#     measured counties next door (Sussex 75%% ginger n=8, between Kent 84%% and
#     Surrey 89%%; Devon 62%% n=8, between Somerset 64%% and Wiltshire 92%%;
#     Bedfordshire 57%% n=7, between Hertfordshire 84%% and Buckinghamshire 50%%).
#     Without those three the South West and the Chilterns read as false holes.
#   * chicky melly is dropped entirely: n=2 nationally, and both respondents are
#     outside the Wearside area the word actually belongs to. Two people is not
#     a map.
#   * ding dong ditch is an option in the quiz but has NO surface -- it is an
#     Americanism scattered evenly at 1-4%% everywhere (Lancashire 1%%, Middlesex
#     3%%, Yorkshire 1%%) with no regional home to draw.
#   * the Scottish values below are the two YouGov figures that are stated
#     (chap door run 27 + chappie 23 = 50) plus an allocation of the unstated
#     other half: run 20, ginger 8, and the rest to chicken mellie (7, no option
#     of its own) and YouGov's long tail. The 50 is measured; the split of the
#     remainder is an estimate, and is flagged as one in the (i) panel.
PRANK = {
    # Scotland's own words. YouGov gives each of the three separately and at the
    # same scale, so unlike the run names these get one surface EACH at their own
    # measured rate -- pointing all three at a combined 50%% would have told a
    # chicken-mellie speaker their word is used by half of Scotland.
    "chap":     mk(0, [(SCOT, 27)], {}),
    "chappie":  mk(0, [(SCOT, 23)], {}),
    "mellie":   mk(0, [(SCOT, 7)], {}),
    "ginger": mk(2, [(SCOT, 8)], {"Wiltshire": 92, "Surrey": 89, "Middlesex": 84,
                           "Hertfordshire": 84, "Kent": 84, "Essex": 76,
                           "Somerset": 64, "Buckinghamshire": 50,
                           "Gloucestershire": 26, "Hampshire": 25,
                           "Leicestershire": 8, "Northamptonshire": 6,
                           "Yorkshire": 4, "Shropshire": 4, "Cheshire": 3,
                           "Derbyshire": 3, "Staffordshire": 2, "Lancashire": 1,
                           "Nottinghamshire": 0, "Warwickshire": 0,
                           "Worcestershire": 0, "Denbighshire": 0,
                           "Sussex": 75, "Devon": 62, "Bedfordshire": 57}),
    # Our Dialects call this one "clustered in a southern region of the Midlands".
    # The respondents put it more precisely than the county grid can: 14 of the 46
    # sit between Gloucester, Cheltenham and Stroud, and four more around
    # Northampton. Both are blobs below.
    "cherry": mk(0, [(SCOT, 0)], {"Gloucestershire": 68, "Northamptonshire": 24,
                           "Somerset": 14, "Buckinghamshire": 12,
                           "Nottinghamshire": 8, "Leicestershire": 8,
                           "Worcestershire": 6, "Kent": 4, "Warwickshire": 4,
                           "Hampshire": 3, "Derbyshire": 3, "Staffordshire": 2,
                           "Yorkshire": 1, "Cheshire": 1}),
    # entirely a blob: Northumberland (n=11) and Durham (n=7) are both under the
    # threshold, but 11 of the 15 respondents nationally sit in one tight cluster
    # from Newcastle down to Darlington, which is a stronger statement than either
    # county count on its own.
    "nicky":  mk(0, [(SCOT, 0)], {}),
    # YouGov singles this out as the West Country's alternative to knock down
    # ginger ("unlike the rest of the South, many in the West Country use the
    # alternative"): 16%% in the South West and 16%% in Wales, against 4%% nationally.
    "kkginger": mk(1, [(SW_ENG, 16), (WALES, 16)], {}),
    # The two names YouGov found ONLY in Wales, both at 11%% of Welsh respondents.
    # Weighted to the south: north-east Wales patterns with Cheshire and Lancashire
    # (Denbighshire and Flintshire are 100%% run family in Our Dialects), so the
    # Welsh-specific words belong to the rest of the country, not the Marches.
    "wales":  mk(0, [(WALES, 8), (SWALES, 13)], {}),
    "run":    mk(76, [(SCOT, 20)], {"Denbighshire": 100, "Lancashire": 97, "Shropshire": 96,
                            "Cheshire": 95, "Staffordshire": 95, "Worcestershire": 94,
                            "Yorkshire": 93, "Derbyshire": 92, "Nottinghamshire": 92,
                            "Warwickshire": 86, "Leicestershire": 83,
                            "Northamptonshire": 71, "Hampshire": 69,
                            "Buckinghamshire": 31, "Somerset": 21, "Essex": 20,
                            "Hertfordshire": 16, "Middlesex": 11, "Surrey": 11,
                            "Kent": 8, "Wiltshire": 8, "Gloucestershire": 0,
                            "Sussex": 25, "Devon": 12, "Bedfordshire": 14}),
}
PRANK_BLOBS = {
    "cherry": [(57.5, 114.8, 92), (68.0, 109.4, 43)],  # Gloucester/Cheltenham, Northampton
    "nicky":  [(62.5, 68.0, 60)],                      # Tyneside/Wearside
    # Bristol is inside historic Gloucestershire but does not share its word: the
    # eleven respondents within 25km are 82%% ginger and 0%% cherry, while the
    # cherry cluster sits 30-50km north. Same shape as Leeds vs Bradford.
    "ginger": [(54.0, 120.0, 80)],                     # Bristol
}
# ...and the matching subtractions, where a county's headline word smooths into a
# place that measured otherwise: cherry into Bristol (9%%), run into Tyneside
# (Northumberland 18%%, Durham 29%%, against a base of 76%%).
PRANK_CUTS = {"cherry": [(54.0, 120.0, 30)], "run": [(62.5, 68.0, 58)]}

# ---- "your pants are on backwards": pants or trousers? -------------------------
# Our Dialects, n=6,291 -- the biggest lexical sample in the set, and unusually
# clean: a North West core that falls away fast in every direction. Lancashire
# 55%%, Cheshire 39%%, then Yorkshire 14%% and Staffordshire 2%%. Everything from
# the Midlands south sits at 0-7%%, and Scotland (n=135) at 3%%.
# The county grid understates the cities, because historic Lancashire runs from
# Liverpool to the Lakes and the word peaks in the conurbation: Liverpool measures
# 58%%, Manchester 50%%, against a county-wide 55%% that includes the rural north.
# The pair worth having is Manchester (50%%) against Stoke (3%%) -- thirty miles,
# and the sharpest separation of those two anywhere in the question set.
PANTS = mk(2, None, {
    "Lancashire": 55, "Cheshire": 39, "Northumberland": 36, "Cumberland": 27,
    "Flintshire": 26, "Durham": 18, "Denbighshire": 16, "Yorkshire": 14,
    "Westmorland": 27,          # no sample of its own; sits between Cumberland and Lancashire
    "Midlothian": 9, "Renfrewshire": 8,
    "Derbyshire": 7, "Lincolnshire": 7, "Berkshire": 7, "Warwickshire": 7,
    "Cambridgeshire": 5, "Glamorgan": 5, "Devon": 5, "Worcestershire": 5,
    "Norfolk": 4, "Buckinghamshire": 4,
    "Dorset": 3, "Nottinghamshire": 3, "Surrey": 3, "Sussex": 3,
    "Leicestershire": 3, "Northamptonshire": 3,
    "Staffordshire": 2, "Middlesex": 2, "Gloucestershire": 2, "Shropshire": 2,
    "Suffolk": 2, "Kent": 2,
    "Essex": 1, "Hampshire": 1, "Hertfordshire": 1,
    "Cornwall": 0, "Somerset": 0, "Wiltshire": 0, "Oxfordshire": 0,
    "Monmouthshire": 0, "Bedfordshire": 0, "Herefordshire": 0,
    "Huntingdonshire": 0, "Lanarkshire": 0, "Stirlingshire": 0, "Pembrokeshire": 0,
})
PANTS_BLOBS = [(51.0, 91.1, 58), (57.0, 90.0, 50), (61.0, 85.0, 27), (63.0, 68.0, 33)]
# Stoke sits two cells below Cheshire's 39%% and the smoothing handed it 12%%, but
# its own 100 respondents measure 3%%. It is the whole point of this question --
# Manchester 50%% against Stoke 3%% -- so the inherited value comes back off.
PANTS_CUTS = [(57.0, 97.0, 11)]

# ---- "Look at them animals" ----------------------------------------------------
# Our Dialects, n=2,659, rated 1-5 for acceptability; the values here are the mean
# rating rescaled to 0-100. Demonstrative "them" for "those" -- a third distinct
# grammatical variable alongside the plural pronoun (yous) and the dative (give it
# me), and the widest north/south spread of the three: Yorkshire 81 and Lancashire
# 80 against Oxfordshire 54, with Scotland lowest of all.
# Scotland has 46 respondents -- too few to split by county, enough for one
# national figure (mean 50), so it is applied flat rather than left blank.
THEM = mk(72, [(SCOT, 50)], {
    "Yorkshire": 81, "Lancashire": 80, "Northumberland": 79, "Derbyshire": 76,
    "Durham": 75, "Cheshire": 75, "Shropshire": 75, "Staffordshire": 74,
    "Warwickshire": 72, "Bedfordshire": 72, "Nottinghamshire": 69,
    "Essex": 65, "Kent": 65, "Sussex": 65, "Gloucestershire": 65,
    "Leicestershire": 64, "Somerset": 63, "Hertfordshire": 63, "Middlesex": 62,
    "Surrey": 60, "Worcestershire": 60, "Glamorgan": 57, "Buckinghamshire": 56,
    "Berkshire": 56, "Hampshire": 56, "Oxfordshire": 54,
    # Cumberland and Westmorland have no sample; they sit inside the northern
    # block on every other variable in this survey, so they take its level.
    "Cumberland": 78, "Westmorland": 78,
})


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


# ---- school PE canvas shoes -------------------------------------------------
# Straight from YouGov (Aug 2025, n~38,000), which published COUNTY-level rates
# rather than the usual five broad regions -- so these are transcribed real
# percentages, not values read off a picture. Nationally: plimsolls 53%,
# pumps 23%, gym shoes 6%, daps 5%, sandshoes 5%.
# YouGov's published Scottish regions -> the historic counties in this grid.
# Their table breaks Scotland into 13 areas, which is finer than the article
# prose suggested and corrects several of the estimates it implied.
YG_TAYSIDE  = ["Angus", "Perthshire", "Kinross-shire"]
YG_BORDERS  = ["Berwickshire", "Peeblesshire", "Roxburghshire", "Selkirkshire"]
YG_GRAMPIAN = ["Aberdeenshire", "Banffshire", "Kincardineshire", "Morayshire"]
YG_LOTHIAN  = ["Midlothian", "East Lothian", "West Lothian"]
YG_HIGHLAND = ["Inverness-shire", "Ross-shire", "Cromartyshire", "Sutherland",
               "Caithness", "Nairnshire", "Argyllshire", "Orkney", "Buteshire"]
YG_CENTRAL  = ["Stirlingshire", "Clackmannanshire"]
YG_DANDG    = ["Dumfriesshire", "Kirkcudbrightshire", "Wigtownshire"]

FOOTWEAR = {
    # the southern/eastern default; peaks in Norfolk at 91%. Scotland is far from
    # zero on this one (19-39% depending on region), which the prose obscured.
    "plimsolls": mk(50, [(EANG, 83), (SE_ENG, 79), (EMIDS, 72), (SW_ENG, 55), (WALES, 25),
                         (YG_TAYSIDE, 39), (YG_BORDERS, 37), (YG_GRAMPIAN, 36),
                         (YG_LOTHIAN, 32), (YG_HIGHLAND, 28), (YG_CENTRAL, 22),
                         (YG_DANDG, 19)],
                    {"Norfolk": 91, "Middlesex": 67,
                     "Lancashire": 15, "Cheshire": 15, "Staffordshire": 15,
                     "Warwickshire": 22, "Worcestershire": 25, "Shropshire": 26,
                     "Yorkshire": 40, "Northumberland": 60, "Durham": 58,
                     "Cumberland": 62, "Westmorland": 60,
                     "Glamorgan": 20, "Monmouthshire": 20,
                     "Gloucestershire": 40, "Somerset": 42,
                     "Fife": 36, "Dunbartonshire": 30, "Ayrshire": 22,
                     "Lanarkshire": 20, "Renfrewshire": 10}),
    # North West + West Midlands; barely registers in Scotland (0-12%)
    "pumps": mk(20, [(WMIDS, 62), (SE_ENG, 5), (EANG, 5), (NWALES, 45),
                     (YG_HIGHLAND, 12), (YG_GRAMPIAN, 9), (YG_TAYSIDE, 8),
                     (YG_LOTHIAN, 6), (YG_BORDERS, 6), (YG_DANDG, 4), (YG_CENTRAL, 0)],
                {"Lancashire": 74, "Cheshire": 73, "Staffordshire": 72,
                 "Derbyshire": 45, "Yorkshire": 40, "Nottinghamshire": 30,
                 "Leicestershire": 25, "Warwickshire": 64, "Worcestershire": 60,
                 "Cumberland": 25, "Westmorland": 28, "Durham": 20,
                 "Northumberland": 15, "Middlesex": 6, "Kent": 3, "Sussex": 3,
                 "Surrey": 4, "Flintshire": 50, "Denbighshire": 48,
                 "Caernarfonshire": 40,
                 "Fife": 5, "Lanarkshire": 2, "Ayrshire": 1, "Dunbartonshire": 1,
                 "Renfrewshire": 0}),
    # tight cluster either side of the Severn Estuary
    "daps": mk(1, None,
               {"Monmouthshire": 50, "Glamorgan": 50, "Gloucestershire": 40,
                "Somerset": 40, "Wiltshire": 20, "Dorset": 15, "Devon": 12,
                "Herefordshire": 12, "Brecknockshire": 25, "Carmarthenshire": 30,
                "Pembrokeshire": 22, "Cardiganshire": 15}),
    # strongest on the Clyde but also right across the Borders and Galloway --
    # much wider than the article's Clyde-only framing implied
    "sandshoes": mk(2, [(YG_BORDERS, 42), (YG_DANDG, 34), (YG_HIGHLAND, 28),
                        (YG_TAYSIDE, 19), (YG_CENTRAL, 13), (YG_LOTHIAN, 11),
                        (YG_GRAMPIAN, 7)],
                    {"Renfrewshire": 52, "Ayrshire": 37, "Dunbartonshire": 35,
                     "Lanarkshire": 23, "Fife": 22}),
    "gymshoes": mk(3, [(YG_GRAMPIAN, 39), (YG_TAYSIDE, 29), (YG_DANDG, 24),
                       (YG_HIGHLAND, 23), (YG_LOTHIAN, 22), (YG_CENTRAL, 18),
                       (YG_BORDERS, 7)],
                   {"Renfrewshire": 24, "Fife": 24, "Ayrshire": 22,
                    "Dunbartonshire": 19, "Lanarkshire": 13,
                    "Brecknockshire": 20, "Radnorshire": 20, "Montgomeryshire": 20}),
    # genuinely narrow: Lanarkshire and Central Scotland, not the whole Clyde
    "gutties": mk(1, [(YG_DANDG, 15), (YG_CENTRAL, 32), (YG_BORDERS, 3),
                      (YG_LOTHIAN, 3), (YG_TAYSIDE, 2), (YG_HIGHLAND, 2),
                      (YG_GRAMPIAN, 1)],
                  {"Lanarkshire": 38, "Ayrshire": 12, "Renfrewshire": 8,
                   "Dunbartonshire": 4, "Fife": 3}),
    # the most geographically specific term in the survey: Lothian 18%, and
    # essentially nowhere else in Britain
    "rubbers": mk(0.5, [(YG_LOTHIAN, 18), (YG_BORDERS, 4)],
                  {"Lanarkshire": 1}),
}

grids_all = {"footstrut": grid_json(q1),
             # was-leveling: decoded from Map 13 ("you was" panel) of MacKenzie,
             # Bailey & Turton (2022). The panel is a Gi* hotspot surface on
             # viridis, registered to this grid by matching the N/S/E/W extremes
             # of the two GB mainlands (fit to 0.6 of a cell). Gi* is a
             # z-score, not a rate, so it is converted to acceptance %% by a
             # weighted least-squares fit against the four city figures the
             # paper states in the text: Newcastle 32%% (N=1045), Teesside 42%%
             # (N=177), Liverpool 60%% (N=89), Manchester 60%% (N=152).
             "giveitme": grid_json(decoded_surface("giveitme", 0.12, 0.82)),
             "bookspook": grid_json(decoded_surface("bookspook", 0.10, 0.85)),
             "stirstare": grid_json(surface(NURSESQUARE)),
             "scone": grid_json(decoded_pct_surface("scone")),
             "northforce": grid_json(decoded_surface("northforce", 0.10, 0.85)),
             "forcecure": grid_json(decoded_surface("forcecure", 0.10, 0.85)),
             "youse": grid_json(decoded_surface("youse", 0.20, 0.92)),
             "thfronting": grid_json(decoded_surface("thfronting", 0.04, 0.94)),
             # calibrated against the paper's own reported regional rates (NW 70%,
             # W Mids 61%, NE 26%, East 31%, Kent 26%) -> a true P(rhyme), no rescale
             "singerfinger": grid_json(decoded_pct_surface("singerfinger")),
             "mother_mum": grid_json(decoded_pct_surface("mother_mum")),
             "mother_mam": grid_json(decoded_pct_surface("mother_mam")),
             "mother_mom": grid_json(decoded_pct_surface("mother_mom")),
             "mother_mummy": grid_json(decoded_pct_surface("mother_mummy")),
             "mother_maw": grid_json(decoded_pct_surface("mother_maw")),
             "mother_mammy": grid_json(decoded_pct_surface("mother_mammy")),
             "skiveclass_bunk": grid_json(decoded_pct_surface("skiveclass_bunk")),
             "skiveclass_hookey": grid_json(decoded_pct_surface("skiveclass_hookey")),
             "skiveclass_skip": grid_json(decoded_pct_surface("skiveclass_skip")),
             "skiveclass_skive": grid_json(decoded_pct_surface("skiveclass_skive")),
             "skiveclass_wag": grid_json(decoded_pct_surface("skiveclass_wag")),
             # store as P(rhyme = short a) so "yes, they rhyme" -> North (matches the option)
             "trapbath": grid_json(1 - surface(TRAPBATH)),
             "spelk": grid_json(surface(SPELK)), "spell": grid_json(surface(SPELL)),
             "shiver": grid_json(surface(SHIVER)), "sliver": grid_json(surface(SLIVER)),
             "splinter": grid_json(surface(SPLINTER)),
             "skelf": grid_json(surface(SKELF)),
             "icelolly": grid_json(surface(ICELOLLY)),
             # lolly ice: Liverpool/Merseyside + a North Wales coast cluster (Flintshire/
             # Denbighshire), which shares the Merseyside form; NOT Manchester.
             "lollyice": grid_json(point_blob([(50.7, 91.3, 88), (47.8, 95.0, 76)], 3.0))}
# footwear: county rates, plus the Hull "sandshoes" island that a county-level
# surface cannot express on its own (Yorkshire is one county in this grid)
for term, vm in FOOTWEAR.items():
    surf = surface(vm)
    # Glasgow and Lanarkshire are ONE county here but YouGov separates them, and
    # they disagree sharply: sandshoes 47 vs 23, gutties 6 vs 38. Hull is the same
    # problem inside Yorkshire. Stamp the city values on top of the county surface.
    if term == "sandshoes":
        surf = np.maximum(surf, point_blob([(73.0, 86.0, 48), (39.8, 54.6, 47)], 2.3))
    elif term == "gutties":      # Glasgow is a hole in the Lanarkshire gutties zone
        surf = np.clip(surf - point_blob([(39.8, 54.6, 20)], 2.0), 0.0, 1.0)
    elif term == "rubbers":      # smoothing washes out Lothian, its only stronghold
        surf = np.maximum(surf, point_blob([(49.0, 53.3, 18)], 2.2))
    grids_all["shoe_" + term] = grid_json(surf)

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


# sigma 1.1, not the usual 2.0: Bradford and Leeds are two grid cells apart, and
# at the default width Bradford's snicket peak washed straight over Leeds (51%%
# against a measured 8%%). This variable turns over faster than the grid resolves.

for _t, _vm in {"sofa": SOFA, "settee": SETTEE, "couch": COUCH}.items():
    _surf = surface(_vm)
    if _t in SOFA_BLOBS:
        # couch gets a wider sigma: it is a contiguous Merseyside/west-Lancashire
        # AREA (six adjacent towns, 34-76%%), not isolated peaks. At 1.5 the blobs
        # decayed so fast that Liverpool's 66%% scored a lower neighbourhood mean
        # than Scotland's flat 41%%, and the star went to Glasgow.
        _surf = np.maximum(_surf, point_blob(SOFA_BLOBS[_t], 2.4 if _t == "couch" else 1.5))
    if _t in SOFA_CUTS:
        _surf = np.clip(_surf - point_blob(SOFA_CUTS[_t], 1.5), 0.0, 1.0)
    grids_all["sofa_" + _t] = grid_json(_surf)

for _t, _vm in {"tea": TEA, "dinner": DINNER, "supper": SUPPER}.items():
    grids_all["meal_" + _t] = grid_json(surface(_vm))

for _t, _pts in GUM_BLOBS.items():
    grids_all["gum_" + _t] = grid_json(point_blob(_pts, 1.6))
# "chewing gum" is the national default: whatever the local terms leave behind
grids_all["gum_gum"] = grid_json(np.clip(
    1.0 - sum(np.array(grids_all["gum_" + t], dtype=float) for t in GUM_BLOBS), 0.2, 1.0))

for _term, _vm in ALLEY.items():
    _surf = surface(_vm)
    if _term in ALLEY_BLOBS:
        _surf = np.maximum(_surf, point_blob(ALLEY_BLOBS[_term], 1.1))
    # Three cities sit just inside a county whose headline word they do not share,
    # and county-level smoothing hands it to them anyway. Measured rates: Leeds
    # snicket 8%%, Sheffield jitty ~1%% (it is on the Derbyshire line), Liverpool
    # ginnel 4%%. Subtract the inherited value back off.
    if _term == "snicket":
        _surf = np.clip(_surf - point_blob([(63.0, 85.0, 44)], 1.1), 0.0, 1.0)
    elif _term == "jitty":
        _surf = np.clip(_surf - point_blob([(64.0, 92.0, 40)], 1.6), 0.0, 1.0)
    elif _term == "ginnel":
        _surf = np.clip(_surf - point_blob([(51.0, 91.0, 38)], 1.5), 0.0, 1.0)
    grids_all["alley_" + _term] = grid_json(_surf)
# alley(way) is the national default: whatever the regional words leave behind
grids_all["alley_alley"] = grid_json(np.clip(
    1.0 - sum(np.array(grids_all["alley_" + t], dtype=float) for t in ALLEY), 0.15, 1.0))

for _term, _vm in PRANK.items():
    # cherry turns over fast (Gloucester is ginger-free, Bristol 4 cells south is
    # 82%% ginger), so it gets a tighter county sigma than the usual 2.0.
    _surf = surface(_vm, 1.8 if _term == "cherry" else 2.0)
    if _term in PRANK_BLOBS:
        _surf = np.maximum(_surf, point_blob(PRANK_BLOBS[_term], 1.7))
    if _term in PRANK_CUTS:
        _surf = np.clip(_surf - point_blob(PRANK_CUTS[_term], 1.7 if _term == "run" else 1.5), 0.0, 1.0)
    grids_all["prank_" + _term] = grid_json(_surf)

_pants = np.maximum(surface(PANTS), point_blob(PANTS_BLOBS, 1.6))
_pants = np.clip(_pants - point_blob(PANTS_CUTS, 1.4), 0.0, 1.0)
grids_all["pants"] = grid_json(_pants)
grids_all["trousers"] = grid_json(np.clip(1.0 - _pants, 0.15, 1.0) * land)
# Acceptability spans only 50-81 nationally -- a real finding (the construction is
# broadly accepted everywhere) but, left raw, every rating on the slider reads back
# as "much of Britain". Stretched onto the same 0.12-0.82 band the give-it-me
# slider already uses, so the two behave alike and the map shows the north-south
# gradient instead of a flat wash. The band is relative, like give-it-me's.
_them = surface(THEM)
_tv = _them[land]
_them = np.where(land, 0.12 + (_them - _tv.min()) / (_tv.max() - _tv.min()) * 0.70, 0.0)
grids_all["them"] = grid_json(_them)

grids_all["none_gum"] = negative_union(["gum_" + t for t in list(GUM_BLOBS) + ["gum"]])
grids_all["none_alley"] = negative_union(["alley_" + t for t in list(ALLEY) + ["alley"]])
# "no word for this" is the inverse of COVERAGE, and negative_union takes the max
# across surfaces -- fine when the variants are alternatives to each other, wrong
# for Scotland, where chap door run (27), chappie (23) and chicken mellie (7) are
# three DIFFERENT people's answers. Taking the max there would say 73%% of Scots
# have no word for this when 57%% just named one. So the Scottish three are summed
# into a coverage surface first, and the union runs over that.
_prank_cover = dict(grids_all)
_prank_cover["prank_scotfam"] = grid_json(np.clip(
    sum(np.array(grids_all["prank_" + t], dtype=float) for t in ("chap", "chappie", "mellie")),
    0.0, 1.0))
grids_all["prank_scotfam"] = _prank_cover["prank_scotfam"]
grids_all["none_prank"] = negative_union(
    ["prank_" + t for t in PRANK if t not in ("chap", "chappie", "mellie")] + ["prank_scotfam"])
del grids_all["prank_scotfam"]      # scaffolding only; never referenced by an option
grids_all["none_splinter"] = negative_union(["splinter", "spelk", "spell", "shiver", "sliver", "skelf"])
grids_all["none_bread"] = negative_union(list(BREAD.keys()))
grids_all["none_shoe"] = negative_union(["shoe_" + t for t in FOOTWEAR])
grids_all["none_mother"] = negative_union(["mother_" + t for t in ["mum","mam","mom","mummy","maw","mammy"]])
grids_all["none_tag"] = negative_union(["tag_" + t for t in TAG_TERMS])
grids_all["none_skiveclass"] = negative_union(["skiveclass_" + t for t in ["bunk", "hookey", "skip", "skive", "wag"]])

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
    ("Leeds", "Yorkshire", 63, 85), ("Bradford", "West Yorkshire", 61, 85), ("Sheffield", "", 64, 92), ("Hull", "", 73, 86),
    ("Birmingham", "Brummie", 60, 105), ("Stoke", "the Potteries", 57, 97),
    ("Nottingham", "the East Midlands", 67, 98), ("Derby", "Derbyshire", 63, 99),
    ("London", "", 75, 120), ("Norwich", "East Anglia", 88, 103),
    ("Margate", "Kent", 86, 122),
    ("Bristol", "the West Country", 54, 120), ("Exeter", "the West Country", 46, 131),
    # added for cherry knocking: the Severn Vale measures 95%% for it and Bristol,
    # four cells south, measures 9%%. Without a place here the star for that answer
    # landed on Bristol, i.e. on the one nearby city that does not use the word.
    ("Gloucester", "the Severn Vale", 57, 115),
    ("Edinburgh", "Scotland", 49, 53), ("Glasgow", "Scotland", 40, 55),
    ("Aberdeen", "", 58, 35), ("Cardiff", "Wales", 49, 119),
    ("Swansea", "South Wales", 41, 118),
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

# ---- fine-grained dialect GROUPS (for the landing-page teaser), colours sampled
# directly off updated-landing-page-map.jpg (Starkey-Comics-style British Isles
# dialect map). County-level base regions, with small city-centred "patches"
# (Geordie, Brummie, Scouse, ...) stamped on top for the areas the source map
# calls out individually. Great Britain only (the grid has no Ireland/Man).
# NOTE: Highland/Lowland Scottish boundary pulled a bit further south than
# NSCOT/CSCOT's own split (Perthshire/Stirlingshire are visually Lowland-shaded
# on the source map even though they're grouped under "Highland" everywhere
# else in this file) so the two areas match the source's proportions.
_HL_SOUTH = {"Perthshire", "Stirlingshire", "Argyllshire"}
_HIGHLAND = [c for c in NSCOT if c not in _HL_SOUTH]
_LOWLAND = [c for c in CSCOT if c != "Lanarkshire"] + SSCOT + list(_HL_SOUTH - {"Argyllshire"})
DIALECT = [
    ("Highland Scottish", _HIGHLAND, (0, 0, 82)),
    ("Lowland Scottish", _LOWLAND + ["Lanarkshire", "Argyllshire"], (2, 0, 128)),
    ("Cumbrian", ["Cumberland", "Westmorland"], (93, 68, 96)),
    ("Northumbrian", ["Northumberland"], (104, 69, 107)),
    ("Pitmatic", ["Durham"], (128, 68, 127)),
    ("Yorkshire", ["Yorkshire"], (224, 116, 224)),
    ("Lancashire", ["Lancashire", "Cheshire"], (222, 138, 223)),
    ("West Midlands", WMIDS, (134, 75, 39)),
    ("East Midlands", ["Derbyshire", "Leicestershire", "Nottinghamshire", "Rutland", "Lincolnshire"],
     (232, 112, 40)),
    ("South-East Midlands", ["Northamptonshire", "Bedfordshire", "Buckinghamshire"], (233, 126, 63)),
    # deep gold, not the source map's pale yellow: on the cream page that only
    # reached ~1.2:1 contrast and read as a blank gap in the map
    ("East Anglia", ["Cambridgeshire", "Huntingdonshire", "Norfolk", "Suffolk", "Essex"],
     (210, 172, 28)),
    ("South East", ["Surrey", "Hampshire", "Berkshire", "Oxfordshire", "Hertfordshire"], (56, 0, 0)),
    ("Sussex", ["Sussex"], (96, 1, 0)),
    ("Kentish", ["Kent"], (115, 1, 1)),
    ("London", ["Middlesex"], (169, 14, 18)),
    ("West Country", ["Dorset", "Wiltshire", "Gloucestershire"], (0, 46, 0)),
    ("Somerset", ["Somerset"], (1, 92, 0)),
    ("Devonshire", ["Devon"], (0, 113, 0)),
    ("Anglo-Cornish", ["Cornwall"], (0, 130, 0)),
    ("Welsh", WALES, (67, 44, 28)),
]
county_dialect = {}
for di, (dn, cl, _col) in enumerate(DIALECT):
    for cn in cl:
        county_dialect[cn] = di
dialect_grid = [[(county_dialect.get(names[cg[r][c]], -1) if land[r][c] else -1)
                 for c in range(W)] for r in range(H)]
dialect_colors = [[dn, list(col)] for dn, _cl, col in DIALECT]

# ---- city-centred patches: small named dialects the source map calls out
# individually inside a larger county (Geordie inside Northumberland, Brummie
# inside the West Midlands, ...). Each is a distance-from-city falloff LIMITED
# to a set of real counties, so it follows the county's actual coastline/border
# instead of stamping a bare geometric circle across it.
PATCHES = [
    ("Glaswegian", 39.8, 54.6, 7, (35, 36, 176), ["Lanarkshire", "Renfrewshire", "Dunbartonshire"]),
    ("Geordie", 62.6, 67.9, 7, (126, 68, 119), ["Northumberland"]),
    ("Mackem", 63, 69, 5, (193, 68, 193), ["Durham"]),
    ("Teesside", 64, 74, 5, (227, 78, 226), ["Durham", "Yorkshire"]),   # source legend misspells this "Tesside"
    ("Scouse", 50.7, 91.3, 8, (226, 172, 225), ["Lancashire", "Cheshire"]),
    ("Mancunian", 57.2, 90.2, 8, (227, 200, 226), ["Lancashire", "Cheshire"]),
    ("Brummie", 60.2, 105.0, 8, (160, 85, 37), ["Warwickshire", "Worcestershire", "Staffordshire"]),
    ("Coventry", 66, 103, 5, (180, 93, 38), ["Warwickshire"]),
    ("Potteries", 57, 97, 5, (207, 101, 36), ["Staffordshire"]),
    ("Bristolian", 54.2, 120.4, 5, (0, 58, 0), ["Gloucestershire", "Somerset"]),
    ("Cardiff", 49.1, 120.0, 5, (101, 66, 44), ["Glamorgan", "Monmouthshire"]),
    ("Estuary", 84, 118, 6, (188, 0, 0), ["Essex", "Kent"]),
    ("Cockney", 77, 120, 4, (244, 26, 27), ["Middlesex", "Essex"]),
]
dialect_colors += [[pn, list(pcol)] for pn, _c, _r, _rad, pcol, _cl in PATCHES]
for pi, (pn, pc, pr, prad, _pcol, allowed) in enumerate(PATCHES):
    di = len(DIALECT) + pi
    allowed_set = set(allowed)
    for r in range(H):
        for c in range(W):
            if (land[r][c] and (c - pc) ** 2 + (r - pr) ** 2 <= prad ** 2
                    and names[cg[r][c]] in allowed_set):
                dialect_grid[r][c] = di

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
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Great British Dialect Quiz</title>
<meta name="description" content="Answer a few questions and see where in Britain your speech places you, on maps redrawn from published dialect research.">
<!-- favicon: a plain globe -- flat blue disc, equator, axis, one meridian.
     No shading and no continents on purpose: three lines is the least that
     still reads as a globe, and anything more turns to mush at 16px.
     Drawn here, so there is nothing to license. -->
<link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,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">
<link rel="icon" type="image/png" sizes="64x64" href="data:image/png;base64,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">
<link rel="apple-touch-icon" sizes="180x180" href="data:image/png;base64,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">
<style>
  :root{--bg:#f6f1e7;--ink:#2b2b2b;--inkstrong:#000;--body:#555;--muted:#8a857c;--accent:#c0141f;--accent-h:#a5101a;--card:#fff;--line:#e6e1d8;--line2:#cfc7b8;--track:#e4ddcf;--sel:#fdf0f0;--soft:#ece5d8;--disabled:#dcbcbe;--box:#c9c3b8;--teal:#0a7a63;--faint:#b3ada2;--bgimg:radial-gradient(circle at 6%% 8%%, rgba(38,84,168,.085), rgba(38,84,168,0) 38%%),radial-gradient(circle at 95%% 92%%, rgba(192,20,31,.075), rgba(192,20,31,0) 40%%),url("data:image/svg+xml,%%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%%3E%%3Cfilter id='n'%%3E%%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%%3E%%3C/filter%%3E%%3Crect width='160' height='160' filter='url(%%23n)' opacity='0.32'/%%3E%%3C/svg%%3E")}
  *{box-sizing:border-box;}
  html,body{margin:0;min-height:100%%;}
  body{background-color:var(--bg);background-image:var(--bgimg);background-attachment:fixed;
       color:var(--ink);transition:background-color .25s ease,color .25s ease;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;}
  #app{max-width:1060px;margin:0 auto;padding:20px 20px 16px;position:relative;}
  #qimg{display:block;width:230px;height:230px;object-fit:cover;border-radius:16px;margin:48px auto 0;box-shadow:0 6px 18px rgba(0,0,0,.16);}
  .sliderbox{margin:20px 0 8px;}
  .sliderbox input[type=range]{width:100%%;accent-color:var(--accent);height:6px;cursor:pointer;}
  .slabels{display:flex;justify-content:space-between;align-items:flex-start;font-size:12px;color:var(--muted);margin-top:10px;line-height:1.35;}
  .slabels .sval{font-size:26px;font-weight:750;color:var(--accent);align-self:center;}
  #restart{position:absolute;top:8px;right:16px;display:inline-flex;align-items:center;gap:7px;
       font-size:13.5px;font-weight:650;color:#fff;background:var(--accent);
       border:none;border-radius:10px;padding:10px 18px;cursor:pointer;z-index:20;box-shadow:0 3px 10px rgba(192,20,31,.22);}
  #restart:hover{background:var(--accent-h);}
  #restart .ricon{font-size:30px;line-height:1;display:flex;align-items:center;}
  header{text-align:center;margin-bottom:6px;}
  .site-title{font-size:27px;font-weight:750;letter-spacing:-.015em;margin:0 0 3px;}
  .site-sub{font-size:13px;color:var(--muted);margin:0 0 18px;}
  .progress-wrap{max-width:420px;margin:0 auto 7px;height:6px;background:var(--track);border-radius:99px;overflow:hidden;}
  .progress-bar{height:100%%;width:0;background:var(--accent);border-radius:99px;transition:width .3s ease;}
  #progress{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);text-align:center;margin-bottom:14px;}
  #stage{display:flex;gap:30px;align-items:flex-start;justify-content:center;flex-wrap:wrap;}
  #left{flex:0 0 340px;max-width:340px;}
  #left.wide{flex:0 0 470px;max-width:470px;}
  #left.wide #opts{display:grid;grid-template-columns:1fr 1fr;gap:0 10px;align-items:start;}
  #qtext{font-size:24px;line-height:1.3;margin:0 0 7px;font-weight:650;}
  #qtag{font-size:12px;color:var(--muted);margin-bottom:20px;}
  #qtag.real{color:var(--teal);} #qtag::before{content:"\\25CF  ";font-size:9px;vertical-align:middle;}
  .opt{display:flex;align-items:center;gap:10px;width:100%%;margin:5px 0;padding:9px 14px;font-size:14.5px;text-align:left;
       background:var(--card);border:1.5px solid var(--line);border-radius:12px;cursor:pointer;transition:all .12s;color:var(--ink);}
  .opt:not(:disabled):hover{border-color:var(--accent);transform:translateY(-1px);box-shadow:0 4px 14px rgba(0,0,0,.07);}
  .opt.sel{border-color:var(--accent);background:var(--sel);}
  .opt:disabled{cursor:default;opacity:.45;} .opt.sel:disabled{opacity:1;}
  .opt .box{flex:0 0 18px;height:18px;border:1.5px solid var(--box);border-radius:5px;display:inline-flex;
            align-items:center;justify-content:center;font-size:12px;line-height:1;color:#fff;}
  .opt.sel .box{background:var(--accent);border-color:var(--accent);}
  /* An option's label is set as raw children of the .opt flex row, so a bare
     <span> gloss would become its OWN flex column and squeeze the text beside
     it. .opt-txt keeps label + gloss together as one item that wraps normally. */
  .opt .opt-txt{flex:1;min-width:0;}
  .opt .opt-alt{color:var(--muted);font-weight:400;font-size:13px;white-space:nowrap;}
  .hint{font-size:12px;color:var(--muted);margin:0 0 8px;min-height:16px;}
  #next{margin-top:12px;width:100%%;padding:13px;font-size:15px;font-weight:600;color:#fff;background:var(--accent);
        border:none;border-radius:12px;cursor:pointer;transition:background .12s;}
  #next:hover:not(:disabled){background:var(--accent-h);} #next:disabled{background:var(--disabled);cursor:not-allowed;}
  #leftdone{font-size:16px;color:var(--body);line-height:1.55;} #leftdone b{color:var(--ink);}
  /* ---- result screen: one headline, one gauge, one small table, one footnote.
     Everything used to be a separate free-floating sentence, which read as noise. */
  .res-eyebrow{font-size:12.5px;color:var(--body);margin:0 0 3px;}
  .res-name{font-size:32px;font-weight:750;line-height:1.1;color:var(--accent);letter-spacing:-.015em;}
  .res-gauge{margin-top:18px;max-width:300px;}
  .res-gauge .glab{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
  .res-gauge .track{height:8px;background:var(--track);border-radius:99px;overflow:hidden;}
  .res-gauge .fill{height:100%%;background:linear-gradient(to right,#e79aa1,var(--accent));border-radius:99px;}
  .res-gauge .gcap{font-size:14px;font-weight:650;color:var(--ink);margin-top:8px;}
  .res-rows{margin-top:20px;max-width:340px;border-top:1px solid var(--line);}
  .res-row{display:flex;gap:14px;align-items:baseline;padding:9px 0;border-bottom:1px solid var(--line);}
  .res-row .k{flex:0 0 86px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
       color:var(--muted);white-space:nowrap;}
  .res-row .v{font-size:14px;font-weight:600;color:var(--ink);line-height:1.4;}
  .res-foot{margin-top:13px;font-size:12px;color:var(--faint);}
  .res-foot .dot{margin:0 7px;opacity:.6;}
  .res-foot .srcwrap{margin-top:0;vertical-align:baseline;}
  /* hover prompt under the map title -- the summary map is hoverable like the
     per-question ones, but nothing on screen said so */
  .rsub{display:block;font-size:12px;color:var(--muted);font-weight:400;margin-top:2px;}
  .navbtns{margin-top:14px;display:flex;gap:12px;align-items:center;}
  #back{font-size:14px;font-weight:600;color:var(--ink);background:var(--card);border:1.5px solid var(--line);
        border-radius:10px;padding:11px 18px;cursor:pointer;transition:all .12s;}
  #back:hover{border-color:var(--line2);background:var(--soft);}
  #startover{font-size:14px;color:#fff;background:var(--ink);border:none;border-radius:9px;padding:9px 16px;cursor:pointer;}
  #startover:hover{background:var(--inkstrong);}
  /* 470px == W*BASE_CELL, the map at its full 5px-per-cell resolution. Capping
     lower than that just threw away detail on big screens; 340 (or 470 when the
     options grid goes wide) + 30 gap + this still fits #app's 1060px. */
  #right{flex:0 0 auto;text-align:center;max-width:470px;}
  #rtitle{font-size:14px;line-height:1.4;margin-bottom:6px;min-height:2em;color:var(--muted);}
  #rtitle b{color:var(--ink);}
  #out{position:relative;display:inline-block;}
  canvas{display:block;cursor:pointer;max-height:52vh;max-width:86vw;width:auto;height:auto;border-radius:6px;}
  #rprompt{color:var(--faint);font-size:14px;padding:60px 44px;border:2px dashed var(--line);border-radius:14px;}
  .srcwrap{position:relative;display:inline-block;margin-top:18px;}
  .srcbtn{font-size:12px;color:var(--teal);cursor:help;}
  .srcbtn:hover{text-decoration:underline;}
  .srcpop{display:none;position:absolute;bottom:135%%;left:0;width:320px;max-width:82vw;
       font-size:12px;line-height:1.5;color:var(--body);text-align:left;background:var(--card);
       border:1px solid var(--line);border-radius:10px;padding:11px 13px;z-index:40;
       box-shadow:0 8px 24px rgba(0,0,0,.16);}
  .srcpop ul{list-style:none;margin:7px 0 0;padding:0;}
  .srcpop li{margin-bottom:5px;color:var(--muted);}
  /* CC BY-SA obliges us to name the author and link the licence, so these
     have to be real links rather than plain text */
  .srcpop a{color:var(--teal);text-decoration:underline;text-underline-offset:2px;}
  .srcpop a:hover{color:var(--accent);}
  .srcwrap:hover .srcpop,.srcwrap.open .srcpop{display:block;}
  #match{font-size:16px;margin-top:10px;min-height:1.3em;font-weight:600;line-height:1.35;} #match b{color:var(--accent);}
  /* legend above the (i) line, always stacked - side-by-side on wide maps and
     wrapped on narrow ones read as two different layouts, so pin it to one. */
  #mapfoot{display:flex;flex-direction:column;align-items:center;justify-content:center;
       gap:7px;margin-top:10px;}
  #legend{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--muted);margin:0;}
  #legend .bar{width:104px;height:7px;border-radius:99px;
       background:linear-gradient(to right,rgb(18,86,222),rgb(232,232,236),rgb(214,16,32));border:1px solid var(--line);}
  #infowrap{position:relative;display:inline-block;margin:0;}
  #infobtn{font-size:11.5px;color:var(--teal);cursor:help;} #infobtn:hover{text-decoration:underline;}
  #info{display:none;position:absolute;bottom:calc(100%% + 9px);left:50%%;transform:translateX(-50%%);width:340px;max-width:82vw;
        font-size:12.5px;line-height:1.5;color:var(--body);text-align:left;background:var(--card);border:1px solid var(--line);
        border-radius:10px;padding:11px 13px;box-shadow:0 8px 24px rgba(0,0,0,.16);z-index:40;}
  #info::before{content:"";position:absolute;top:100%%;left:50%%;transform:translateX(-50%%);
        border:7px solid transparent;border-top-color:var(--card);}
  #infowrap::after{content:"";position:absolute;left:0;right:0;bottom:100%%;height:11px;}
  #infowrap:hover #info,#infowrap.open #info{display:block;}
  .ilinks{margin-top:9px;padding-top:8px;border-top:1px solid var(--line);
       font-size:11.5px;line-height:1.55;color:var(--muted);}
  .ilinks a{color:var(--teal);text-decoration:underline;text-underline-offset:2px;}
  .ilinks a:hover{color:var(--accent);}
  #info .src{color:var(--faint);font-size:11px;margin-top:8px;} #info .isep{border:none;border-top:1px solid var(--line);margin:9px 0;}
  #detail{font-size:13px;color:#444;margin-top:6px;min-height:1.2em;}
  .tip{position:absolute;background:var(--ink);color:#fff;padding:6px 10px;border-radius:6px;font-size:12px;
       line-height:1.35;white-space:nowrap;pointer-events:none;opacity:0;transform:translate(-50%%,-118%%);
       transition:opacity .1s;} .tip b{font-weight:600;} .tip small{opacity:.82;}
  /* landing page: two panels, vertically centred and filling the viewport */
  #intro{display:flex;flex-direction:column;align-items:center;justify-content:center;max-width:1180px;
       margin:0 auto;min-height:calc(100vh - 40px);}
  .intro-head{text-align:center;margin-bottom:14px;width:100%%;}
  .intro-panels{display:flex;gap:8vw;align-items:center;justify-content:center;flex-wrap:wrap;width:100%%;}
  .intro-left{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;text-align:center;}
  .intro-right{flex:0 1 400px;max-width:400px;display:flex;flex-direction:column;justify-content:center;text-align:left;}
  .intro-title{font-size:clamp(30px,4.3vw,48px);font-weight:800;letter-spacing:-.025em;margin:0 0 6px;line-height:1.03;}
  .intro-sub{font-size:13.5px;color:var(--muted);margin:0;}
  /* soft radial pool under the map so it sits on the page instead of floating */
  #introcvwrap{position:relative;display:inline-block;}
  #introcvwrap::before{content:"";position:absolute;inset:-6%% -10%%;pointer-events:none;z-index:0;
       background:radial-gradient(ellipse at 50%% 52%%, rgba(64,54,44,.075), rgba(64,54,44,0) 68%%);}
  #introcv{position:relative;z-index:1;display:block;margin:0 auto;width:290px;image-rendering:auto;
       cursor:crosshair;max-width:100%%;height:auto;}
  #introtip{position:absolute;pointer-events:none;opacity:0;transform:translate(-50%%,-135%%);white-space:nowrap;
       font-size:12px;font-weight:650;color:#fff;padding:4px 9px;border-radius:6px;transition:opacity .08s;z-index:5;
       box-shadow:0 3px 10px rgba(0,0,0,.18);}
  /* the tour/hover read-out: a pill that takes on the colour of the live region */
  .intro-cap{display:inline-block;font-size:12px;letter-spacing:.085em;text-transform:uppercase;font-weight:700;
       color:var(--muted);margin:10px 0 0;padding:5px 14px;border-radius:99px;line-height:1.35;
       transition:color .35s ease,background-color .35s ease;}
  /* standing affordance: the tour names regions, this tells you that you can drive it */
  .intro-hint{font-size:12px;color:var(--muted);margin:9px 0 0;letter-spacing:.01em;
       transition:opacity .3s ease;}
  .intro-hint.dim{opacity:.28;}
  .intro-lead{font-size:21px;line-height:1.35;font-weight:700;letter-spacing:-.01em;margin:0 0 11px;}
  .intro-body{font-size:14px;line-height:1.5;color:var(--body);margin:0 0 11px;}
  /* three numbers instead of a paragraph of claims: the scale IS the argument,
     and it reads in a glance where prose has to be waded through */
  .intro-stats{display:flex;gap:26px;margin:16px 0 14px;padding:12px 0;
       border-top:1px solid var(--line);border-bottom:1px solid var(--line);}
  .intro-stats div{line-height:1.1;}
  .intro-stats b{display:block;font-size:20px;font-weight:750;color:var(--accent);
       letter-spacing:-.015em;margin-bottom:3px;}
  .intro-stats span{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);}
  .intro-why{font-size:13px;line-height:1.5;color:var(--body);margin:0 0 12px;}
  .intro-points{list-style:none;margin:0 0 16px;padding:0;}
  .intro-points li{position:relative;padding-left:15px;font-size:13px;line-height:1.45;
       color:var(--body);margin-bottom:6px;}
  .intro-points li:last-child{margin-bottom:0;}
  .intro-points li::before{content:"";position:absolute;left:0;top:8px;width:5px;height:5px;
       border-radius:50%%;background:var(--accent);opacity:.55;}
  .intro-points b{color:var(--ink);font-weight:680;}
  .authorlink{color:var(--teal);text-decoration:underline;text-underline-offset:2px;}
  .authorlink:hover{color:var(--accent);}
  #startbtn{font-size:16px;font-weight:650;color:#fff;background:var(--accent);border:none;border-radius:12px;
       padding:15px 36px;cursor:pointer;transition:all .14s;box-shadow:0 6px 18px rgba(192,20,31,.24);}
  #startbtn:hover:not(:disabled){background:var(--accent-h);transform:translateY(-2px);box-shadow:0 9px 22px rgba(192,20,31,.30);}
  #startbtn:disabled{background:var(--disabled);cursor:not-allowed;box-shadow:none;transform:none;}
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
  .combo-item:hover,.combo-item.active{background:var(--sel);color:var(--accent);}
  .combo-item:hover .cc,.combo-item.active .cc{color:var(--accent);}
  /* first (hometown) question: a distinct tinted card so it doesn't look like a dialect Q */
  .htq{background:var(--soft);border:1.5px solid var(--line);border-radius:14px;padding:16px;}
  .htq-note{font-size:12px;color:var(--muted);margin:0 0 12px;line-height:1.45;}
  .or-div{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:11px;
       text-transform:uppercase;letter-spacing:.09em;margin:14px 0;}
  .or-div::before,.or-div::after{content:"";flex:1;height:1px;background:var(--line2);}
  .altbtn{width:100%%;padding:12px 14px;font-size:13.5px;line-height:1.35;background:transparent;
       border:1.5px dashed var(--line2);border-radius:10px;color:var(--muted);cursor:pointer;transition:all .12s;}
  .altbtn:hover{border-color:var(--accent);color:var(--accent);background:var(--card);}
  .altbtn.chosen{border-style:solid;border-color:var(--accent);color:var(--accent);background:var(--sel);}
  .consent{display:flex;align-items:flex-start;gap:8px;font-size:12px;color:var(--muted);
       margin:0 0 18px;text-align:left;line-height:1.45;cursor:pointer;}
  .consent input{margin:1px 0 0;flex:0 0 auto;width:15px;height:15px;accent-color:var(--accent);cursor:pointer;}
  .tlink{color:var(--teal);text-decoration:underline;cursor:help;position:relative;}
  .terms-pop{display:none;position:absolute;bottom:150%%;left:0;width:330px;max-width:80vw;font-size:11.5px;
       line-height:1.55;color:var(--body);text-align:left;font-weight:400;background:var(--card);border:1px solid var(--line);
       border-radius:10px;padding:12px 14px;box-shadow:0 10px 26px rgba(0,0,0,.18);z-index:60;}
  .tlink:hover .terms-pop,.tlink.open .terms-pop{display:block;}
  .aboutwrap{position:relative;display:inline-block;vertical-align:middle;}
  .aboutbtn{color:var(--teal);cursor:help;font-size:15px;line-height:1;}
  .aboutinfo{display:none;position:absolute;bottom:150%%;left:50%%;transform:translateX(-50%%);width:430px;max-width:88vw;
       font-size:12.5px;line-height:1.6;color:var(--body);text-align:left;background:var(--card);border:1px solid var(--line);
       border-radius:10px;padding:14px 16px;box-shadow:0 10px 28px rgba(0,0,0,.18);z-index:60;letter-spacing:normal;}
  .aboutinfo::after{content:"";position:absolute;top:100%%;left:50%%;transform:translateX(-50%%);
       border:7px solid transparent;border-top-color:var(--card);}
  .aboutwrap:hover .aboutinfo,.aboutwrap.open .aboutinfo{display:block;}
  button,.opt,#infobtn,.aboutbtn{touch-action:manipulation;}   /* removes tap delay on phones */
  /* ---- narrow screens / phones: stop anything running off the edge ---- */
  @media (max-width:640px){
    #app{padding:16px 14px 14px;}
    #left,#left.wide{flex:1 1 auto;max-width:100%%;width:100%%;}
    #left.wide #opts{display:block;}
    /* relocated by applyLayout(): image sits in the question column, buttons below the map */
    #stage>#next,#stage>.navbtns{width:100%%;}
    #stage>#next{margin-top:18px;}
    #stage>.navbtns{justify-content:center;margin-top:10px;}
    #left>#qimg{margin:4px auto 16px;}
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
      <div class="intro-sub">Answer a few questions &mdash; see where each answer places you on the map.</div>
    </div>
    <div class="intro-panels">
    <div class="intro-left">
      <div id="introcvwrap"><canvas id="introcv"></canvas><div id="introtip"></div></div>
      <p class="intro-cap" id="introcap">&mdash;</p>
      <p class="intro-hint" id="introhint">Hover over the map to explore each dialect</p>
    </div>
    <div class="intro-right">
      <p class="intro-lead">How you say a few everyday words &mdash; and what you call bread, or a splinter &mdash; quietly gives away where in Britain you&rsquo;re from.</p>
      <div class="intro-stats">
        <div><b>25</b><span>questions</span></div>
        <div><b>8 km</b><span>resolution</span></div>
        <div><b>1</b><span>result</span></div>
      </div>
      <p class="intro-why">Britain packs more dialect variation into one island than almost anywhere in the
        English-speaking world, and most of it sits in decades-old atlases. This one is live &mdash; and it
        gets redrawn by everyone who plays.</p>
      <ul class="intro-points">
        <li><b>Every map is live.</b> Hover over any city to see the word it actually uses &mdash;
          scored across 4,025 points of Britain.</li>
        <li><b>Nothing is invented.</b> Every map comes from published research.</li>
        <li><b>It sharpens as you play.</b> With consent, your answers redraw the maps. Everything is
          stored anonymously and used solely to study regional language variation and to improve
          future versions of the quiz.</li>
      </ul>
      <label class="consent"><input type="checkbox" id="consent"><span>I agree to the <span class="tlink" id="termsbtn">terms of data collection<span class="terms-pop" id="termspop">By ticking this box, you consent to the collection and storage of your quiz answers and, if you provide it, your hometown. This information is stored <b>anonymously</b>: no name or email address is recorded, we do not store your IP address with your answers, and the record cannot be traced back to you. It is used solely to study regional language variation and to improve future versions of the quiz. Your data will not be sold or shared with third parties.</span></span></span></label>
      <button id="startbtn">Start the quiz &rarr;</button>
      <p class="intro-note"><span class="aboutwrap"><span class="aboutbtn">&#9432;</span><span class="aboutinfo">Made by <b id="authorname">Alan Levita</b> during an internship at the Intellectual Forum, Jesus College, Cambridge.<br><br>Your answers are used to estimate roughly where you&rsquo;re from. Every map is a pixel-art rendering of an existing map from published dialect research or a large-scale survey &mdash; each one is redrawn onto a coarse grid of Great Britain, not traced.<br><br>Each question cites the research behind its own map: open the &#9432; beside any map to see it. The full list is also on the results screen.</span></span> Designed and produced by Alan Levita</p>
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
      <div id="mapfoot">
        <div id="legend" style="display:none"><span>uncommon</span><span class="bar"></span><span>common</span></div>
        <div id="infowrap" style="display:none"><span id="infobtn"></span><div id="info"></div></div>
      </div>
      <div id="detail"></div>
    </div>
  </div>
</div>
<script>
const W=%d,H_=%d;
// The pixel-art grid is drawn at CELL device pixels per cell with a GAP-wide
// gutter. These are NOT constants: the canvas used to be drawn at a fixed 5px
// and then scaled by CSS to fit the viewport, which meant every cell landed on
// a fractional number of screen pixels (3.08 at 1280x860) -- so the 1px gutters
// got resampled away on some columns and doubled on others, and the pattern
// shifted every time the window changed size. fitCanvas() instead picks a whole
// number of device pixels per cell, so the grid is exact at any window size.
// BASE_* are the reference proportions everything else is scaled against.
const BASE_CELL=5,BASE_GAP=1;
let CELL=BASE_CELL,GAP=BASE_GAP,PXS=1,GW=1;   // PXS scales dots/stars; GW = gutter strength
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
const GUM_IMG=%s;
// NOTE: ordered newest-added -> oldest-added (not the usual thematic order) so
// new features land at the front of the quiz for quick manual testing. hometown
// stays first regardless (it's an intake field, not a dialect question).
const QUESTIONS=[
  // first question: where did you grow up? (a GB town type-ahead, or "not from GB").
  // Not a heat-map question — collected (with consent) for future training, not scored.
  {id:"hometown",text:"Where in Great Britain did you grow up?",hometownq:true},
  {id:"prank",text:"What do you call knocking on someone&rsquo;s door and running away?",
   tag:"real data (YouGov 2025, n&gt;12,000; Our Dialects, n=1,469)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"prank",infoLabel:"words for knock-a-door-run",
   opts:[
     {label:"Knock a door run",v:"kadr",term:"knock a door run",grid:"prank_run"},
     {label:"Knock and run",v:"run",term:"knock and run",grid:"prank_run"},
     {label:"Knock knock run",v:"kkrun",term:"knock knock run",grid:"prank_run"},
     {label:"Knock down ginger",v:"ginger",term:"knock down ginger",grid:"prank_ginger"},
     {label:"Knock knock ginger",v:"kkginger",term:"knock knock ginger",grid:"prank_kkginger"},
     {label:"Rat a tat ginger",v:"ratatat",term:"rat a tat ginger",grid:"prank_wales"},
     {label:"Bobby knocking",v:"bobby",term:"bobby knocking",grid:"prank_wales"},
     {label:"Cherry knocking",v:"cherry",term:"cherry knocking",grid:"prank_cherry"},
     {label:"<span class='opt-txt'>Knocking nine doors <span class='opt-alt'>(or nicky nicky nine doors)</span></span>",
      v:"nicky",term:"knocking nine doors",grid:"prank_nicky"},
     {label:"Chap door run",v:"chap",term:"chap door run",grid:"prank_chap"},
     {label:"Chappie",v:"chappie",term:"chappie",grid:"prank_chappie"},
     {label:"Chicken mellie",v:"chickenmellie",term:"chicken mellie",grid:"prank_mellie"},
     // no grid: 6%% nationally in YouGov and graded by AGE, not region -- 25%% of
     // 18-29s against 2%% of over-70s -- so there is no place to point to
     {label:"Ding dong ditch",v:"ditch",term:"ding dong ditch",none:true},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_prank",excl:true,none:true}
   ]},
  {id:"pants",text:"&ldquo;Your &#95;&#95;&#95;&#95; are on backwards.&rdquo;",
   tag:"real data (Our Dialects, n=6,291)",real:true,phon:false,metric:"prevalence",
   info:"pants",infoLabel:"pants vs trousers",
   opts:[
     {label:"Trousers",v:"trousers",term:"trousers",grid:"trousers"},
     {label:"Pants",v:"pants",term:"pants",grid:"pants"}
   ]},
  {id:"them",text:"How natural does &ldquo;<i>Look at them animals</i>&rdquo; sound to you (for <i>those animals</i>)?",
   tag:"real data (Our Dialects, n=2,659)",real:true,metric:"pct",
   slider:true,grid:"them",sliderLabels:["Sounds wrong","Sounds fine"],
   info:"them",infoLabel:"demonstrative &lsquo;them&rsquo;"},
  {id:"sofa",text:"What do you call the long soft seat in your living room?",
   tag:"real data (Our Dialects, n=6,302)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"sofa",infoLabel:"sofa, settee or couch",
   opts:[
     {label:"Sofa",v:"sofa",term:"sofa",grid:"sofa_sofa"},
     {label:"Settee",v:"settee",term:"settee",grid:"sofa_settee"},
     {label:"Couch",v:"couch",term:"couch",grid:"sofa_couch"}
   ]},
  {id:"gum",text:"This is called &#95;&#95;&#95;&#95;",img:GUM_IMG,
   tag:"real data (Our Dialects, n=3,524)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"gum",infoLabel:"words for chewing gum",
   opts:[
     {label:"Chewing gum",v:"gum",term:"chewing gum",grid:"gum_gum"},
     {label:"Chewy",v:"chewy",term:"chewy",grid:"gum_chewy"},
     {label:"Chuddy",v:"chuddy",term:"chuddy",grid:"gum_chuddy"},
     {label:"Chud",v:"chud",term:"chud",grid:"gum_chud"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_gum",excl:true,none:true}
   ]},
  {id:"singerfinger",
   // "Do singer and finger rhyme?" was ambiguous -- the -er endings rhyme for
   // everyone in the loose sense, so speakers WITHOUT the feature said yes.
   // The finger contrast is what makes the question answerable at all, so it
   // stays; what goes is the claim that EVERY accent has a hard g there. Asking
   // whether singer matches the g "that you use in finger" compares the two
   // words inside one speaker's own accent, which is the actual variable and
   // asserts nothing about anybody else.
   text:"Does <i>singer</i> have the same hard <b>g</b> sound that <i>you</i> use in <i>finger</i>?",
   tag:"real data",real:true,metric:"pct",grid:"singerfinger",
   info:"singerfinger",infoLabel:"velar nasal plus",
   opts:[{label:"Yes &mdash; <i>sing-<b>g</b>er</i>, the same <b>g</b> as in <i>finger</i>",v:1,word:"pronounce the hard g"},
         {label:"No &mdash; <i>sing-er</i>, unlike <i>finger</i>",v:0,word:"drop the hard g"}]},
  {id:"alley",text:"What do you call the narrow walkway between or behind houses?",
   tag:"real data (Our Dialects, n=2,087)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"alley",infoLabel:"words for an alleyway",
   opts:[
     {label:"Alley / alleyway",v:"alley",term:"alley(way)",grid:"alley_alley"},
     {label:"Ginnel",v:"ginnel",term:"ginnel",grid:"alley_ginnel"},
     {label:"Snicket",v:"snicket",term:"snicket",grid:"alley_snicket"},
     {label:"Gennel",v:"gennel",term:"gennel",grid:"alley_gennel"},
     {label:"Jitty",v:"jitty",term:"jitty",grid:"alley_jitty"},
     {label:"Entry",v:"entry",term:"entry",grid:"alley_entry"},
     {label:"Cut",v:"cut",term:"cut",grid:"alley_cut"},
     {label:"Passage",v:"passage",term:"passage",grid:"alley_passage"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_alley",excl:true,none:true}
   ]},
  {id:"splinter",text:"What do you call a small piece of wood stuck in your skin?",tag:"",real:true,multi:true,metric:"prevalence",
   info:"splinter",infoLabel:"words for a splinter",
   opts:[
     {label:"Splinter",v:"splinter",term:"splinter",grid:"splinter"},
     {label:"Spelk",v:"spelk",term:"spelk",grid:"spelk"},
     {label:"Spell",v:"spell",term:"spell",grid:"spell"},
     {label:"Shiver",v:"shiver",term:"shiver",grid:"shiver"},
     {label:"Sliver",v:"sliver",term:"sliver",grid:"sliver"},
     {label:"Skelf",v:"skelf",term:"skelf",grid:"skelf"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_splinter",excl:true,none:true}
   ]},
  {id:"mother",text:"Growing up, what did you call your mother?",
   tag:"real data (Starkey Comics dialect survey)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"mother",infoLabel:"words for &lsquo;mother&rsquo;",
   opts:[
     {label:"Mum",v:"mum",term:"mum",grid:"mother_mum"},
     {label:"Mam",v:"mam",term:"mam",grid:"mother_mam"},
     {label:"Mom",v:"mom",term:"mom",grid:"mother_mom"},
     {label:"Mummy",v:"mummy",term:"mummy",grid:"mother_mummy"},
     {label:"Maw",v:"maw",term:"maw",grid:"mother_maw"},
     {label:"Mammy",v:"mammy",term:"mammy",grid:"mother_mammy"},
     {label:"Something else",v:"none",term:"another word",grid:"none_mother",excl:true,none:true}
   ]},
  {id:"shoes",text:"What did you call the soft canvas shoes you wore for PE at primary school?",
   tag:"real data (YouGov, 2025 &mdash; ~38,000 respondents)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"shoes",infoLabel:"names for PE plimsolls",
   opts:[
     {label:"Plimsolls",v:"plimsolls",term:"plimsolls",grid:"shoe_plimsolls"},
     {label:"Pumps",v:"pumps",term:"pumps",grid:"shoe_pumps"},
     {label:"Daps / dappers",v:"daps",term:"daps",grid:"shoe_daps"},
     {label:"Sandshoes / sannies",v:"sandshoes",term:"sandshoes",grid:"shoe_sandshoes"},
     {label:"Gym shoes",v:"gymshoes",term:"gym shoes",grid:"shoe_gymshoes"},
     {label:"Gutties",v:"gutties",term:"gutties",grid:"shoe_gutties"},
     {label:"Rubbers",v:"rubbers",term:"rubbers",grid:"shoe_rubbers"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_shoe",excl:true,none:true}
   ]},
  {id:"skiveclass",text:"What do you call skipping school without permission?",
   tag:"real data (BBC Voices, via Grieve et al. 2019)",real:true,phon:false,multi:true,metric:"prevalence",
   info:"skiveclass",infoLabel:"words for skipping school",
   opts:[
     {label:"Skive (off)",v:"skive",term:"skive",grid:"skiveclass_skive"},
     {label:"Bunk off",v:"bunk",term:"bunk off",grid:"skiveclass_bunk"},
     {label:"Wag (it)",v:"wag",term:"wag",grid:"skiveclass_wag"},
     {label:"Skip (it)",v:"skip",term:"skip",grid:"skiveclass_skip"},
     {label:"Play hookey",v:"hookey",term:"hookey",grid:"skiveclass_hookey"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_skiveclass",excl:true,none:true}
   ]},
  {id:"thfronting",text:"Would you ever pronounce &ldquo;th&rdquo; as an &ldquo;f&rdquo; or &ldquo;v&rdquo; (so <i>think</i> sounds like <i>fink</i>, <i>brother</i> like <i>bruvver</i>)?",
   tag:"real data",real:true,metric:"pct",grid:"thfronting",
   info:"thfronting",infoLabel:"th-fronting",
   opts:[{label:"Yes",v:1,word:"front their &lsquo;th&rsquo;s"},{label:"No",v:0,word:"keep &lsquo;th&rsquo;"}]},
  {id:"youse",text:"Would you ever call a group of two or more people &ldquo;<i>yous</i>&rdquo; or &ldquo;<i>youse</i>&rdquo;?",
   tag:"real data",real:true,metric:"pct",grid:"youse",
   info:"youse",infoLabel:"plural &lsquo;yous(e)&rsquo;",
   opts:[{label:"Yes",v:1,word:"say yous(e)"},{label:"No",v:0,word:"don&rsquo;t say yous(e)"}]},
  {id:"tag",text:"What do you call the playground game where one person chases the others?",tag:"real data (Starkey Comics dialect survey)",real:true,phon:false,multi:true,metric:"prevalence",
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
     {label:"Dob / dobby",v:"dobby",term:"dob",grid:"tag_dobby"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_tag",excl:true,none:true}
   ]},
  {id:"scone",text:"Does <i>scone</i> rhyme with <i>gone</i> or <i>bone</i>?",tag:"real data",real:true,metric:"pct",
   info:"scone",infoLabel:"how you say &lsquo;scone&rsquo;",
   opts:[{label:"Gone (&ldquo;skon&rdquo;)",v:1,word:"rhyme it with gone"},
         {label:"Bone / cone (&ldquo;skohn&rdquo;)",v:0,word:"rhyme it with bone"}]},
  {id:"forcecure",text:"Do <i>poor</i> and <i>pour</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"forcecure",infoLabel:"the cure&ndash;force merger",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (merged)"},{label:"No, they sound different",v:0,word:"keep them distinct"}]},
  {id:"northforce",text:"Do <i>horse</i> and <i>hoarse</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"northforce",infoLabel:"the north&ndash;force merger",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (merged)"},{label:"No, they sound different",v:0,word:"keep them distinct"}]},
  {id:"stirstare",text:"Do <i>stir</i> and <i>stare</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"nursesquare",infoLabel:"the nurse&ndash;square merger",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme stir/stare"},{label:"No, they sound different",v:0,word:"keep them distinct"}]},
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
  {id:"giveitme",text:"How natural does &ldquo;<i>Give it me</i>&rdquo; sound to you (for <i>give it to me</i>)?",tag:"real data",real:true,metric:"pct",
   slider:true,grid:"giveitme",sliderLabels:["Sounds wrong","Sounds fine"],info:"giveitme",infoLabel:"the &lsquo;give it me&rsquo; dative"},
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
     {label:"Softie",v:"softie",term:"softie",grid:"softie"},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_bread",excl:true,none:true}
   ]},
  // binary + the paper gives real proportions -> metric "pct"
  {id:"tvd",text:"What do you call your evening meal?",
   tag:"real data (Our Dialects, n=7,732)",real:true,phon:false,metric:"prevalence",
   info:"tvd",infoLabel:"tea, dinner or supper",
   opts:[
     {label:"Tea",v:"tea",term:"tea",grid:"meal_tea"},
     {label:"Dinner",v:"dinner",term:"dinner",grid:"meal_dinner"},
     {label:"Supper",v:"supper",term:"supper",grid:"meal_supper"}
   ]},
  {id:"trapbath",text:"Do <i>gas</i> and <i>grass</i> rhyme for you?",tag:"blended: BBC Future + English Dialect App",real:true,metric:"pct",
   info:"trapbath",infoLabel:"the trap&ndash;bath split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (short a)"},
         {label:"No, they sound different",v:0,word:"split (long a)"}]},
  {id:"bookspook",text:"Do <i>book</i> and <i>spook</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"bookspook",infoLabel:"book vs spook",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme book/spook"},{label:"No, they sound different",v:0,word:"don&rsquo;t rhyme"}]},
  // metric "pct": a clean binary the paper reports as proportions -> show a percent.
  {id:"footstrut",text:"Do <i>foot</i> and <i>cut</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"footstrut",infoLabel:"the foot&ndash;strut split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme"},{label:"No, they sound different",v:0,word:"split"}]}
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
  softie:"<b>softie</b> &mdash; a North East Scotland (Doric) word, from the same root as <i>soft</i>: a plain, soft-crusted white roll, as opposed to a crustier <i>bap</i>. Strongest on the Aberdeenshire coast.",
  footstrut:"<b>The FOOT&ndash;STRUT split</b> &mdash; in the 17th century the Middle English short <i>u</i> /&#650;/ split, in southern England, into two vowels: /&#650;/ (FOOT) and a new unrounded /&#652;/ (STRUT). The split never reached most of the North &amp; Midlands, where <i>foot</i> and <i>strut</i> still share /&#650;/ and rhyme.",
  tvd:"<b>Tea, dinner or supper</b> &mdash; the evening meal, and one of the sharpest north/south splits in Britain. <b>Tea</b> is northern: 80%% in Westmorland and 77%% in Yorkshire, against 4%% in Middlesex. <b>Dinner</b> runs the other way, near 90%% across Essex and Hertfordshire. <b>Supper</b> is a distant third nationally but reaches 23%% in Oxfordshire, and carries a class edge as well as a regional one.",
  bookspook:"<b>book vs spook</b> &mdash; in some accents the <i>-ook</i> words (book, cook, look) keep the old long vowel /u&#720;/, so <i>book</i> is [bu&#720;k] and rhymes with <i>spook</i> &mdash; putting it in the GOOSE set rather than FOOT. Traditional in the North East and Stoke (and once Liverpool); Scotland has no foot&ndash;goose distinction at all.",
  nursesquare:"<b>The NURSE&ndash;SQUARE merger</b> &mdash; in some accents the vowels of NURSE (<i>stir, fur</i>) and SQUARE (<i>stare, fair</i>) fall together, so <i>stir</i> and <i>stare</i> rhyme. Best documented in Liverpool and the North West, and now strong &mdash; and apparently spreading &mdash; along the east coast, in Hull and on Teesside.",
  scone:"<b>scone</b> &mdash; the great teatime shibboleth: does it rhyme with <i>gone</i> or with <i>bone</i>? Most of Britain &mdash; Scotland and the North especially &mdash; rhymes it with <i>gone</i>. The <i>bone</i> pronunciation is the local norm in the <b>East Midlands</b>, with the far South West leaning that way too.",
  northforce:"<b>The NORTH&ndash;FORCE merger</b> &mdash; whether <i>horse</i> and <i>hoarse</i> (or <i>for</i> and <i>four</i>, <i>war</i> and <i>wore</i>) sound identical. Most of England and Wales merged them long ago, so they rhyme; <b>Scotland</b> keeps them clearly distinct, as do pockets around <b>Manchester</b> and Merseyside.",
  forcecure:"<b>The CURE&ndash;FORCE merger</b> &mdash; whether <i>poor</i> and <i>pour</i> (or <i>sure</i> and <i>shore</i>, <i>tour</i> and <i>tore</i>) sound identical. Across most of England they have merged, so they rhyme; the older distinct <i>poor</i>/<i>sure</i> vowel /&#650;&#601;/ survives in <b>Scotland</b>, the <b>North East</b>, and <b>West Yorkshire</b>.",
  youse:"<b>Plural &lsquo;yous(e)&rsquo;</b> &mdash; a second-person plural pronoun, filling the gap English left when <i>thou/you</i> collapsed to just <i>you</i>. Strongest in <b>Scotland</b> and the <b>North East</b>, fading through the Midlands, and rare in southern England.",
  mother:"<b>Words for &lsquo;mother&rsquo;</b> &mdash; <b>mum</b> is general across England and Scotland. <b>Mam</b> is the Welsh, North East and Cumbrian word. <b>Mom</b> is almost entirely <b>West Midlands</b>, centred on Birmingham &mdash; not an Americanism there but long-standing local usage. <b>Maw</b> is a Scots clipping, <b>mummy</b> survives in adults mainly in the South East, and <b>mammy</b> has a real foothold in south-west Wales.",
  shoes:"<b>Names for PE plimsolls</b> &mdash; the canvas shoes worn for primary-school PE, and one of the most sharply regional words in Britain. <b>Plimsolls</b> is the southern and eastern norm (91%% in Norfolk) but flips to <b>pumps</b> across the North West and West Midlands. <b>Daps</b> clusters either side of the Severn Estuary. Scotland splits several ways: <b>sandshoes</b> around the Clyde, <b>gutties</b> in Lanarkshire, <b>rubbers</b> almost only in the Lothians.",
  singerfinger:"<b>Velar nasal plus</b> &mdash; whether a hard [&#609;] survives after the <i>ng</i>, so <i>singer</i> rhymes with <i>finger</i>. English generally dropped it around the 17th century, but the change never took hold across the <b>North West</b> and <b>West Midlands</b>: Manchester, Liverpool, Stoke, Birmingham and north-east Wales keep it. Elsewhere the two words are distinct.",
  thfronting:"<b>TH-fronting</b> &mdash; replacing /&#952;/ and /&#240;/ with /f/ and /v/, so <i>think</i> &rarr; <i>fink</i> and <i>brother</i> &rarr; <i>bruvver</i>. Once a London feature, it has spread rapidly since the late 20th century across urban England, especially among younger speakers, while staying rare in Scotland and Wales.",
  skiveclass:"<b>Words for skipping school</b> without permission. <b>Skive</b> is the general British term, strongest in Scotland and the South West; <b>bunk off</b> is a London/South East form; <b>wag</b> belongs to the North West and North East; <b>play hookey</b> is chiefly Tyneside.",
  trapbath:"<b>The trap&ndash;bath split</b> &mdash; in the 18th century southern English lengthened the <i>a</i> in a set of words (<i>bath, grass, last, dance</i>) to /&#593;&#720;/, splitting them from TRAP words (<i>cat, trap</i>). The North, Wales and Scotland kept the short /a/ &mdash; so a northerner says [ba&#952;], a southerner [b&#593;&#720;&#952;]. It&rsquo;s one of the sharpest north&ndash;south markers.",
  splinter:"<b>Words for a splinter</b> of wood in the skin. <b>Splinter</b> is the nationwide standard; <b>spelk</b> (from Old English <i>spelc</i>) belongs to the North East and the Borders; <b>spell</b> is northern; <b>shiver</b> is East Anglian; <b>sliver</b> is a South East word; and <b>skelf</b> is the Scots term.",
  giveitme:"<b>&lsquo;Give it me&rsquo;</b> &mdash; the theme (<i>it</i>) before the goal (<i>me</i>) with no <i>to</i>. A North West &amp; Midlands feature, strongest around Manchester and the Potteries, thinning towards the North East and the South.",
  lolly:"<b>Ice lolly vs lolly ice</b> &mdash; <i>ice lolly</i> is the standard British term; <i>lolly ice</i>, the words reversed, is the Merseyside form. Further afield you&rsquo;ll hear <i>ice pop</i> (Ireland, Scotland) or <i>popsicle</i> (North America)."
,
  sofa:"<b>Sofa, settee or couch</b> &mdash; <b>sofa</b> is the majority term nationally (58%%) and dominant across the South. <b>Settee</b> is the northern and Midlands word. <b>Couch</b> looks national but is really two places: Merseyside and west Lancashire, where it is overwhelming (76%% in Wigan), and Scotland at 41%%. Blackburn and Wigan, thirty miles apart, are near-exact opposites.",
  gum:"<b>Words for chewing gum</b> &mdash; <b>chewing gum</b> is what four in five Britons say. Three local words survive inside that: <b>chewy</b> on Merseyside (78%% in Liverpool), <b>chuddy</b> along the Pennines from Manchester across to Leeds, and <b>chud</b>, which is Newcastle&rsquo;s alone. The three barely overlap, which is unusual even among lexical variants.",
  prank:"<b>Words for knock-a-door-run</b> &mdash; one of the sharpest splits in Britain: four countries&rsquo; worth of words on one island. The <b>run</b> names (<i>knock a door run</i>, <i>knock and run</i>) cover the North and Midlands &mdash; 97%% in Lancashire. The South says <b>knock down ginger</b> (63%% in London). Scotland has its own word entirely: half of Scots say <b>chap door run</b> or the clipped <b>chappie</b>, from Scots <i>chap</i> &lsquo;to knock&rsquo;. Two smaller English words survive inside that: <b>cherry knocking</b> around Gloucester, and <b>knocking nine doors</b> on Tyneside.",
  pants:"<b>Pants or trousers</b> &mdash; in most of Britain <i>pants</i> means underwear; across the North West it means both. 55%% in Lancashire and 39%% in Cheshire, falling to 14%% in Yorkshire and 2%% in Staffordshire. The cities are sharper still: <b>Liverpool 58%%</b> and <b>Manchester 50%%</b> against <b>Stoke 3%%</b>, thirty miles down the road.",
  them:"<b>Demonstrative &lsquo;them&rsquo;</b> &mdash; <i>them animals</i> for <i>those animals</i>: the object pronoun used as a determiner. It has been in English since the Middle Ages rather than being a recent error. Acceptance runs north to south, highest in Yorkshire and Lancashire and lowest in the South &mdash; and lower still in Scotland, which has its own demonstratives (<i>thae</i>, <i>thon</i>).",
  alley:"<b>Words for an alleyway</b> &mdash; one of the most finely divided words in Britain. <b>Alley</b> is the national default and almost the only word in the south. The north fragments: <b>ginnel</b> across Lancashire and West Yorkshire, <b>snicket</b> around Bradford, <b>gennel</b> around Sheffield, <b>jitty</b> through Derby and Nottingham, <b>entry</b> on Merseyside, <b>cut</b> in Newcastle. The sharpest divide is Bradford against Leeds, ten miles apart.",
  tag:"<b>Names for tag/it</b> &mdash; <i>tig</i> covers most of England, Scotland &amp; Wales; <i>it</i> is the South East&rsquo;s word instead. Local pockets survive: <i>tiggy</i> and <i>tuggy</i> side by side around Durham, <i>tick</i> in North Wales, <i>touch</i> in South Wales, <i>had</i> on the Suffolk/Essex coast, and <i>dobby</i> in a tight pocket around Sheffield."
};
// only etymology sources are cited (the maps are our own recreations, not originals)
// Shown as "Source: ..." at the foot of each info bubble. Deliberately left
// empty for questions whose provenance I could not verify -- an unattributed
// bubble is better than a citation that might be wrong.
// External reading, verified 2026-08-10. Wikipedia for the phenomenon itself,
// plus the study behind the map where the provenance is actually known. Left
// empty for questions I can't source -- a dead or wrong link on a page arguing
// for its own rigour is worse than no link at all.
// "Read more" only -- links to the phenomenon itself, not to the study behind the
// map. Sources are credited once, on the results screen, rather than repeated in
// nineteen bubbles.
const INFO_LINKS={
  footstrut:[["the foot&ndash;strut split","https://en.wikipedia.org/wiki/Foot%%E2%%80%%93strut_split"]],
  trapbath:[["the trap&ndash;bath split","https://en.wikipedia.org/wiki/Trap%%E2%%80%%93bath_split"]],
  thfronting:[["th-fronting","https://en.wikipedia.org/wiki/Th-fronting"]],
  singerfinger:[["ng-coalescence","https://en.wikipedia.org/wiki/Ng-coalescence"]],
  youse:[["plural <i>yous</i>","https://en.wikipedia.org/wiki/Yous"]],
  scone:[["the scone debate","https://en.wikipedia.org/wiki/Scone"]],
  shoes:[["the plimsoll","https://en.wikipedia.org/wiki/Plimsoll_(shoe)"]],
  skiveclass:[["playing hooky","https://en.wikipedia.org/wiki/Hooky"]],
  lolly:[["ice pops","https://en.wikipedia.org/wiki/Ice_pop"]],
  tag:[["tag","https://en.wikipedia.org/wiki/Tag_(game)"]],
  alley:[["ginnel","https://en.wikipedia.org/wiki/Ginnel"]],
  bread:[["the bread roll","https://en.wikipedia.org/wiki/Bread_roll"]]
};
// Every source in one place, shown on the results screen rather than repeated in
// each info bubble. Only sources I can actually stand behind are named; the rest
// of the maps are covered by the general line.
// One consolidated credit list behind an (i) on the results screen. Names only:
// pairing each source to a specific question implied the other fifteen were
// unsourced, which is not what a partial list means.
// Every source, with a link to the thing itself. A named citation with no way to
// reach it is not much use to a reader who wants to check the map against it.
const A=(t,u)=>"<a href='"+u+"' target='_blank' rel='noopener noreferrer'>"+t+"</a>";
const SOURCES=[
  "<i>Our Dialects</i> &mdash; L. MacKenzie, G. Bailey &amp; D. Turton, "+
    A("ourdialects.uk","https://www.ourdialects.uk/maps/")+", &copy; George Bailey, "+
    A("CC BY-SA 4.0","https://creativecommons.org/licenses/by-sa/4.0/"),
  "MacKenzie, Bailey &amp; Turton (2022), &lsquo;Towards an updated dialect atlas of British English&rsquo;, "+
    "<i>Journal of Linguistic Geography</i> &mdash; "+
    A("full text (PDF)","https://www.laurelmackenzie.com/publication/2022-mackenzie-et-al/2022-mackenzie-et-al.pdf"),
  "YouGov, August 2025 (n&asymp;38,000) &mdash; "+
    A("what Britons call school canvas trainers","https://yougov.com/en-gb/articles/52768-plimsolls-pumps-or-something-else-what-do-britons-call-school-canvas-trainers"),
  "YouGov, February 2025 (n&gt;12,000) &mdash; "+
    A("&lsquo;knock down ginger&rsquo; or &lsquo;knock a door run&rsquo;?","https://yougov.com/en-gb/articles/51544-is-it-knock-down-ginger-or-knock-a-door-run"),
  "Grieve, Montgomery, Nini, Murakami &amp; Guo (2019), &lsquo;Mapping Lexical Dialect Variation in "+
    "British English Using Twitter&rsquo;, <i>Frontiers in Artificial Intelligence</i> &mdash; "+
    A("open access","https://doi.org/10.3389/frai.2019.00011"),
  "Tweetolectology Twitter survey (2020&ndash;21) &mdash; "+A("tweetolectology.com","https://tweetolectology.com/"),
  "Starkey Comics dialect surveys &mdash; "+A("starkeycomics.com","https://starkeycomics.com/2023/11/07/map-of-british-english-dialects/"),
  "Survey of English Dialects (Orton et al., 1978) &mdash; "+
    A("about the survey","https://en.wikipedia.org/wiki/Survey_of_English_Dialects")
];
// lexical prevalence: a relative band, no misleading headcount
function band(v){return v>=0.5?"the main word(s) here":v>=0.3?"common here":v>=0.15?"one of several here":"rarely used here";}
// for the "no word" negative map: high v = the words are absent here
function bandNone(v){return v>=0.5?"few people have a word":v>=0.3?"a word is less usual":v>=0.15?"most people have a word":"nearly everyone has a word";}
let idx=0; const answers={}; const revealedSet=new Set();
const cv=document.getElementById("cv"),cx=cv.getContext("2d");cv.width=W*CELL;cv.height=H_*CELL;
// Size the canvas so one grid cell == a whole number of DEVICE pixels. The
// stylesheet still decides how much room the map gets (52vh / 86vw); we read
// that budget, then snap DOWN to the nearest exact fit, so the map never
// overflows its column but the gutters are always pixel-crisp and identical.
// Clears the canvas, so call it immediately before drawing.
function fitCanvas(){
  const dpr=window.devicePixelRatio||1;
  cv.style.width="";cv.style.height="";                 // hand sizing back to the CSS
  cv.width=W*BASE_CELL;cv.height=H_*BASE_CELL;          // provisional: fixes the aspect
  const r=cv.getBoundingClientRect();
  if(!r.width||!r.height) return;
  // The canvas sits in a flex column sized to its own content, so the column's
  // max-width never actually constrains it -- at tall viewports the map grew to
  // its natural 470px and hung out of a 360px column. Clamp to it explicitly.
  // NB only a px max-width is a real constraint. On phones the rule is
  // max-width:100%%, and parseFloat("100%%") is 100 -- which silently shrank the
  // map to a 94px postage stamp. Require the unit before trusting the number.
  let budgetW=r.width;
  const mws=getComputedStyle(document.getElementById("right")).maxWidth;
  if(mws.slice(-2)==="px"){ const mw=parseFloat(mws);
    if(isFinite(mw)&&mw>0) budgetW=Math.min(budgetW,mw); }
  const k=Math.max(2, Math.floor(Math.min(budgetW/W, r.height/H_)*dpr));
  // Below 3px a cell cannot carry a gutter and a fill: 1px of each is a
  // checkerboard, not a map. Drop the gutter entirely there and let it render as
  // a clean solid surface -- the pixel-art styling simply needs 3px to exist.
  CELL=k; PXS=k/BASE_CELL;
  GAP=(k>=3)?Math.max(1, Math.round(k*BASE_GAP/BASE_CELL)):0;
  GW=GAP?Math.min(1, (k*BASE_GAP/BASE_CELL)/GAP):0;   // 1px gutter, dialled down when too wide
  cv.width=W*k; cv.height=H_*k;
  cv.style.width=(W*k/dpr)+"px"; cv.style.height=(H_*k/dpr)+"px";
}
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
  applyLayout();
  // drop any tooltip left over from the previous map -- moving on used to strand
  // e.g. 'Newcastle "spelk"' on top of the next question's map
  {const _t=document.getElementById("tip"); if(_t)_t.style.opacity=0;}
  const atEnd=idx>=QUESTIONS.length;
  // hometown is a pre-question: the bar only starts moving with the real questions
  document.getElementById("pbar").style.width=(atEnd?100:idx/(QUESTIONS.length-1)*100)+"%%";
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
      // always the next two places down the ranking. There used to be a 0.7x
      // similarity cut here, which meant a sharply localised result showed only
      // one runner-up (or none) and the row looked broken rather than decisive.
      const runners=scored.filter(s=>s.p!==cs.place).slice(0,2).map(s=>s.p.name);
      const least=scored.length?scored[scored.length-1].p:null;   // bottom of the same ranking = the blue end of the map
      const ms=matchScore(cs.place);
      pingEvent("complete");         // funnel counter: fires regardless of consent
      submitResponses(top, ms);      // the answers themselves: consent + complete run only
      done.innerHTML=
        "<p class='res-eyebrow'>You sound most like</p>"+
        "<div class='res-name'>"+top+"</div>"+
        (ms!=null?"<div class='res-gauge'>"+
            "<div class='glab'>Accent match</div>"+
            "<div class='track'><div class='fill' style='width:"+ms+"%%'></div></div>"+
            "<div class='gcap'>"+matchLabel(ms)+"</div>"+
          "</div>":"")+
        ((runners.length||least)?"<div class='res-rows'>"+
          (runners.length?"<div class='res-row'><span class='k'>Also close</span><span class='v'>"+runners.join(", ")+"</span></div>":"")+
          (least?"<div class='res-row'><span class='k'>Least like</span><span class='v' style='color:"+rgbOf(THEME.blue)+"'>"+placeName(least)+"</span></div>":"")+
        "</div>":"")+
        "<div class='res-foot'>Based on your "+cs.count+" answers<span class='dot'>&middot;</span>"+
          "<span class='srcwrap'><span class='srcbtn' id='srcbtn'>&#9432; sources</span>"+
          "<span class='srcpop'>Maps are redrawn from published dialect research and large-scale surveys, including:"+
          "<ul>"+SOURCES.map(x=>"<li>"+x+"</li>").join("")+"</ul></span></span></div>";
    }
    return;
  }
  const q=QUESTIONS[idx];
  document.getElementById("left").classList.toggle("wide",
    !q.hometownq && Array.isArray(q.opts) && q.opts.length>7);
  prog.textContent=QUESTIONS[idx].hometownq?"Before we start":"Question "+idx+" of "+(QUESTIONS.length-1);
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
    // hometown is required: you must either name a town or explicitly say you
    // didn't grow up in GB. Continue stays disabled until one of those is true,
    // so the question can't be skipped past on the way into the quiz.
    const htIn=document.getElementById("hometown");
    const htOK=()=>answers.hometown==="notgb" || (htIn && htIn.value.trim().length>0);
    const syncNext=()=>{ next.disabled=!htOK(); };
    if(htIn){ htIn.addEventListener("input",syncNext); htIn.addEventListener("picked",syncNext); }
    next.style.display="block"; next.textContent=contLabel; syncNext();
    next.onclick=()=>{ if(!htOK())return; leaveHometown(); idx++; render(); };
    return;
  }
  const answered=q.multi?(Array.isArray(ans)&&ans.length>0):(ans!==undefined);
  const isRevealed=revealedSet.has(q.id) && answered;
  // ---- options: ALWAYS editable (so Back lets you change answers), BUT changing the
  // selection hides the map — you only ever see it by pressing "See map". ----
  if(q.slider){
    // 1-5 acceptability slider
    // Once the map is up the hint line is free, so it says what to do next: the
    // maps are per-answer, and nothing else on screen tells you that moving the
    // slider redraws it.
    hint.innerHTML=isRevealed?"&#8635; Move the slider to see a different map."
                             :"Drag the slider, then press &ldquo;See map&rdquo;.";
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
    hint.innerHTML=isRevealed?"&#8635; Select a different option to see its map."
                             :"Select all that apply";
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
    hint.innerHTML=isRevealed?"&#8635; Select a different option to see its map.":"";
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

// ---- theme ----------------------------------------------------------------
// Canvas pixels can't inherit CSS variables, so every colour the maps draw with
// lives here and is swapped wholesale when the theme flips. The diverging scale
// in particular cannot just be reused: its midpoint is near-white, which would
// glow against a dark page, so dark mode gets its own dark neutral.
const THEME={grid:"#c9c9d2", nullCell:[214,214,220], nullDot:[200,200,205], dotStroke:"#2b2b2b",
      halo:"#ffffff", star:[214,16,32],
      blue:[18,86,222], mid:[232,232,236], red:[214,16,32],
      // Landing-map base tint. desat = how far each region is pulled toward its
      // own grey; wash = how far the result is then blended into the page. The
      // old pair of constants kept only 16%% of each colour channel under a flat
      // +112 lift, which is why every region came out near-grey and the map read
      // as a pale wash instead of a coloured atlas.
      desat:0.18, wash:0.34, page:[246,241,231], veil:0.58};
const rgbOf=a=>"rgb("+a[0]+","+a[1]+","+a[2]+")";
// Gutter strength. A cell can only ever be a whole number of device pixels, so
// at small sizes the thinnest drawable gutter (1px) is proportionally far too
// wide -- a third of a 3px cell instead of the intended fifth -- and the map
// reads as scattered dots rather than pixel blocks. We cannot draw a 0.6px line
// crisply, so we draw a 1px line at 60%% strength: same visual weight, still
// perfectly sharp. GW is 1 (a plain grid-coloured gutter) whenever the cell size
// is an exact multiple of 5 and no correction is needed.
const _gridRGB=(()=>{const h=THEME.grid.replace("#","");
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];})();
// The landing map's resting colour for a region. Each one is desaturated a
// little, then washed toward a target until it sits a guaranteed distance from
// its own vivid colour -- so the tour highlight reads as strongly on the pale
// pinks as on the deep blues. Light regions mute DOWNWARD toward a soft grey and
// dark ones UPWARD toward the page: washing a pale pink into a pale page barely
// moves it, which is what left Manchester's highlight almost invisible.
const MINPOP=70, WASHCAP=0.62;
// The spotlight colour for the region the tour is on. Several colours in the
// source palette are so pale that "full colour" barely reads as lit up at all --
// Mancunian is [227,200,226], which is very nearly the page. Saturate every
// region away from its own grey, and pull the light ones down in luminance, so
// each one actually switches on when the tour reaches it.
function vividTint(col){
  const g=col[0]*0.3+col[1]*0.5+col[2]*0.2;
  let out=col.map(t=>Math.round(Math.max(0,Math.min(255,g+(t-g)*1.45))));
  // Hard luminance ceiling. The veiled surround sits around 210-230, so a
  // spotlight has to be decisively darker than that or it vanishes into it --
  // at a ceiling of 170 the pale Mancunian pink still collided with veiled Wales.
  const lum=out[0]*0.3+out[1]*0.5+out[2]*0.2;
  if(lum>140){const k=140/lum; out=out.map(t=>Math.round(t*k));}
  return out;
}
function baseTint(col){
  const vivid=vividTint(col);
  const g=col[0]*0.3+col[1]*0.5+col[2]*0.2;
  // Always wash toward the page. An earlier version muted light regions DOWNWARD
  // instead, back when the spotlight was the raw palette colour and washing a
  // pale pink into a pale page barely moved it. Now that vividTint pulls light
  // regions down, muting them down too made base and spotlight converge --
  // Manchester's separation fell to 27. Base up, spotlight down, maximum gap.
  const tgt=THEME.page;
  const d=[0,1,2].map(i=>col[i]*(1-THEME.desat)+g*THEME.desat);
  let w=THEME.wash, out=[0,1,2].map(i=>Math.round(d[i]*(1-w)+tgt[i]*w));
  // measured against the VIVID colour, which is what actually gets drawn on top
  while(w<WASHCAP && Math.hypot(out[0]-vivid[0],out[1]-vivid[1],out[2]-vivid[2])<MINPOP){
    w=Math.min(WASHCAP,w+0.04);
    out=[0,1,2].map(i=>Math.round(d[i]*(1-w)+tgt[i]*w));
  }
  return out;
}
// Darken a region colour until its text clears a contrast ratio against the
// page. 13 of the 33 region names failed WCAG AA as raw palette colours, and
// the palest were at 1.4:1 -- legible only if you already knew what it said.
function readableOn(col,minR,against){
  const lin=v=>{v/=255;return v<=0.03928?v/12.92:Math.pow((v+0.055)/1.055,2.4);};
  const L=c=>0.2126*lin(c[0])+0.7152*lin(c[1])+0.0722*lin(c[2]);
  const bg=L(against||THEME.page);
  const ratio=c=>{const l=L(c);return (Math.max(l,bg)+0.05)/(Math.min(l,bg)+0.05);};
  let out=col.slice();
  for(let i=0;i<28 && ratio(out)<minR;i++) out=out.map(t=>Math.round(t*0.90));
  return out;
}
function gutterCol(r,g,b,w){
  const k=(w===undefined)?GW:w;
  if(k>=1) return THEME.grid;
  return "rgb("+Math.round(r+(_gridRGB[0]-r)*k)+","+
                Math.round(g+(_gridRGB[1]-g)*k)+","+
                Math.round(b+(_gridRGB[2]-b)*k)+")";
}
function heatBar(){return "linear-gradient(to right,"+rgbOf(THEME.blue)+","+rgbOf(THEME.mid)+","+rgbOf(THEME.red)+")";}
function heat(t){t=Math.max(0,Math.min(1,t));
  // stretch contrast around the midpoint so the maps read boldly instead of washing out
  t=Math.max(0,Math.min(1, 0.5+(t-0.5)*1.55));
  const blue=THEME.blue,white=THEME.mid,red=THEME.red;
  const mix=(A,B,k)=>[Math.round(A[0]+(B[0]-A[0])*k),Math.round(A[1]+(B[1]-A[1])*k),Math.round(A[2]+(B[2]-A[2])*k)];
  return t<0.5?mix(blue,white,t/0.5):mix(white,red,(t-0.5)/0.5);}

// Where does this answer concentrate? Mean surface per region; if one/a few
// regions clearly stand out, name them (grouping adjacent ones); if it's
// scattered or nothing stands out, say "multiple regions".
function joinRegions(a){return a.length<=1?a[0]:a.slice(0,-1).join(", ")+" &amp; "+a[a.length-1];}
// score each representative place by the surface value around it, most-representative first
// ---- accent match score: how "local" your answer set is for a given place ----
// For each answered question: how common is YOUR answer there, as a fraction of
// the most common answer there. Averaged over questions this reads as "you gave
// the local answer this much of the time" — a correlation-style 0-100 fit that
// is comparable between places (a place where every variant is 50/50 can still
// score 100, because the denominator is that place's own best answer).
// Raw fit is scored for EVERY place, then the winner is expressed against the
// all-places average. Raw fit alone has a high floor (a random answer set still
// scores about two-thirds, as most variants are common in most places), which
// would make every result look like a strong match. Measuring the winner's
// headroom above the average place is self-calibrating and needs no constants.
function matchScores(){
  const perQ=[];
  for(const q of QUESTIONS){
    if(q.hometownq)continue;
    const a=answers[q.id]; if(a===undefined)continue;
    const mine=answerSurface(q,a); if(!mine)continue;
    const alts=[];
    const vals=q.slider?[1,2,3,4,5]:q.opts.map(o=>o.v);
    for(const v of vals){ const s=answerSurface(q,q.multi?[v]:v); if(s)alts.push(s); }
    perQ.push({mine:mine,alts:alts});
  }
  if(!perQ.length) return null;
  const out=new Map();
  for(const p of PLACES){
    const R=p.row|0,C=p.col|0;
    const valAt=s=>{ let t=0,n=0;
      for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){const r=R+dr,c=C+dc;
        if(r<0||r>=H_||c<0||c>=W)continue; const v=s[r]?s[r][c]:null; if(v!=null){t+=v;n++;}}
      return n?t/n:null; };
    let sum=0,n=0;
    for(const e of perQ){
      const mv=valAt(e.mine); if(mv==null)continue;
      let best=mv;
      for(const s of e.alts){ const t=valAt(s); if(t!=null&&t>best)best=t; }
      if(best>1e-6){ sum+=mv/best; n++; }
    }
    out.set(p, n?sum/n:0);
  }
  return out;
}
function matchScore(place){
  if(!place) return null;
  const fits=matchScores(); if(!fits) return null;
  const raw=fits.get(place); if(raw==null) return null;
  const all=[...fits.values()];
  const base=all.reduce((a,b)=>a+b,0)/all.length;
  const headroom=Math.max(1e-6, 1-base);
  return Math.max(0, Math.min(100, Math.round((raw-base)/headroom*100)));
}
function matchLabel(s){
  if(s>=85)return "You sound like a local";
  if(s>=70)return "A strong match";
  if(s>=55)return "A good match";
  if(s>=40)return "A partial match";
  return "A mixed accent &mdash; no single home";
}
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
  // zones() names the hot regions, collapsing fine ones into big zones.
  const nameZones=(hot)=>{
    const zones=[]; const used=new Set();
    const grp=(set,label)=>{const inh=set.filter(n=>hot.includes(n));
      if(inh.length>=2){ zones.push([label, Math.max.apply(null,inh.map(n=>mean[n]))]); set.forEach(n=>used.add(n)); }};
    grp(NORTHSET,"the North of England"); grp(SOUTHSET,"the South of England"); grp(MIDS,"the Midlands");
    hot.filter(n=>!used.has(n)).forEach(n=>zones.push([n,mean[n]]));
    zones.sort((a,b)=>b[1]-a[1]);
    const nm=zones.map(z=>z[0]);
    if(nm.includes("the North of England") && nm.includes("the South of England")) return "much of Britain";
    return nm.length?joinRegions(nm.slice(0,3)):null;
  };
  if(topMean>=0.40){
    const r=nameZones(regionNames.filter(n=>mean[n]>=0.6*topMean && mean[n]>=0.32));
    return r||"several regions";
  }
  // a sharp LOCAL peak in an otherwise-cool region (muffin=Manchester, batch=Coventry)
  if(peakVal>=0.45 && meanA[peakReg]<0.35) return regionNames[peakReg];
  // LAST resort, reached only where the two branches above already gave up. A word
  // can be low everywhere and still unmistakably belong somewhere: "knock knock
  // ginger" is 16%% in the South West and Wales and ~1%% elsewhere -- sharply
  // regional, but nowhere near the 0.40/0.32 absolute bars, so it used to fall
  // through to the vague "several regions". Judge those on the RATIO to the
  // national mean instead of the level, the same way the star floor does. Each
  // named region must also clear twice the national mean, or cross-border bleed
  // gets named alongside the real thing.
  const allMean=(()=>{let s=0,n=0;
    for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const v=surf[r]?surf[r][c]:null; if(v!=null){s+=v;n++;}}
    return n?s/n:0;})();
  if(allMean>0 && topMean/allMean>=2.5){
    const r=nameZones(regionNames.filter(n=>mean[n]>=0.6*topMean && mean[n]>=2.0*allMean));
    if(r) return r;
  }
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
  if(strokeW){cx.lineWidth=strokeW;cx.strokeStyle=THEME.dotStroke;cx.stroke();}
}
function drawMap(q,ans){
  document.getElementById("right").style.display="";   // map revealed -> show the right panel
  // On desktop the photo and the map share the right column, so revealing the map
  // has to hide the photo. On a phone they are in different places (photo sits up
  // with the question), so it should stay visible.
  {const _im=document.getElementById("qimg");
   if(MQ_NARROW.matches && q.img){ if(!_im.getAttribute("src")) _im.src=q.img; _im.style.display="block"; }
   else _im.style.display="none";}
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
      (q.metric==="pct"?"Hover over a city to see the approximate percentage"
                       :"Hover over a city to see the word most likely used there")+"</span>";
  }
  fitCanvas();                       // snap to whole device pixels before painting
  cx.clearRect(0,0,cv.width,cv.height);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){
    if(!land[r][c])continue;
    const v=surf[r][c];
    const [rr,gg,bb]=(incon||v==null)?THEME.nullCell:hcol(v);
    cx.fillStyle=gutterCol(rr,gg,bb);cx.fillRect(c*CELL,r*CELL,CELL,CELL);
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
  // one threshold, used by BOTH the "is this localised?" gate and the per-place
  // filter. They used to be 0.18 and 0.20, so a surface peaking between the two
  // passed the gate and then had every candidate rejected, silently degrading to
  // the vague "several regions" text (this is what hid mummy -> London).
  const STAR_MIN=0.18;
  // A term can be rare nationally and still unmistakably belong somewhere:
  // "rubbers" is only 18%% even in the Lothians, but 18x the national mean and
  // near-absent elsewhere. Judging that on absolute level alone hid it, so a
  // sufficiently concentrated answer qualifies at a lower floor.
  const FLOOR = (meanV>0 && topV/meanV>=6) ? 0.08 : STAR_MIN;
  const localised = starAllowed && topV>FLOOR && meanV>0 && (topV/meanV)>=2.4;
  const starPlaces=[];
  if(localised) for(const s of scored){ if(starPlaces.length>=3)break;
    if(s.v<0.8*topV||s.v<FLOOR)continue;
    // 9 cells is ~90km and swallowed genuinely separate dialect centres:
    // Manchester sits 7.8 cells from Leeds and suppressed it entirely. 6 keeps
    // one-conurbation pairs together (Leeds/Bradford are 2 apart) while letting
    // Manchester and Leeds both star, which is what the survey describes.
    if(starPlaces.some(p=>Math.hypot(p.col-s.p.col,p.row-s.p.row)<6))continue;
    starPlaces.push(s.p);}
  SHOWN.shownPlaces=starPlaces;   // so the stars are hoverable too
  const starKey=new Set(starPlaces.map(p=>p.col+","+p.row));
  // ordinary cities as circles (skip any spot that will get a star)
  for(const ct of cities){ if(starKey.has(ct.col+","+ct.row))continue;
    const v=surf[ct.row|0]?surf[ct.row|0][ct.col|0]:null;
    const [rr,gg,bb]=(incon||v==null)?THEME.nullDot:hcol(v);
    cx.beginPath();cx.arc((ct.col+0.5)*CELL,(ct.row+0.5)*CELL,5*PXS,0,7);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fill();cx.lineWidth=2*PXS;cx.strokeStyle=THEME.dotStroke;cx.stroke();}
  // representative places as bold, red-shaded stars with a white halo (stand out clearly)
  for(const p of starPlaces){const v=surf[p.row]?surf[p.row][p.col]:null;
    const [rr,gg,bb]=(v==null)?THEME.nullDot:hcol(v);
    const x=(p.col+0.5)*CELL,y=(p.row+0.5)*CELL;
    drawStar(x,y,16*PXS,THEME.halo,0);                          // white halo
    drawStar(x,y,12*PXS,"rgb("+rr+","+gg+","+bb+")",3*PXS);}    // the star
  // legend reflects the scale in use: word maps run pale->red; yes/no maps run blue->red
  {const lg=document.getElementById("legend"); const sp=lg.querySelectorAll("span");
   if(seq){ sp[1].style.background="linear-gradient(to right,rgb(232,232,236),rgb(214,16,32))";
     sp[0].textContent="not used here"; sp[2].textContent="common"; }
   else { sp[1].style.background=heatBar();
     sp[0].textContent="uncommon"; sp[2].textContent="common"; }}
  // no more "closest to" — the STAR is the result. Localised answers name their place(s);
  // broad/widespread answers just describe the pattern (the map shows it).
  let matchHTML;
  if(noResult || !scored.length){
    matchHTML="<span style='color:var(--ink);font-weight:600'>Inconclusive &mdash; this doesn&rsquo;t point to a particular place.</span>";
  } else if(starPlaces.length>=3 &&
            new Set(starPlaces.map(p=>regionGrid[p.row|0][p.col|0])).size===1 &&
            regionGrid[starPlaces[0].row|0][starPlaces[0].col|0]>=0){
    // Three tied places inside ONE region isn't three findings, it's one: a word
    // that is uniform across the whole region. Listing them spells the region's
    // name out three times ("Glasgow (Scotland) & Aberdeen & Edinburgh
    // (Scotland)"), and implies a precision the surface doesn't have.
    matchHTML="&#128205; most common in <b>"+
      regionNames[regionGrid[starPlaces[0].row|0][starPlaces[0].col|0]]+"</b>";
  } else if(starPlaces.length){
    matchHTML="&#9733; <b>"+starPlaces.map(placeName).join(" &amp; ")+"</b>";
  } else {
    // broad answers: describe the REGION (e.g. "the North of England"), or "much of Britain"
    const rn=matchRegion(surf);
    matchHTML = (rn==="much of Britain")
      // ink, not the muted grey: this is a real finding (your answer IS the
      // national norm), and sharing a colour with "Inconclusive" made it read
      // as a failed lookup rather than an answer
      ? "<span style='color:var(--ink);font-weight:600'>Used across much of Britain</span>"
      : "&#128205; most common in <b>"+rn+"</b>";
  }
  document.getElementById("match").innerHTML=matchHTML;
  // (i) more info — resolved per question so it's never "undefined"
  let infoHTML="", infoLabel="";
  if(q.info){ infoHTML=ETYM[q.info]||""; infoLabel=q.infoLabel||""; }
  else if(q.multi){ const parts=sel.map(o=>ETYM[o.grid]).filter(Boolean);
    infoHTML=parts.join("<hr class='isep'>"); infoLabel="your word"+(sel.length>1?"s":""); }
  const infowrap=document.getElementById("infowrap"), infobtn=document.getElementById("infobtn"), info=document.getElementById("info");
  if(infoHTML){ infowrap.style.display="inline-block"; infobtn.innerHTML="&#9432; about "+infoLabel;
    const _lk=INFO_LINKS[q.info];
    info.innerHTML=infoHTML+(_lk?"<div class='ilinks'>Read more: "+_lk.map(function(x){
      return "<a href='"+x[1]+"' target='_blank' rel='noopener noreferrer'>"+x[0]+"</a>";
    }).join(" &middot; ")+"</div>":"");
  } else { infowrap.style.display="none"; }
  document.getElementById("detail").innerHTML="";
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
  // z-score each surface before averaging: a sharply localized answer (lolly ice,
  // spelk) then counts as much as a broad yes/no plateau. Without this the wide
  // plateaus swamp the localized signals and drag the peak toward their centroid
  // (a fully "London" answer set used to land on Margate).
  const zs=[];
  for(const s of surfs){ let sum=0,sq=0,n=0;
    for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const v=s[r]?s[r][c]:null; if(v!=null){sum+=v;sq+=v*v;n++;}}
    const mu=n?sum/n:0, sd=n?Math.sqrt(Math.max(sq/n-mu*mu,1e-9)):1;
    zs.push({s:s,mu:mu,sd:sd});}
  let comb=[];
  for(let r=0;r<H_;r++){comb.push([]);for(let c=0;c<W;c++){ if(!land[r][c]){comb[r].push(null);continue;}
    let sum=0,n=0; for(const z of zs){const v=z.s[r]?z.s[r][c]:null; if(v!=null){sum+=(v-z.mu)/z.sd;n++;}} comb[r].push(n?sum/n:null);}}
  comb=blurLand(comb,3);   // Gaussian smoothing over the overlaid maps
  let mx=-1e9,mn=1e9,pr=-1,pc=-1;
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const m=comb[r][c]; if(m!=null){if(m>mx){mx=m;pr=r;pc=c;} if(m<mn)mn=m;}}
  if(mx>mn) for(let r=0;r<H_;r++)for(let c=0;c<W;c++){ if(comb[r][c]!=null) comb[r][c]=(comb[r][c]-mn)/(mx-mn); }
  // winner = highest 5x5 neighbourhood mean around a NAMED place (same scoring
  // as the runners-up list), not the raw global peak cell: blurLand averages
  // over land only, which flatters coastal cells (fewer neighbours to dilute
  // them), so a raw-peak rule kept snapping inland-London answers to the coast
  let best=null,bv=-1e9;
  for(const p of PLACES){let s=0,n=0;
    for(let dr=-2;dr<=2;dr++)for(let dc=-2;dc<=2;dc++){const r=(p.row|0)+dr,c=(p.col|0)+dc;
      if(r<0||r>=H_||c<0||c>=W)continue; const v=comb[r]?comb[r][c]:null; if(v!=null){s+=v;n++;}}
    const sc=n?s/n:-1e9; if(sc>bv){bv=sc;best=p;}}
  return {surf:comb, count:surfs.length, peak:best?[best.row|0,best.col|0]:[pr,pc], place:best};
}
function drawCombined(cs){
  const comb=cs.surf;
  document.getElementById("right").style.display="";
  document.getElementById("qimg").style.display="none";
  document.getElementById("rprompt").style.display="none"; cv.style.display="block";
  document.getElementById("legend").style.display="flex";
  document.getElementById("detail").innerHTML=""; document.getElementById("infowrap").style.display="none";
  document.getElementById("rtitle").innerHTML=
    "<b>Where your speech fits</b><span class='rsub'>Hover any city to see how close it is</span>";
  const col=v=>heat(v);   // full blue -> white -> red diverging = least .. most like you
  fitCanvas();                       // snap to whole device pixels before painting
  cx.clearRect(0,0,cv.width,cv.height);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){ if(!land[r][c])continue;
    const v=comb[r][c]; const [rr,gg,bb]=(v==null)?THEME.nullCell:col(v);
    cx.fillStyle=gutterCol(rr,gg,bb);cx.fillRect(c*CELL,r*CELL,CELL,CELL);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fillRect(c*CELL,r*CELL,CELL-GAP,CELL-GAP);}
  const star=cs.place;   // the named place nearest the smoothed peak
  const ranked=matchPlaces(comb);
  // Dots are drawn at PLACES, not `cities`: PLACES is exactly the set that gets
  // ranked (and that "also close"/"least like" draw from), so every dot on this
  // map is hoverable and carries a rank. Using `cities` here left five ranked
  // places (Hull, Stoke, Middlesbrough, Margate, Swansea) with no dot at all.
  for(const ct of PLACES){ if(star&&star.col===ct.col&&star.row===ct.row)continue;
    const v=comb[ct.row|0]?comb[ct.row|0][ct.col|0]:null; const [rr,gg,bb]=(v==null)?THEME.nullDot:col(v);
    cx.beginPath();cx.arc((ct.col+0.5)*CELL,(ct.row+0.5)*CELL,5*PXS,0,7);cx.fillStyle="rgb("+rr+","+gg+","+bb+")";cx.fill();cx.lineWidth=2*PXS;cx.strokeStyle=THEME.dotStroke;cx.stroke();}
  if(star){const x=(star.col+0.5)*CELL,y=(star.row+0.5)*CELL; drawStar(x,y,17*PXS,THEME.halo,0); drawStar(x,y,13*PXS,rgbOf(THEME.star),3*PXS);}
  {const lg=document.getElementById("legend"); const sp=lg.querySelectorAll("span");
   sp[1].style.background=heatBar(); sp[0].textContent="least like you"; sp[2].textContent="most like you";}
  // summary map IS hoverable: hoverPts overrides the usual city list, and each
  // place carries its own similarity so the tooltip can describe the fit
  const fit=new Map(); ranked.forEach(s=>fit.set(s.p,s.v));
  SHOWN={summary:true, surf:comb, hoverPts:PLACES, fit:fit,
         topPlace:ranked.length?ranked[0].p:null,
         lastPlace:ranked.length?ranked[ranked.length-1].p:null};
  document.getElementById("match").innerHTML="";
  return ranked;
}

const tip=document.getElementById("tip");
function cvTip(clientX,clientY){
  if(!SHOWN){tip.style.opacity=0;return;}
  const rect=cv.getBoundingClientRect(),sx=cv.width/rect.width,sy=cv.height/rect.height;
  const x=(clientX-rect.left)*sx,y=(clientY-rect.top)*sy;
  const pts=SHOWN.hoverPts||[...cities,...(SHOWN.shownPlaces||[])];
  let best=null,bd=1e9;for(const ct of pts){const dd=Math.hypot((ct.col+0.5)*CELL-x,(ct.row+0.5)*CELL-y);if(dd<bd){bd=dd;best=ct;}}
  if(best&&bd<=24*PXS){const v=SHOWN.surf[best.row|0]?SHOWN.surf[best.row|0][best.col|0]:null;
    const br=best.row|0,bc=best.col|0;
    let line;
    if(SHOWN.summary){
      // result map: describe how close this place is IN WORDS. The underlying
      // number is a rank, but showing "11th of 20" reads as a league table and
      // implies a precision the model doesn't have -- the gap between 9th and
      // 12th is usually noise. The bands match the colour already on the dot.
      const f=SHOWN.fit.get(best);
      line = (best===SHOWN.topPlace)  ? "your closest match"
           : (best===SHOWN.lastPlace) ? "least like you"
           : (f==null)   ? "&mdash;"
           : (f>=0.80)   ? "very like you"
           : (f>=0.60)   ? "quite like you"
           : (f>=0.40)   ? "somewhat like you"
           : (f>=0.20)   ? "not much like you"
                         : "very unlike you";
      tip.innerHTML="<b>"+placeName(best)+"</b><br><small>"+line+"</small>";
      tip.style.left=((best.col+0.5)*CELL/sx)+"px";tip.style.top=((best.row+0.5)*CELL/sy)+"px";tip.style.opacity=1;
      return;
    }
    if(SHOWN.slider){ const rv=SHOWN.raw[br]?SHOWN.raw[br][bc]:null; line=(rv==null)?"&mdash;":fmtPct(rv*100)+" acceptance rating"; }
    else if(SHOWN.q.metric==="pct"){
      // "no word"/"another term" has no surface of its own, but the question's own
      // map still does -- fall back to it so the city can still be read
      const g=SHOWN.incon?grids[SHOWN.q.id]:null, rv=SHOWN.incon?(g&&g[br]?g[br][bc]:null):v;
      const yes=SHOWN.q.opts.find(o=>o.word&&!o.none);
      const w=SHOWN.incon?((yes&&yes.word)||""):(SHOWN.sel[0].word||"");
      line=(rv==null)?"&mdash;":fmtPct(rv*100)+(w?" "+w:""); }
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

// ---- mobile reading order ---------------------------------------------------
// Stacked on a phone, the two columns put the photo AFTER the answers and let
// you press Continue without ever seeing the map. On narrow screens the image
// moves up between question and answers, and the buttons move below the map, so
// the map always sits on the path to the next question.
const MQ_NARROW = window.matchMedia("(max-width: 640px)");
function applyLayout(){
  const stage=document.getElementById("stage"), left=document.getElementById("left"),
        right=document.getElementById("right"), img=document.getElementById("qimg"),
        opts=document.getElementById("opts"), next=document.getElementById("next"),
        nav=document.querySelector(".navbtns"), done=document.getElementById("leftdone");
  if(!stage||!left||!right||!img||!next||!nav) return;
  if(MQ_NARROW.matches){
    if(img.parentNode!==left) left.insertBefore(img, opts);
    if(next.parentNode!==stage) stage.appendChild(next);
    if(nav.parentNode!==stage) stage.appendChild(nav);
  } else {
    if(img.parentNode!==right) right.insertBefore(img, right.firstChild);
    if(next.parentNode!==left) left.insertBefore(next, done);
    if(nav.parentNode!==left) left.appendChild(nav);
  }
}
if(MQ_NARROW.addEventListener) MQ_NARROW.addEventListener("change", applyLayout);
// Resizing (or zooming, or moving to a different-DPI screen) changes how many
// whole device pixels a cell can have, so the map has to be refitted and
// repainted. Debounced: a drag-resize fires this continuously.
let _rsz=null, _lastW=window.innerWidth;
window.addEventListener("resize",()=>{ clearTimeout(_rsz);
  _rsz=setTimeout(()=>{
    const w=window.innerWidth, widthChanged=(w!==_lastW); _lastW=w;
    // A phone's on-screen keyboard fires resize when an input is focused, and it
    // changes the HEIGHT only. Re-rendering there rebuilt the hometown question
    // from scratch mid-typing -- new input node, text wiped, suggestion list
    // destroyed, focus lost -- so a town could never actually be picked. Never
    // re-render out from under someone who is typing.
    const el=document.activeElement;
    const typing=el&&(el.tagName==="INPUT"||el.tagName==="TEXTAREA");
    if(typing&&!widthChanged) return;
    const onIntro=document.getElementById("intro").style.display!=="none";
    if(onIntro){ drawMini(); return; }
    // the hometown question has no map to refit, so a re-render is all cost
    if(QUESTIONS[idx]&&QUESTIONS[idx].hometownq) return;
    render();
  }, 120); });

// ---- landing page hero map: smooth (not the quiz's chunky pixel style) ----
// bumped on every drawMini() call so a re-render (e.g. theme flip) retires the
// previous requestAnimationFrame loop instead of leaving two running at once
let miniToken=0;
function drawMini(){
  const myToken=++miniToken;
  // Draw at a large supersampled resolution (one flat-coloured square per
  // grid cell, no gaps) then let the browser's own bilinear image scaling
  // shrink that down to the on-page size. Shrinking a high-res hard-edged
  // image is what real anti-aliasing looks like -- crisp region colours,
  // smoothly stepped edges -- as opposed to a Gaussian blur, which softens
  // the whole picture into a fog. That's the "8-bit"/blurry look to avoid.
  // Built at exactly the size it is displayed at, by the same rule as the quiz
  // maps: a whole number of DEVICE pixels per cell. This canvas used to be drawn
  // at 470x725 and then squeezed to a 290px CSS box -- a 0.617 scale, so every
  // cell landed on 3.08 screen pixels and the gutters were smeared unevenly,
  // exactly the artefact the quiz maps had. Now there is no scaling step at all,
  // so the two maps share one pixel grid and one gutter treatment.
  const mdpr=window.devicePixelRatio||1;
  const mc=document.getElementById("introcv");
  mc.style.width="";mc.style.height="";
  mc.width=W*BASE_CELL;mc.height=H_*BASE_CELL;          // provisional: sets the aspect
  const mrect=mc.getBoundingClientRect();
  const SS=Math.max(2, Math.floor(((mrect.width||290)*mdpr)/W));
  const MGAP=(SS>=3)?Math.max(1,Math.round(SS*BASE_GAP/BASE_CELL)):0;
  const MGW=MGAP?Math.min(1,(SS*BASE_GAP/BASE_CELL)/MGAP):0;
  const PW=W*SS, PH=H_*SS;
  // cells grouped by region, so a region can be repainted on its own
  const cellsBy=[]; for(let i=0;i<dialectColors.length;i++) cellsBy.push([]);
  for(let r=0;r<H_;r++)for(let c=0;c<W;c++){const d=dialectGrid[r][c]; if(d>=0) cellsBy[d].push(r*W+c);}
  function blank(){const o=document.createElement("canvas");o.width=PW;o.height=PH;return o;}
  // one pre-rendered layer per region (bounded to its own bounding box, so the
  // whole set costs a few map-areas of memory rather than one full canvas each)
  const layers=[];
  for(let i=0;i<dialectColors.length;i++){
    const cells=cellsBy[i];
    if(!cells.length){layers.push(null);continue;}
    let r0=1e9,r1=-1,c0=1e9,c1=-1;
    for(const k of cells){const r=(k/W)|0,c=k%%W; if(r<r0)r0=r; if(r>r1)r1=r; if(c<c0)c0=c; if(c>c1)c1=c;}
    const o=document.createElement("canvas");
    o.width=(c1-c0+1)*SS; o.height=(r1-r0+1)*SS;
    const x=o.getContext("2d");
    const col=vividTint(dialectColors[i][1]);
    x.fillStyle="rgb("+col[0]+","+col[1]+","+col[2]+")";
    for(const k of cells){const r=(k/W)|0,c=k%%W; x.fillRect((c-c0)*SS,(r-r0)*SS,SS-MGAP,SS-MGAP);}
    layers.push({cv:o,x:c0*SS,y:r0*SS});
  }
  // muted base: every region desaturated toward the page background, so the
  // highlighted one reads clearly without the map going flat grey
  const baseC=blank(); {const x=baseC.getContext("2d");
    for(let i=0;i<dialectColors.length;i++){
      const col=dialectColors[i][1];
      const g=col[0]*0.3+col[1]*0.5+col[2]*0.2;
      const [m0,m1,m2]=baseTint(col);
      for(const k of cellsBy[i]){const r=(k/W)|0,c=k%%W;
        x.fillStyle=gutterCol(m0,m1,m2,MGW); x.fillRect(c*SS,r*SS,SS,SS);
        x.fillStyle="rgb("+m0+","+m1+","+m2+")";
        x.fillRect(c*SS,r*SS,SS-MGAP,SS-MGAP);}
    }}
  const buf=blank(), bx=buf.getContext("2d");
  mc.width=PW; mc.height=PH;
  mc.style.width=(PW/mdpr)+"px"; mc.style.height=(PH/mdpr)+"px";
  const mx=mc.getContext("2d");
  mx.imageSmoothingEnabled=true; mx.imageSmoothingQuality="high";
  const tip=document.getElementById("introtip"), cap=document.getElementById("introcap"),
        hint=document.getElementById("introhint");

  // tour order: north -> south, skipping slivers too small to register at this size
  const tour=[];
  for(let i=0;i<dialectColors.length;i++) if(cellsBy[i].length>=22) tour.push(i);
  tour.sort((a,a2)=>{const m=l=>cellsBy[l].reduce((s,k)=>s+((k/W)|0),0)/cellsBy[l].length;
    return m(a)-m(a2);});
  const BEAT=2500, FADE=700;
  // "inside" is tracked from enter/leave, NOT from whether a region sits under the
  // cursor: the canvas box includes sea, and deriving it from the region lookup
  // made the tour start advancing again whenever the pointer crossed open water.
  let cur=tour.length?tour[0]:-1, prev=-1, fadeStart=-1e9, nextAt=0, inside=false, started=0;

  function setLabel(di){
    if(di<0){ cap.textContent="\\u2014"; cap.style.color=""; cap.style.backgroundColor=""; return; }
    const nm=dialectColors[di][0], col=vividTint(dialectColors[di][1]);
    const txt=readableOn(col,4.5);
    cap.textContent=nm;
    cap.style.color="rgb("+txt[0]+","+txt[1]+","+txt[2]+")";
    cap.style.backgroundColor="rgba("+col[0]+","+col[1]+","+col[2]+",.16)";
  }
  function show(di,t){ if(di===cur)return; prev=cur; cur=di; fadeStart=t; setLabel(di); }

  function frame(t){
    if(myToken!==miniToken) return;          // superseded by a newer drawMini()
    if(!started) started=t;
    if(!inside && t>=nextAt){                       // idle -> advance the tour
      const i=tour.indexOf(cur);
      show(tour[(i+1)%%tour.length]||tour[0], t);
      nextAt=t+BEAT;
    }
    const a=Math.min(1,(t-fadeStart)/FADE);
    bx.clearRect(0,0,PW,PH);
    bx.globalAlpha=1; bx.drawImage(baseC,0,0);
    // Push the whole map back, then paint the spotlit region over the top at full
    // strength. Colour alone cannot carry this: the palette runs adjacent regions
    // through the same hue ramp (Lancashire and Yorkshire are 49 apart), so a
    // highlight that only changes shade reads as one more coloured patch among 33.
    // source-atop keeps the veil on the land -- a plain fillRect would tint the
    // transparent sea and leave a visible rectangle around the island.
    bx.globalCompositeOperation="source-atop";
    bx.fillStyle="rgba("+THEME.page[0]+","+THEME.page[1]+","+THEME.page[2]+","+THEME.veil+")";
    bx.fillRect(0,0,PW,PH);
    bx.globalCompositeOperation="source-over";
    if(a<1&&prev>=0&&layers[prev]){bx.globalAlpha=1-a;const L=layers[prev];bx.drawImage(L.cv,L.x,L.y);}
    if(cur>=0&&layers[cur]){bx.globalAlpha=a;const L=layers[cur];bx.drawImage(L.cv,L.x,L.y);}
    bx.globalAlpha=1;
    mx.clearRect(0,0,mc.width,mc.height);
    mx.drawImage(buf,0,0,mc.width,mc.height);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // hover takes over the spotlight; leaving hands it back to the tour
  function at(clientX,clientY){
    const rect=mc.getBoundingClientRect();
    const c=Math.floor((clientX-rect.left)/(rect.width/W)), r=Math.floor((clientY-rect.top)/(rect.height/H_));
    return (r>=0&&r<H_&&c>=0&&c<W)?dialectGrid[r][c]:-1;
  }
  function enter(){ inside=true; if(hint)hint.classList.add("dim"); }
  function leave(delay){ inside=false; tip.style.opacity=0;
    if(hint)hint.classList.remove("dim"); nextAt=performance.now()+delay; }
  function onMove(clientX,clientY){
    enter();
    const di=at(clientX,clientY), rect=mc.getBoundingClientRect(), t=performance.now();
    if(di>=0){
      show(di,t);
      // the pill carries white text, so its fill has to be dark enough for that --
      // a pale region colour straight from the palette was white-on-white
      const nm=dialectColors[di][0], col=readableOn(vividTint(dialectColors[di][1]),4.5,[255,255,255]);
      tip.textContent=nm; tip.style.background="rgb("+col[0]+","+col[1]+","+col[2]+")";
      tip.style.left=(clientX-rect.left)+"px"; tip.style.top=(clientY-rect.top)+"px"; tip.style.opacity=1;
    } else { tip.style.opacity=0; }   // over sea: keep the last region lit, just drop the tooltip
  }
  mc.onmouseenter=enter;
  mc.onmousemove=(e)=>onMove(e.clientX,e.clientY);
  mc.onmouseleave=()=>leave(900);
  mc.ontouchstart=(e)=>{if(e.touches[0])onMove(e.touches[0].clientX,e.touches[0].clientY);};
  mc.ontouchmove=(e)=>{if(e.touches[0]){e.preventDefault();onMove(e.touches[0].clientX,e.touches[0].clientY);}};
  mc.ontouchend=()=>leave(1600);
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
  }
}

// ---- response collection ---------------------------------------------------
// Paste a URL into COLLECT.endpoint to switch collection on; leave it empty and
// nothing is ever sent. Submission happens ONCE, when the result screen is first
// reached AND every question has an answer (see runIsComplete), so a stored
// response is always a complete run rather than a fragment.
// Optional outbound links. Leave a value empty and it simply renders as plain
// text -- no dead links on the page.
const LINKS={ linkedin:"" };      // <-- paste your LinkedIn profile URL
const COLLECT={
  endpoint:"https://script.google.com/macros/s/AKfycbynutWpL2Z3-7nigKUseLYBgrzfEuANgxA7RapaKn_ME1e5pG-Vu5F_-SrM2UvHdair/exec",
  version:"2026-08-13"      // bump when the question set changes, so responses stay comparable
};
// Random per page-load, NOT persisted: enough to drop an accidental double-submit
// within one sitting, while never linking a visitor across visits.
const RUN_ID=(function(){try{
  const a=new Uint8Array(8); crypto.getRandomValues(a);
  return Array.from(a,b=>b.toString(16).padStart(2,"0")).join("");
}catch(e){return String(Date.now())+Math.random().toString(16).slice(2);}})();
let _submitted=false;
// A run is only recorded if it is COMPLETE: the result screen has been reached
// and every dialect question carries an answer. The quiz flow already forces an
// answer before "Continue" unlocks, but this is checked again here rather than
// assumed, because a half-finished run would bias the maps toward whichever
// questions people happen to give up on.
// Hometown is deliberately NOT required here -- it's research metadata, not part
// of the dialect record, so a complete run still counts without it. Filter on
// hometown at retrain time instead, where it actually matters.
function runIsComplete(){
  if(idx<QUESTIONS.length) return false;              // "Finish" not pressed yet
  for(const q of QUESTIONS){
    if(q.hometownq) continue;
    const a=answers[q.id];
    if(a===undefined) return false;
    if(q.multi && (!Array.isArray(a) || a.length===0)) return false;
  }
  return true;
}
function submitResponses(placeLabel,score){
  if(_submitted || !consented || !COLLECT.endpoint) return;
  if(!runIsComplete()) return;      // partial run: drop it, and stay droppable
  _submitted=true;
  const out={};
  for(const q of QUESTIONS){ if(q.hometownq)continue;
    if(answers[q.id]!==undefined) out[q.id]=answers[q.id]; }
  const payload={version:COLLECT.version, run:RUN_ID, ts:new Date().toISOString(),
    hometown:answers.hometown||"", result:placeLabel||"", match:(score==null?"":score),
    answers:out};
  beacon(payload);
}
// text/plain dodges the CORS preflight that blocks application/json on Apps
// Script and similar simple endpoints; sendBeacon survives the tab being closed
// the moment the result appears.
function beacon(payload){
  if(!COLLECT.endpoint) return;
  // Never send from a local build. Opening index.html off disk, or running it on
  // localhost, is testing -- those hits must not land in the live Sheet next to
  // real responses, or turn up in the traffic counts as phantom visitors.
  const h=location.hostname;
  if(location.protocol==="file:"||h===""||h==="localhost"||h==="127.0.0.1"||h==="::1") return;
  try{
    const body=JSON.stringify(payload);
    const blob=new Blob([body],{type:"text/plain;charset=UTF-8"});
    if(navigator.sendBeacon && navigator.sendBeacon(COLLECT.endpoint, blob)) return;
    fetch(COLLECT.endpoint,{method:"POST",mode:"no-cors",keepalive:true,
      headers:{"Content-Type":"text/plain;charset=UTF-8"},body:body}).catch(()=>{});
  }catch(e){}
}

// ---- traffic ---------------------------------------------------------------
// A first-party funnel counter, written to a separate `traffic` tab: visit ->
// start -> complete. Three numbers answer the questions that matter (how many
// came, how many began, how many finished) without a third-party analytics
// script, a cookie, or anything that persists between visits.
//
// Deliberately non-identifying: the referrer is reduced to its HOST (so you can
// see "came from reddit.com" but not which thread or which search terms), the
// viewport is rounded to the nearest 100px, and `run` is the same per-page-load
// id used above, which is regenerated on every load and never stored. There is
// nothing here that can single out or re-identify a visitor.
const _seen={};
function pingEvent(name){
  if(_seen[name] || !COLLECT.endpoint) return;
  _seen[name]=true;
  let ref="";
  try{ if(document.referrer){ const u=new URL(document.referrer);
        if(u.host!==location.host) ref=u.host; } }catch(e){}
  beacon({type:"event", event:name, version:COLLECT.version, run:RUN_ID,
          ts:new Date().toISOString(), ref:ref,
          vw:Math.round((window.innerWidth||0)/100)*100});
}
function startQuiz(){
  recordConsent();                                       // record agreement before the quiz
  pingEvent("start");
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
  // Refit the hero map. It can only measure itself while it is on screen, so a
  // window resize that happened during the quiz leaves it sized for the old
  // viewport until we come back here.
  if(typeof drawMini==="function") drawMini();
}
document.getElementById("startbtn").onclick=startQuiz;
// ---- hometown type-ahead: attached to the input built for the first question ----
function attachCombo(inp, list){
  if(!inp||!list) return;
  let matches=[], active=-1;
  function close(){ list.style.display="none"; list.innerHTML=""; active=-1; inp.setAttribute("aria-expanded","false"); }
  function paint(){ [...list.children].forEach((li,i)=>li.classList.toggle("active",i===active)); }
  // "picked" (not "input") so listeners can react to a selection without
  // re-triggering the type-ahead filter and reopening the dropdown
  function choose(t){ inp.value=t[0]+", "+t[1]; if(answers.hometown==="notgb") delete answers.hometown; close();
    inp.dispatchEvent(new CustomEvent("picked")); }
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
document.addEventListener("click",function(e){
  const sb=e.target.closest&&e.target.closest(".srcbtn");
  document.querySelectorAll(".srcwrap").forEach(w=>{
    if(sb&&w.contains(sb)){e.stopPropagation();w.classList.toggle("open");} else w.classList.remove("open");});
  if(sb)return;
  document.getElementById("infowrap").classList.remove("open");var aw=document.querySelector(".aboutwrap");if(aw)aw.classList.remove("open");if(_tb)_tb.classList.remove("open");});
// turn the author credit into a link only if one has been configured
(function(){const a=document.getElementById("authorname");
  if(a&&LINKS.linkedin){a.innerHTML='<a class="authorlink" href="'+LINKS.linkedin+
    '" target="_blank" rel="noopener noreferrer">'+a.textContent+'</a>';}})();
applyLayout();
drawMini();
showIntro();
pingEvent("visit");
// landing is now interactive; parse the heavy heat-map data in the background so it's
// ready by the time the user taps "Start" (startQuiz also calls this, just in case)
setTimeout(ensureGrids, 60);
</script></body></html>""" % (
    W, H, json.dumps(json.dumps(grids_all)), json.dumps(landj),
    json.dumps(cities), json.dumps(cg.tolist()), json.dumps(names), json.dumps(places),
    json.dumps(region_grid), json.dumps(region_names),
    json.dumps(dialect_grid), json.dumps(dialect_colors),
    json.dumps(GB_TOWNS),
    json.dumps(icelolly_uri), json.dumps(bread_uri), json.dumps(gum_uri),
)

with open("index.html", "w") as f:
    f.write(html)
print("wrote index.html — done. Now just: git add index.html build_quiz.py && git commit -m 'update' && git push")
