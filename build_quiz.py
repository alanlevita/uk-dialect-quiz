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

grids_all = {"footstrut": grid_json(q1), "tvd": grid_json(decoded_surface("tvd", 0.05, 0.72)),
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

grids_all["none_alley"] = negative_union(["alley_" + t for t in list(ALLEY) + ["alley"]])
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
<!-- favicon: a plain world globe, drawn here in the site's own red. Our own
     artwork, so no third-party mark and nothing to clear permission for. -->
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
  #detail{font-size:13px;color:#444;margin-top:6px;min-height:1.2em;} #detail .ipa{font-size:18px;color:var(--ink);margin:0 6px;letter-spacing:.5px;}
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
  .intro-kicker{font-size:11px;letter-spacing:.09em;text-transform:uppercase;font-weight:750;
       color:var(--muted);margin:0 0 7px;padding-top:12px;border-top:1px solid var(--line);}
  .intro-why{font-size:13px;line-height:1.5;color:var(--body);margin:0 0 10px;}
  .intro-points{list-style:none;margin:0 0 14px;padding:0;}
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
      <p class="intro-kicker">A living atlas of British English</p>
      <p class="intro-why">Britain packs more dialect variation into one island than almost anywhere in the
        English-speaking world &mdash; and most maps of it sit in decades-old atlases. This one is rebuilt from
        the research, put in your hands, and redrawn by everyone who plays.</p>
      <ul class="intro-points">
        <li><b>Interactive, not illustrative.</b> Other quizzes hand you a picture. Every map here is live &mdash;
          hover any city to see the word it really uses, and how common it is.</li>
        <li><b>It shows its working.</b> Nothing is invented: every isogloss is redrawn from published dialect
          research and large-scale surveys, and each question names the feature behind it.</li>
        <li><b>It gets sharper as you play.</b> With consent, your answers feed straight back into the maps.
          Most dialect surveys are a snapshot from one year; this one is still being drawn.</li>
      </ul>
      <label class="consent"><input type="checkbox" id="consent"><span>I agree to the <span class="tlink" id="termsbtn">terms of data collection<span class="terms-pop" id="termspop">By ticking this box, you consent to the collection and storage of your quiz answers and, if you provide it, your hometown. This information is stored <b>anonymously</b>: no name, email address, or IP address is recorded, and it cannot be traced back to you. It is used solely to study regional language variation and to improve future versions of the quiz. Your data will not be sold or shared with third parties.</span></span></span></label>
      <button id="startbtn">Start the quiz &rarr;</button>
      <p class="intro-note"><span class="aboutwrap"><span class="aboutbtn">&#9432;</span><span class="aboutinfo">This is a pixel-art version of the British dialect map, made by <b id="authorname">Alan Levita</b> during a research internship at the Intellectual Forum.<br><br>Your answers are used to estimate roughly where you&rsquo;re from. All the maps were redrawn by hand in a pixel-art style, based on isoglosses from published dialect research and large-scale surveys.</span></span> Powered by the Intellectual Forum at Jesus College, Cambridge</p>
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
// NOTE: ordered newest-added -> oldest-added (not the usual thematic order) so
// new features land at the front of the quiz for quick manual testing. hometown
// stays first regardless (it's an intake field, not a dialect question).
const QUESTIONS=[
  // first question: where did you grow up? (a GB town type-ahead, or "not from GB").
  // Not a heat-map question — collected (with consent) for future training, not scored.
  {id:"hometown",text:"Where in Great Britain did you grow up?",hometownq:true},
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
  {id:"tvd",text:"What do you call your evening meal?",tag:"real data",real:true,metric:"pct",
   info:"tvd",infoLabel:"tea vs dinner",
   opts:[{label:"Tea",v:1,word:"say tea"},{label:"Dinner",v:0,word:"say dinner"}]},
  {id:"trapbath",text:"Do <i>gas</i> and <i>grass</i> rhyme for you?",tag:"blended: BBC Future + English Dialect App",real:true,metric:"pct",
   info:"trapbath",infoLabel:"the trap&ndash;bath split",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme (short a)"},
         {label:"No, they sound different",v:0,word:"split (long a)"}]},
  {id:"bookspook",text:"Do <i>book</i> and <i>spook</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   info:"bookspook",infoLabel:"book vs spook",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme book/spook"},{label:"No, they sound different",v:0,word:"don&rsquo;t rhyme"}]},
  // metric "pct": a clean binary the paper reports as proportions -> show a percent.
  // ipa:true enables the click-a-city foot-strut IPA readout (foot-strut only).
  {id:"footstrut",text:"Do <i>foot</i> and <i>cut</i> rhyme for you?",tag:"real data",real:true,metric:"pct",
   ipa:true,info:"footstrut",infoLabel:"the foot&ndash;strut split",
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
  tvd:"<b>tea vs dinner</b> &mdash; the name for the evening meal. <i>Tea</i> is the northern (and traditionally working-class) term; <i>dinner</i> the southern one, and historically the &lsquo;U&rsquo;/upper-class usage (Ross, 1954). So it carries a class edge as well as a regional one.",
  bookspook:"<b>book vs spook</b> &mdash; in some accents the <i>-ook</i> words (book, cook, look) keep the old long vowel /u&#720;/, so <i>book</i> is [bu&#720;k] and rhymes with <i>spook</i> &mdash; putting it in the GOOSE set rather than FOOT. Traditional in the North East and Stoke (and once Liverpool); Scotland has no foot&ndash;goose distinction at all.",
  nursesquare:"<b>The NURSE&ndash;SQUARE merger</b> &mdash; in some accents the vowels of NURSE (<i>stir, fur, her</i>) and SQUARE (<i>stare, fair, hair</i>) fall together, so <i>stir</i> and <i>stare</i> (or <i>fur</i> and <i>fair</i>) rhyme. It is long-established and best-documented in Liverpool / Merseyside and the North West, and is now also strong &mdash; and apparently spreading &mdash; along the east coast (Hull, Teesside). An older East-Midlands merger has largely faded, though north-east Lincolnshire still has it.",
  scone:"<b>scone</b> &mdash; the great teatime shibboleth: does it rhyme with <i>gone</i> (/sk&#594;n/) or with <i>bone</i>/<i>cone</i> (/sko&#650;n/)? Most of Britain &mdash; Scotland and the North especially &mdash; rhymes it with <i>gone</i>. The <i>bone</i> pronunciation is the local norm in the <b>East Midlands</b> (Nottingham, Derby, Leicester), with the far South West leaning that way a little too.",
  northforce:"<b>The NORTH&ndash;FORCE merger</b> &mdash; whether <i>horse</i> and <i>hoarse</i> (or <i>for</i> and <i>four</i>, <i>war</i> and <i>wore</i>) sound identical. Most of England and Wales merged them long ago, so they rhyme; <b>Scotland</b> keeps them clearly distinct, as do pockets around <b>Manchester</b> and Merseyside.",
  forcecure:"<b>The CURE&ndash;FORCE merger</b> &mdash; whether <i>poor</i> and <i>pour</i> (or <i>sure</i> and <i>shore</i>, <i>tour</i> and <i>tore</i>) sound identical. Across most of England they have merged, so they rhyme; the older distinct <i>poor</i>/<i>sure</i> vowel /&#650;&#601;/ survives in <b>Scotland</b>, the <b>North East</b>, and <b>West Yorkshire</b>.",
  youse:"<b>Plural &lsquo;yous(e)&rsquo;</b> &mdash; a second-person plural pronoun, filling the gap English left when <i>thou/you</i> collapsed to just <i>you</i>. Strongest in <b>Scotland</b> and the <b>North East</b> (Newcastle, Sunderland, Middlesbrough), fading through the Midlands, and rare in southern England, where <i>you guys</i> or plain <i>you</i> is used instead.",
  mother:"<b>Words for &lsquo;mother&rsquo;</b> &mdash; <b>mum</b> is the general term across most of England and Scotland. <b>Mam</b> is the Welsh and Irish word, and is standard in <b>Wales</b>, the <b>North East</b> and <b>Cumbria</b>. <b>Mom</b> is almost entirely a <b>West Midlands</b> form, centred on Birmingham and reaching west as far as Telford &mdash; it is not an Americanism there but a long-standing local usage. <b>Maw</b> is a Scots clipping heard across the Central Belt, and <b>mummy</b> survives in adults mainly in south-east England. <b>Mammy</b> is overwhelmingly Irish, but has a real foothold in <b>south-west Wales</b> around Swansea and Carmarthenshire.",
  shoes:"<b>Names for PE plimsolls</b> &mdash; the black canvas shoes worn for primary-school PE, and one of the most sharply regional words in Britain. <b>Plimsolls</b> is the southern and eastern norm (91%% in Norfolk), but flips to <b>pumps</b> across the North West and West Midlands (72&ndash;75%% in Cheshire, Lancashire, Merseyside and Staffordshire). <b>Daps</b> clusters either side of the Severn Estuary &mdash; South Wales and the Bristol area. Scotland splits several ways: <b>sandshoes</b> or <b>sannies</b> around the Clyde, <b>gutties</b> in Lanarkshire, <b>gym shoes</b> in the North East, and <b>rubbers</b> almost only in the Lothians. There is also a striking island of <i>sandshoes</i> around Hull.",
  singerfinger:"<b>Velar nasal plus</b> (also called <i>ng</i>-coalescence) &mdash; whether a hard [&#609;] survives after the <i>ng</i>, so <i>singer</i> is [s&#618;&#331;&#609;&#601;] and rhymes with <i>finger</i>. English generally dropped that [&#609;] around the 17th century, but the change never took hold across the <b>North West</b> and <b>West Midlands</b>: Manchester, Liverpool, Stoke, Birmingham and Cheshire keep it, as does north-east Wales (Flintshire, Wrexham). Elsewhere &mdash; the North East, East Anglia, the South &mdash; the two words are distinct.",
  thfronting:"<b>TH-fronting</b> &mdash; replacing the &lsquo;th&rsquo; sounds /&#952;/ and /&#240;/ with /f/ and /v/, so <i>think</i> &rarr; <i>fink</i> and <i>brother</i> &rarr; <i>bruvver</i>. Once a London (Cockney) feature, it has spread rapidly since the late 20th century and is now common across much of urban England, especially among younger speakers &mdash; while remaining rare in Scotland, Wales, and rural areas generally.",
  skiveclass:"<b>Words for skipping school</b> without permission, from the BBC Voices survey. <b>Skive</b> is the general British term, strongest in Scotland and the South West; <b>bunk off</b> is a London/South East form; <b>wag</b> belongs to the North West and North East; <b>play hookey</b> is a chiefly North Eastern (Tyneside) usage; <b>skip</b> is used more loosely nationwide, without a single clear home region.",
  trapbath:"<b>The trap&ndash;bath split</b> &mdash; in the 18th century southern English lengthened the <i>a</i> in a set of words (<i>bath, grass, last, dance</i>) to /&#593;&#720;/, splitting them from TRAP words (<i>cat, trap</i>). The North, Wales and Scotland kept the short /a/ &mdash; so a northerner says [ba&#952;], a southerner [b&#593;&#720;&#952;]. It&rsquo;s one of the sharpest north&ndash;south markers.",
  splinter:"<b>Words for a splinter</b> of wood in the skin. <b>Splinter</b> is the standard nationwide; <b>spelk</b> (from Old Norse / Old English <i>spelc</i>) belongs to the North East &amp; the Borders; <b>spell</b> is northern; <b>shiver</b> is East Anglian; <b>sliver</b> is a South East word; and <b>skelf</b> is the Scots term, dense across the Central Belt and the Highlands.",
  giveitme:"<b>&lsquo;Give it me&rsquo;</b> &mdash; the &lsquo;alternative double-object&rsquo; dative: the theme (<i>it</i>) comes before the goal (<i>me</i>) with no <i>to</i> &mdash; <i>give it me</i> rather than <i>give it to me</i> or <i>give me it</i>. It&rsquo;s a North West &amp; Midlands feature (strongest around Manchester and the Potteries), thinning towards the North East and the South.",
  lolly:"<b>Ice lolly vs lolly ice</b> &mdash; <i>ice lolly</i> is the standard British term; <i>lolly ice</i> (the words reversed) is the well-known Merseyside / Liverpool (&lsquo;Scouse&rsquo;) form. Further afield you&rsquo;ll hear <i>ice pop</i> (Ireland, Scotland) or <i>popsicle</i> (North America)."
,
  alley:"<b>Words for an alleyway</b> &mdash; the narrow walkway between or behind houses, and one of the most finely divided words in Britain. <b>Alley</b> or <b>alleyway</b> is the national default and almost the only word used in the south. The north is where it fragments: <b>ginnel</b> across Lancashire, Greater Manchester and West Yorkshire; <b>snicket</b> in a tight pocket around Bradford and York; <b>gennel</b> around Sheffield; <b>jitty</b> through Derby, Nottingham and Leicester; <b>entry</b> on Merseyside and in the West Midlands; and <b>cut</b> in Newcastle. The sharpest divide of all is Bradford against Leeds &mdash; snicket and ginnel, about ten miles apart.",
  tag:"<b>Names for tag/it</b> &mdash; <i>tig</i> covers most of England, Scotland &amp; Wales; <i>it</i> is the South East&rsquo;s word instead of <i>tig</i>. Distinct local pockets survive within that: <i>tiggy</i> and <i>tuggy</i> side by side around Durham &amp; North Yorkshire, <i>tick</i> and a tiny <i>tip</i> pocket in North Wales, <i>touch</i> around Birmingham and in the South West, <i>had</i> on the Suffolk/Essex coast, <i>hit</i> on the South Devon coast, and <i>dobby</i> &mdash; a well-known Nottinghamshire/South Yorkshire term &mdash; in a tight pocket around Sheffield."
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
const SOURCES=[
  "YouGov, August 2025 (n&asymp;38,000)",
  "BBC Voices, via Grieve et al. (2019)",
  "Starkey Comics dialect surveys",
  "MacKenzie, Bailey &amp; Turton (2022), <i>Journal of Linguistic Geography</i>",
  "Tweetolectology Twitter survey (2020&ndash;21)",
  "Survey of English Dialects (Orton et al., 1978)",
  "<i>Our Dialects</i> &mdash; L. MacKenzie, G. Bailey &amp; D. Turton, "+
    "<a href='https://www.ourdialects.uk/maps/walkway/' target='_blank' rel='noopener noreferrer'>ourdialects.uk</a>, "+
    "&copy; George Bailey, "+
    "<a href='https://creativecommons.org/licenses/by-sa/4.0/' target='_blank' rel='noopener noreferrer'>CC BY-SA 4.0</a>"
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
    matchHTML="<span style='color:#8a857c;font-weight:500'>Inconclusive &mdash; this doesn&rsquo;t point to a particular place.</span>";
  } else if(starPlaces.length){
    matchHTML="&#9733; <b>"+starPlaces.map(placeName).join(" &amp; ")+"</b>";
  } else {
    // broad answers: describe the REGION (e.g. "the North of England"), or "much of Britain"
    const rn=matchRegion(surf);
    matchHTML = (rn==="much of Britain")
      ? "<span style='color:#8a857c;font-weight:500'>Used across much of Britain</span>"
      : "&#9873; most common in <b>"+rn+"</b>";
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
    if(SHOWN.slider){ const rv=SHOWN.raw[br]?SHOWN.raw[br][bc]:null; line=(rv==null)?"&mdash;":fmtPct(rv*100)+" acceptance"; }
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

// click a city (foot-strut only) -> expected local IPA
const detail=document.getElementById("detail");
cv.addEventListener("click",(e)=>{
  if(!SHOWN||!SHOWN.q||!SHOWN.q.ipa)return;   // summary map has no question behind it
  const rect=cv.getBoundingClientRect(),sx=cv.width/rect.width,sy=cv.height/rect.height;
  const x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy;
  let best=null,bd=1e9;for(const ct of [...cities,...(SHOWN.shownPlaces||[])]){const dd=Math.hypot((ct.col+0.5)*CELL-x,(ct.row+0.5)*CELL-y);if(dd<bd){bd=dd;best=ct;}}
  if(best&&bd<=26*PXS){const rhymes=(grids.footstrut[best.row|0][best.col|0]||0)>=0.5;
    const cut=rhymes?"k&#650;t":"k&#652;t";
    detail.innerHTML="<b>"+best.name+"</b> &mdash; "+
      (rhymes?"foot &amp; cut <b>rhyme</b> (both /&#650;/)":"foot &amp; cut are <b>distinct</b> (/&#650;/ vs /&#652;/)")+
      "<br><span class='ipa'>/f&#650;t/ &middot; /"+cut+"/</span>";}
});

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
  version:"2026-08-11"      // bump when the question set changes, so responses stay comparable
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
    json.dumps(icelolly_uri), json.dumps(bread_uri),
)

with open("index.html", "w") as f:
    f.write(html)
print("wrote index.html — done. Now just: git add index.html build_quiz.py && git commit -m 'update' && git push")
