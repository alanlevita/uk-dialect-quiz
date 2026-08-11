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
<!-- favicon: the Jesus College arms, taken from the College's own published
     favicon (already drawn for tab size) and inlined so the page stays one file.
     Used with the College's permission. -->
<link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,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">
<link rel="icon" type="image/png" sizes="64x64" href="data:image/png;base64,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">
<link rel="apple-touch-icon" sizes="180x180" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAACTDElEQVR42uz92Y5lWZKmiX2y1trDmXSy0ceYsysjs2tAESBAXmQBvOAVL6PegG9BdET0JQG+AB+h/Q0KDTIDJMhmd0VVZkaGR4ZH+Gyz6XymPawlwou1j+pRNTN3czNzj8hGHkDdzNXOOXuSJUvkF5H/F77m9Td/8zfh9q9+ZR9A2vzu//CTn/xwHO1/N/H+381C+e5OCOOJd+YM14uJIgnAY4BcfJfb+l7Fht8JTsHjcA4USJL/1Wzz3mc/v/17G75r85LhmO7ae93XXKs+9zyfPYK88Jrye0Fww5V73MW7DVAUfc57r5/D9aPK8F/3Nef7ouv6Jsd47ncqqLt8owdKHB5Qg4jS6eW3eSfPvbcAZqqKiROpxKQSo2nNHq00/f60a3/7pLc/fL44uv/7o6P55jM/H07pl1e/6plX4BVehZlUTtzIuWLqXX0QwnjHeylEpDdEsTRct1y/mOuGoM+9an95h7/mabmvMc7XfbmXWADX368veoPburbr732NY34X13Xlc3rVmYgAzr9w0Vz/vJklIBu0SGlmfpV07KJVc+e8EuVVr+uVDHpmpYy9FbtFUd8qitndUO4feB8q8URMklqyfObOvs6gBXqMzgwzQ0wogIAgAiLDsrDnf4dc8S7uwvfbN/BgX+ep3TXzu/79zxi0gJkRLZ+N2uXq9iL4zc70VXY/XJNcOy99SaN8meu6/v1fdV0XrsbyF3QYa1U6DO+FWoSRODwORUn2YoN2EAEpxJVOJPSm/ZlLjcPGZ5qK9bpzh9+SQct/AP7qZz/jgw8+uPhlXaibSigOfDm9HcqD98rqzs0QqrE4okGnqgaYw9lmUzO2Ntn8QAWIzliJshKlNyMoVNFR4whOcO7yAZhxNYiRre+yqw9uc9xXXep28cDlyoM32Tz4y+uyKwtMUDEiRovRoUQMk7xIS/K1FQZigmEXBiSXKxNBcMaVcMXewHVth03PWzDPnM/WdYuBS4KZsRblxCVWTglO2CWwb57KhIgRzZ55VheLyCQ6oBZXVM67pUY9TDGa2M6D2FWpMPedeujCChkFKabej/Zd2L0dwo27RVXUztGosdYcbhfb0ebwtDYX6VVwanTeOK8SZ7XSOsMnGHWOSe+ok1DYpSczkas3abhpNtxwrnmebWP45g/88nO29Xex7V0hr6htg3YKKkbnjGWhrAql89mgCxXGvWPaOaooOAR1XDWgIXGwrcVz3ZvKK17T9mftOffmMsKTrWPkOy4ClvLFR4x5pbiJUtdGqcLeynOwcoyioAIpXHM+dhmi9MOinPnA1HsWyWMti3PnJxWuaM3k6Ls06GAmznnxIq4U5wtcUTuP84JKoscIJvkHwW+8gRPMZaMICSRC54xYQbPvSSNDkuEWQlgI1cox6oSgYA7UZwOS4YkkM3ozenLUZc7wko9bkI9tWw9QXsIrb4cySn54nSjRLJ+3CcFcvi4nOMnnBuASeM1hxjoYaSx0M0esB4NuhXqejXocBSdCcqCblEFzeGIGUQbvPlzXxrsXdmlsz/OkX73bXHriaHkXSeRQzw33qxJ34Yg2odMmPEoYHUb0itvxjN7yFHue0Bqje4Y/N/waykIgCOIFNH9HXp1GL0ZEic5IPttEoY4AwZl4h5f+NXaeVzLoKGJR1ZJZSliMqt0ipTJhnJA4t0hQmJmnxlGSH95F7KaCjyDJ6BSWQVmOYbVjpGQ0QOwhto5kjpAGAxaQYUGYQmtKY0aD0g6PuBDH2DwjcxTDEfUlt+krnhbJhomycpHGsrEVKYcMNUJljuC2vGiCELMhLMU4r4yzidFMskG3K4FWcGJoyoshufx+AUh5kUYzWsnbeucMU6M2x5R8XTKc2zc16M39T+R715J/ohgiQo0wscDIHN7yTqOAH3bJXo2FJuaF0o4EbgTCLYcujdVRT6uRqoUKR1l6nBPMQNWwlNGQRpQzl6wxY2kqUY0Ulc6si0pMqioi9p0aNEBELJlpMk3JLDaaQifGuUTOXFJxRmvqRjhqPIUKzgyN2QN5B1ILcQyLkbEK2Qh6Saw99AXE2hPNKLvLhy0xP5zkYO2Vlc8GHdXwyRipDWGBXUCD6Rsa9CZpS+TvXlliIUrvwHmhwjHGMVJHqfncdPgCH0ALYT0y5qUxD0YjOfLtnUDpsLHQqeD6fE2ky/i599CKshYdrisnYkkND8jgMRW7wFFfxaA7zd+/dkpTGFrkXW0VjXHnKFQQyQvRi2ACjVPmJM4rJRWOMgi1d/gAVis2SRQRamdUkndpGxLjJHnXWTplGTS23lAhVObEmVpKFk0tGa9uzK9u0OXV4MtENsmZOAF1Jn0weq80AiMT6k5wnRF7JYkhleBmHp051lNoTGmXStsnXAcxCDYTtIaqBVkZ2hqS8hbfB1iPjHWdDdY1UKyE0OWQQ1x+CDI8UXnJ+PJ5Wbm3HF60pdFU0HijVaXtjKoB+ux9LICMBJ1AO4ZlZax7pU2KGSSVfO8OHP0I3MJIS0X6vMCtELrKWBVG45SYDN/ma8peUl4Y57/UdQ0xuRgEzQYcS0c7NvoRtGas14mlKlUvFJKTcjz0wVh7ZeGUdTWEUOfxIpGXyuCu0E+M2BhdUoLmEFMlhyqtKOtgNMFEcu4grheciLitfKt7jVzhlT205yKcFQRxIq4GxjjmTqQpjVVpbuWUsQlT5whJSZoN2kqH7ILuQOuMpkk0TaLtFTFIhcNXgqtBC4ePYG32ur0z2kpYTY22zh5sgqNuhLE5KnP4CxScZwzhZRMnQSjMMTJIIrRB6WqlL421Cr0Xxgou5hBIHTByxBm0ldGo0awTfVRUjRQcjASZCakWgoE1hnSW4+nKWI1hUSVaMUInzJIw6h0jHAF3aZRDXPuNEA+5hN78Bjb0ECujnRiNJTpL+M4YJccMT+UFC7AudEhyjd7nMMEdKe1ZpCwdoXCEA8FqgTPQueE7y87HGyuvrP2QIAfTMYEqealThrGCiHjc64I3r27QXKl+ZU9W4JiYozbHqSoLEuqEKBAKmAQILnvQpEJK0EeltUSzTjSrRN8rzuXqYXIOdTnZKlQwEzqXH8CqVuZB6QSqJEySo7QBEhuenL7mdbHBw3EkYK65GrYwxXtBKkeIW4mrQVLoE6RodL3SNYm2zQathRGcI3oHKXtJZ3kn0SDE0liVyplLdGpMkmNPHSPLBu2Ri8re6z55BwQcFdBieEsZvZCEeSN6ozZhjEdMiCq4lD+YzOiiQpvzmlQ5yrFHCk9p+V7kEBP6AfFZFMrCJ5IzKoSQoE5CqUJ8Hij+nRp0B67YNmhDgMqEZBm68QI9SlvmuHLmhFAEJqVQ9BA7WJ1B1yeiS7RJaTVn3SGBtEaIUJox6oRRIyiCVMp6ZMRSaVXpl0ZoHdLkz5UieJG8Cxiv9PC3Ya0gEMQRFUInKIk2Kq4Sxl6gEqreMUr5gfcrEJSuACOHDZ3lkMNFw5aK64wqwnQpFNGT/BDKlEqHsu4SsTFGLRQ9jIeFajm6eQZq+8ardAioZYAZiwYqARGjj0rvlCrk7XdkjjI5yiRYb8Qu0XlDg+UQywsuGm6RV3PROsZrR9VnhGPlFJNs2K0YJGMUhaoXRgmCQRxAEHXZBZXdn9hDXyZTOfOvBwy5ikKMSqqVzinqjVDmZCr1RjxPuDZhRSIVSvKWIawIvgOv2ahDb/iUQwj1EEOGnOiMsDTqNVQp497eyZtY6FcKK14yHl53wqgTVl3O3NPYSN6wAoIXyhbc2mgj+NKQwlBvqBtwcgVpDLdUfFTKzlMkoStBQzbmPiqsjXIFoy4jD6UIQXLBgm8aZrzo+gZ0xiUo11BHqB2EoYynPiNOJKNIGRQf41iLsiqGpLsUrBBUQVeKrgxpISShREhDXJowkmZ40EeoWmHSCbUJwecYKrn88ycPOZ7nAEoVdpKjUY83iC7HYCeWkCHbFTEazR5MvV0kbjJsPSbZ8/dmLDV741aMs5Q4jYk1iu9g0jj2W88OjtLn7dtep/LwggXrgXFyHKgHg1WhaGEsXEI0b8MjIBm0qsMDHBINGcr4NpTE1WiTstC89a9NOU3KaZ9oValbYdI6DpJnIg7vrxZZ3sh1bSWIZQ/jJMy8Y1E4zBvJjLkmjpLQW4b10gagF7m4OapGUtAEMUGnxtqU5IzewVyUJUqXco2h7AeH1wulE4K/Grr+SQy63GofyqXMjLUkyf0LHmHPPC5l73LUZ9iroWOBYxY8RciZcxMGgN3DUDdBfL55azOsUBZesc5YxcR5TKyaHMPup8BNLbjlCia4HGpsVdteJyyTi+KKDSVhsnFJSW2Rwxg5apUj13MeI2fiGRW5mJBKoy2MOBQOguSiggTJizsoFoy5F1I0lqKcxkiDUajjZvTcscANKZiIy1v2UPR+/ZI3z3QOFgojHDPxrNQTJRvlsUUaUSbOUxUOCUI7FLjECaZG6gxE8EHoJrAolS43sdA5Yx2U1RDK+FYYRWGcHBWOMHyPpGc7Jv9kHlq2blIkV4KcwFQ8JXlfWyTlNCUWXmkKR1MYpRM0ZGQgDjboNg1JIT/AhWWcGQ9SD14tKY0qoygUmnsIDnygQOiGStb1xqVXva6LhiAzHFA7x0gcQYQmGYddYukTOKEbw8jlwFRdzu7V5SXvt3o+OmcklHUBrhRSMpZJOdNEisZeEqbquekK9ggI+V4odqWp6E3tpm5omCoRRmZMkmc1FHTa2jgTZWm5Alt4hy8FLQRnDmsTfTck8ZXgRw5LsGyUrh+qgWGoO0RhFGHSZww/BEGCXPZ86Ztx0W805NBNgigMJVRYo5Sbcm0BVgkWHD3QJyVGQ1O2aBHB+dxs2ydj2ScsQVF5plPPqAhYm9CTHjc3fA+lEyrnciXL7I1uy9tb4aZTrhhi2SJJLqx78FOhGHucczRtRmws5Z0q+NznbUNY0qfs8UMQqtpTFD6XflcGy0TohVodY8mJYBwWqb45IODZxSvZmfiUe2wKEUbjQLkjxGCsm8R8nnAxMa48k1FBgWPdJ5ou5UVbOooyG29vRieGBvCVo9ScAE5XMNN8XZ6t8PANvt6oQcv1Ti4HTnIvR+FAKs9kp2BaelKEftnTd0qKGYcNQXBD+BGj0baJmGA2ckwOCnb2CtbLRIiQFhE3VOiiswHSenMe7EW7kEouopAGr+0do52CvYMyV/BOjOXK6DulLD0+GN7lhh3roY85QcI5piPPbDcwAdyZsO4joybj07LV1fdtGfLFkh1CIlPQZIgXJrWnvBVItXB43DBfRPpOCSPHrAQvOQSJms8xmYGTnC+EXNH0pVCNPJMkTFphFmAqQrXdj3LRlSf4of9MwHDO/uQGzfU2y+FHhvKpeEddOerKE3uj7TydV2yoxIXCUVaC+LzSQ8ilg7J01GPPeBrwJjSFpxXN/QZ22b1l35IXu9LUM3SLbbD30gmjyjEeewRYrxxF4bAIIUAo8kLVof/SLPewFIWnrByjkc8x8lrBK6VcbYz6thbp9QRxE1qp5lJ8cEJdeXQi1GtPUTq0NZxncDxCWTmKymEGReEoQkaEpBTMOTTkHEJUKJwwdsLYu2F3G4z5z9lDP9etbUEXajkTTpovtKocaECT4VyOq4rS4ZxR+nxzYlKqkUfNWK8i7UrRPnefBTdAeRcNSPZc6O11YDt7BvEYjiJyMWYUe6NtlOBziDEZB6qg+XqKfF0GaAWxznCYLxyhEFJSYoLY5zGQTR5hF82Wzz+nNxVH21aoCOQebIXYKutlRH3Gv8djTy3CZBQovEOcMK49tlsAwnjsGVcekVxIsw5Wllh1CTph0nkCnpHL4UZufBqGH97ggg3f0iaWT3Koi/uhMVw76NZKZ4nCO0rvKEf56bkB2vLBDYmhMSkCyXILZb9INGeRfqHoMlGbUPrcxnm9ncVe8PdXvY5LrD1ndm5YTIVziCrtQlkQqcrcbzEbBajAiSHDkAKSt2MY+j7IEF6zyJXEdp6gy6GTvwjarh7/eo/2m3hGW6AQXqByjtKU5TKxOlS0dZiDWRUoCqEIQtDsRcbBU88cTvLC9SE/ixgcmoxlm3tDtBF2m1x4KpzDIfROr65MZ0PDVaKjA57+eXnoDdTlgFIdo+iwVnHO0GhQGiEMhZABq83b8ZCAecm4sglNVNbnkXat2MqoGmFqjrHzFy2i2yXdN7nibes6to9ROcdEPH2CbiloVFKdQ6YyeJzPlqKbbjoneC84l5GfqEbXJuJa0UbxK6Pscy9KSU5yNxb8opGpN+J0to3BOUbe6HC0rbI+UfpOKUaecempQ353avNiLFx+ft4NsWWfrzkkKBIULfiV4da5SipDuC7+2wufvoWkMDduquSn4Q2mKvTRUbWS+2sTuN5wxXCBcrm5XnjDIaHEQKLhG6xqhNA6JsmzI56pCxS4C893farjTUdPbBn3CNhDcCmx7nMrpzPBq8MVQ+M/eczqotPNcTHgICq43lE04FrB955aHTMctQs48dcr1d9qLJ1RHGPkHIojaUTb3AceolGMTLzPz4+UAxRxBj531DFUAjcPMPQwXTuKFqoozJInDB2Qm0TwehekMbS2fgVu/t0nhRfUA/kvwYSpOpzBBKODXCrthTxxuLmoy810gyaY5S3MGdRR8MllSCvlJv6abDh5C7+aYHwb8F26CASM0oRZzI3wI8tDvkkFi7kKenEztuIvRbChCmUmeIUiCj56yiiMklCboxr6GNOwa21H0t/GdV1g0gKlZfKFTVl83Smq4HoQUcSBH3D5nPTbRS/4ZjxNEMoERe9x0VGZsCOeAsloCJcdg/KGr+tbDTk2nqXG4RXqYfX1mhu+LyCwF8S9tvFkAgVOCoMqQtErheY2U3NCkpdcyfaizfYroI3nIThkWKrUvNoKB623XEiQ7Kmuzzkam2YpQcxAciwZLHemVSqUKU+8C0IaoL7tIdyr53/tJN9QlijZ6TJxQhEcrTmiGrHPhqsuxw0Xo222nS9dGrSzHGuXCuUwPuZ4czQM3ylst1mDufFySPbwFGaUmvsdtiezr+bz21Pim4Qxe2mvhkuK9AlLRjTN4cbLtNEONAlmlwYt8hVXIZdDsC96+GaGl02/xrBI2T7GllFfeNohjBIbiGhy0hws/5lbCPIivb4Qr57/9jU8e57C83//dfcon5vggmdEoAoONaFHL2E2uebd7Wqot/G8wXKV1A8NXl8Vvw/AkTnE/iQjWC+Ne8nlBW480hWvc+0yr8SNkkeNokKniVYt47ne5QFMs6uz/19zQvZVHu5Ze35p12fXuuDkG96mdBE7ylemcM+e/1cZ86u47g1PxGYG1AhqBMmddm5rZNyeiX6fu5y+dQz9uzPoF3hvdy12un4TbOvEwlD6nVtM52b9mZM+lqXVo4qyqFwRHCkZamo8B6O9YroiOJ/L07CZRL70eNc/Y2qY6VWPKM9azsXIzvM8zjeCBW0zGP3cQp44d/kjl/u9mWGqz5jUxfVdrAS5wo2x7Tg0dx2YEyfOO2JMpK7VtFrju97tmIZd74qRD95LjoMTX73ot/db+xZi/z+5QctzdrcXxN5D3UKkGGKutVn/SDi7XxcLd/NGuvX2XXdz/6CoqkpiTBa71gxwF6HHxmiHwEYN7wNSFnnaYHiKliIb3gC72PMkT5ekhGrCkl5kTJfb++bvl8NdYq9XxHm2PTQPBaA5rBIfCEVAfLhAddCIJcVSypPVWwtONT0Toohsn/vW8YdEJlS1hCJIWq/t7OnT/uizL9QdPvV3ujQty2pvJxS+FEdrid7MNnOkV8KrrZVir4FW/NkbtF0PLF7w9DeTzMXwAAxYY/EoML+/Mz6pv/d2Ovjrv64mb787moxq18aobdMMBu1eGBqKfDPv+ewWL8910t/ZjXvBcR0vpgS7HqG86Nw1JsQJ5WgsZVU4N1/q0ccfrx+vFm13+MS7vrVbIUzMSSVuqPgOX+zhm9/Y/zUZ9Ne+T65mzwDq0FgUXdqZddV777L3Fz8JN9//Xl2XpYspad82loDgPLjM8JO3aZ+3aO9om4bl2TnNapWTuaokVCVSlBfNOSRFTAnOUVU1VV3jyuLC22dvqEMoom8cSpNtiMgN5+8Dakq3bmjmc9r1ii4mzDlcVVLWNWVVU3if0RNTnAilD/gi5O+QgXJMszffPncUkiYDpKxHUhTBtetGz8Q6/8kfY+NdXGvqkmnaDCvYFjrlvum1/a815Hjx++QKKgFGwNm4KLg1m7k7N26U792+uXP34ODAe4+qoilmPkTnRABVzS2RLiDO4Zzj/PSE/slT1kdHJFX8dILf28WHApxDzZAU8SlRVxWzqmK6u0NZlBdGmwOYbNQ8gzi8MXjoYktx3mPO06eeRYxY29AenxDXa1JZUExneOeoRyOqIuDMEFUK76lHI+q6xhfllZZjSxHTq4vRUrKMbATnvCONxixv3ODWzm4by7ItDXVmlyizXT6nP7FD/tMmhd/U8C9mFUUY++AOyrq4Xdf1QVnNZkUxyy7CgXeZudT5LYO4+m39fEH56DHhiy+RPlLs7lDeXBAmUyyEDEmllAtD4xrEIWWBn7rs+TfZGfl43+XOtl6uaI+OCfcfEB49JqzWSFlQ7O5SLJeUyzXlqGZoGkHqbMTOOQJQeH9ZkvXhmTjBXEScy/2+AIVjr6rag7I8XzgfaxFxf16RxT8fg34eHOYQvDgpnJPCexdErp63uKvx4fb/aELnC9LjJ/gv71N9+jnadpSzGeXRLqEeYc4RvacPnr4s6WcT+i7StA3j0YhRKKmDp6zqbOxV/d3cAFXi2Rndo4d0n36GffIZ4clTJusWioJiekr1+CkynRHHI7SuSHWN7Uxp2o6yaai9ZyTCyAf8eASz2bPAng/PM4wQRFwhIvJnbMD/LAz6On6pICnTfVlvZmlQCLiWql96mAtb7lmcnrK8/4D1H/9I//En+M++wDctjGridIL6kOkByoJuMqGZjmlnE3T6hDAeUZcls7JiZzxhsr9Pffsm5f4e5Xj8rT/m5vyM488+4/wPf2T98SfovQfI2Tlln3Ah4MqS5B19UZDqijidEHd36Pd3sYMD3LimEsdMhJ26Znb7NpN3BD+dXVs4CdxVtx2FlMy0N7OMlnzbize3RGxApVdstvvz9tDX7TWqEWOyPqY0UFAP/5YspYRzTsIQ8wK0XeTs5IyTL+6TPv2C8t5DRo8P0aalqwqa0Qgb+n0pS2w2JUzGxKqkKwJr71h4z2oypbl5g9333mNnMmKyM4OUCN5dZkeve30pDRW64uJ368Wa4y8ecPJPf8Q+/pTJ8TF11+UWTefpgieSe0QoPIzHMJ1g0wnN7g59XSEhsBjVrA8O0KrC377F5GIDUDTH1CpexTl/mZemlGJKZt/CWNs/Cw9tL/Hv16u033zV6/OPZhlrvu4wnS/pWmV9tsSdzJnM14zbSB8TjXcsVemCAxFqYNL1TNOcHVVWKbEwowmeuLdHI8Lozm0IgTCqCT7kZEvjVnlQXuGOXSbBXIMdRQLtoqE5PKU6PmO8bJiKsUY4N2PuHeYdpct0ZaNVQ7FqiE+PWdYly9GI9WxCe+smq9mMNgSkLLcgP0e6UE6wZwszr7rDbs2smf0zNGj7lt9/eaNyM7k4EXdZRUHN4UTZ/t3guYldS9+1uBRJzpGqkr4MrMYlp9Mpy6pAgYnlAkPVtEzXDeOuZ2rGovC04nDLJda2Q6HDXcpVDJPerwtQq3O4ayGTaSKlSNKBP9V7UnCsvOe48BzXFVoUTETY7yNV21Gu1kzXLVOMxWjE+cEezXgCaqjI1eYgB945kpk4567cU8mv78Qe/uwMegPuy1fExLLxyBsC7EE75avu2XUm/q8+iWwMKfXEviP2Hecnh8xP77Nqj1Fp6CujnFX0IXA2GXM8GXEqQtO2+HXDk6blTtPyVtNxIxl7LjAxWKwampMz9PETmgcPWY1GsL9PWVW5Z/sNJcAxdqgZsW9Jsefs+Eu6dEJfdnS10UYlFIH1eMzpbMJpVbDqI6xWTLs1t/rEW5q4lSI7faTqE7VzzGdn9EfH9E+estzdI9wRfFEQiiJ3/b3EPOrXPovtPo/EBSok2x+013zG34VBX7QLbmIJudZju72rKmg/THE4ww0TK9vbk3G1rPfMBedR4Wf2OBGHYTTrJfOTp8xPHzM/eczp0wcsu8esw5J2HDHvsXJMnE5ZhIKj1YrD+Tnz0xNsueIgJn4inn9djfmLIrAjjrLrOX16zLr4jDNxdE3Dzve/z/47b1GNJxdnOtQengkdnjFeMy6BsJxwmRldu2a5OOP85DHrxRHLw4es4yPaacvxpOdk1aAKVb1PfbDLWoSHjx/z+NF9uvmCg6Lkx/WYf1UG/gK4FZV6vqR89ITzwtOL40iNpm2Y3rrJzt4ewXvcc0dIrnbeXImqhCv6MpdDGgOTUswhoHOShx3c5ZykDZ2W29DJxn4cz1TM/1Qhx0AuvtWQIM/xzioDFatl2oJic0G2BR1v7Vl27ecrF5ZzmCX6rmE5P+b46ZcsTx/SNseksGIxVZ42jsVaoCjwpaeNicPlggeHT3lw+ISz1ZqReB5OZ8h4wk5dZm65PlIdnzJv1qyaNWddRywKJjf2qSbTZ+JOecVtuo89q+UZJ4f3OTv8krQ+wsICO4CTReD358aiSxyI8ZZ39H3Pw9MTfnfvHoeLcyY7Ozy4dZf1eEpVFoyayH7XMzk6oe07Vn3PAmPhhVSVTGYzilB8hfje1fOzF+U9W89OBaLPJDs+K1NcEO3I9d5uu2Y/f0oPvc0G3w8TG4Ve6pBc9BMPS9IEUgFxmkn8QhSkFUKXL1QHpiG3ZcE60LH2Lyu/IOCcx/kCH0qKqgYmEHJvSN8pp0TWUUnNmnbdcnp6ypPTU56cn3O0bvJWXFfcdMaNIHgVbsRIWizpFuecWiJOp4Tvv8edvudNvUTccO4B8QUiAV+W+HpMWThk3dOc9BydKqu2YXV0RGxW3Dt8yuPzU56uVnkKfjJlWlfcqCdMnYcUCcsVqVvTBmG+v0N46zaTpkFVXwo+3VA39BjlMCPphyzfiyBqlxMdFaSZoEGQCKxBuvxc1efWddHLyabkNt+t9H8qg66GDNlQeowVytoSwTInsNNMGGMyhBkxe+a2hu6GYCOHW4M8Fdw6s/JTZJaRjTtXzcTmnWxJo3G5RV1HOSB7/aoes3NwF+c97WqfdnXIYnFML8ecLYT5cs580XC8aDlfLJgfnTBfLkldT4gRAeZ9xyd9y07sMHP8xBKTYWGtLdPjNpsB2Otoy0t4aJGtgHIIP5z3lPWI2e5NUlLKoiZ1R6R4gs7PuDEX7uwZzXrB+XzOZ0dHrJZzzo6O6KMSxGFNy9nJCfdCwR/KmnFd08fEfh+JlugzvnzhiOwlYmYsU5G1GK0ZhVkmoxm4+txGXagz1BlxV4h3hDQRbGGER4YtDVSQCvAD45QaKkpnQoPSuGxLCaXsSuAW9gpg9OuSNWbeX5e1OgrRLAexFWdlvjtl7Y11Bc1MYCxEL4RzCD4Pzark1RoH+tUoijqhHTT/lK8hYDRFnKeqxviiZjzbI3ZvsZo/oTq8j3YV7dNILy19s2B9PmdxPictFmjbEtSYSeaMkD7yeL3iw7IilDWjcc370wm+KqhuHiB3b1PuzHAhPAevesUdT3JDVChqinrGzu4t2tVTlidfot097pSJZhwJZeKTozn3jg45X85pm4YqZAix73uYLzirar7c2WM6neLCjDSqqJ0j3LnN6M5tir293Hi1HeurXek02sxy2qCU1UqWyQgI0eXF7LcLsgxSdqWxmglxJpRi2IkhHkodhgLELthN1RvJaRYwEsvUBoN9ffcxtG76AKAvoS1h5S03yHTghzmrXoy2NJa1MS+MZW+whmn0aOnQqaNuh8XhjMZnQkNvysRcrhyll4hLL4bbhOA9wY+hHhOKAotGf9qzrs5J7oySwMw8Oyb4lDVQVqqZckEEYuJ8ueRRUXCrHnF69xbv3rrNzv4e4WCP/vYtDu7coazrK3jypt31dfAi72AyHjMej1nPS2y9ptUz9uM5aiXeDx2CZYnrK85UaQCLMYcQfU/bdZzGnqeF5+Bgh4N6zHg6ZefuHeR77xHeeovZdIZ3fkhQ5Su00HOLfnJGF4xVMNpg6KChUqQcJKcaVrVyUijnrdEL1B3sVQ6dOUbtEHZIZmWNpRGCUTmj642U+BMy+LfkideBY0EKQWthXUJUJXklrA2neTU3YzgfKWck5ucJ08xOmlyB7gSmreC7HIfPC1iVRqmKi0LRCj4NfG8MTeTuebddnkspJFISZERFzVSzeuGtUPOj0YR7MRFWa47dgieaReWdCCFFbN0Qxx1pMoIffo/qL3/Kzbffxs2mpLpitLNDXdeXieCARcs3D54vEwZ3uf8IEHyFiwVh5RidJw6WPaU6JrMd9sqSG+sFHx8f8cXREW3f06bEOIRLOem6grfuEN55j+ndO+zdvk198wZ+OqUajwk+G7Ru0JlrDkJNc0cBuatAB6Gmc8vE7GUDtWVlsK4WzkfKofQcH0ValLF39KHA9gvS2uFXAzd2AU1t1AXsmEMRfL/dM9V9twbd5ZA3f4EIpXM4L7Qhk2S3MVFL5hw2B00B6yITly+XmSHIlbCaesaVEQScGk1SzryxDFmebUczj3A5cE7zkojHBnFIfUd/OkcP58jhgvp8TdVEdnGEouJGNeF0tOSjusb1LZ0mkgiFGZUq0xDY2d9j8oPvM/3rn7L/3nuMq/qinziT41ybCHlVpMiyUcvgNcHo5wv0dI0crymOV0zPVky7nhtVze3phJ1mxLrveXRyQpsSvSo4R1UUTKqKnZ0Ze2+/xe5/8xN23nuP/Rs3mIzGVzdaTbmd9BrUeF2ZoSQzQLXBOCMTzo9dpnYovaOrYV0by05ZzCNtl7CRZ7LvaUZGZUZos9rZuctKX9MCRilT9VZc6kp237WHbofEEKDSgcC6FxpJnGmkiZEd8zluE4fXnPiJkaXCglA7obRM4WopK5s2piyTsuwNr2DRKFQo7aoQkH2NYSiDMZ+c0d57QP/pF9gXXxIeH1KeL9ldt9RtT5uUG0XJ3mTKWCP9UAn0COOq5sbuHrdu3+bG22+x9/ZbjIYuu6vNlK8JN10UmbKnTrEnNg1xtaZ5/JT49Bh3ck5xvsQvVpQxUlIx9Z4Vjv2Br7o3JaXcs1UVBXs7O9y6dZu7777LzffeZefuXcZbfSIvAyPqRc4kjMxRJWHVG0uLLKJiKozwGV2yrKJbmFB5hwSjci4LI8WMbGSNGGOhPYteEStQDYyTYOqphxH5xXdt0F0FyeUTKpNj3GXjPOuMVhMr06zaFLLUbkxK2QrjwlGMHYXA1BxTzbokpGGy2zLxofWKH3S+i0FPb1MWfm6rjFwC/1kPXOhjpDk9ZfXFl7QffUT35QPs9BTXdljXE7uWFBOFCNPxmKl2LFOCtsWHgtneLnfefpu333mHOzdusDMaXduW7c145QHl2Mwrmka65Yr100NWDx/RPnmKnZ1iywWyXuNjpNREij1V31F2PS4jaWhKJFWqquLGrZu8+967vPvOO9za32cSiueGQy86f5Mc7wr52Y6Sp+4dnjQovybigMM5BN8LFcLUe5hCtEAlwjg6QmO4PseD6ow+JbqVos4RxKg7hyRHpR6PoysBnr4SCc0rJ4W60bdWKDuhspydylAFtBKslEwJpsP7EMrSURTZkG1ldE3uMothwCl7qBsYxazMFPyr5Qqqymq55PjxI5aff4Z++YCqaal9yFVFNcw5kICnwIsHVVSVoizZPzjgrbff5u27d7kxnVEjMExXm2W9xRDCaxn08xASMyN1HeuzM86Pj1gePUGPD6nOTqlXSwJC4T1tyOd7CfJo/jGjqmtu3brJu2+/w90bN9kpa/ygxHWR5pnhvf9qo94OOUwoo2Rxpj6LC0kJWma20aSGdkpRC5PKo0HwveGWkFqlNXCFIxlID6EzKoPKZU0e02eFN79TlCNtV4kGZdIRjgmOzmWFpNV4QC5iFo0xnxk51aBJxrpTnJLZ30e56220cri1MOuhdrlsalsC8S+kILlGSeCdpzPlqG05nM+R8zNudD17ownlaEyoqgw7xYa+XbFuGrr1GjGjKAv29va5ffs2N2/eZDIeZ4NTfYaa602/NnQLJrCIPY+XC/qzU/bOz3mraajLmrIo0fGILnriqgTnchyseW6wqir29w+4ffs2O7MdfBGGjkS9TELl5Xzftpf0lkOPmXr6AKH29KMsodd1WQU4aXbvMsg5r5PSAU1hhCqTDAUc006YpZB3cruKDv1pYLuNYxDDHOZcNugd8XQBrITFKAtV92mQKxuwTWKWzm3J4jnlKHMqT/DU0VM7Y8dyUilkMaI0RHTPVYU1u8IS6oCiKGA0oplNON/dQWZTxuuGvqzo64pYliw0cd6vOF+vWSwWtKsVoSgoi5KdnRn7+/vMdnYoqmooi/kBm5Ur3vFNeOaN2TjvKeqaYjrDphMWo4pVCHjvuOk9qQisqoKjquDQG/PgSeJwg8YjZhRFwWxnh/2DgwFazEasGH4zpnWlo+grCitDLSEOPA0jHLvOE4PQV0Izzro6a5/oJUOfIWYqME1G7xJWQaiMuoJCHWUKVGvYlUAtPjdmObPkzETEqsyc9Kdh8E9kNVEGXZWJeBbOWHpj4RUtQVEkZi9ukBVkC+jqLPHWVkoKiktCjWcqnpmDWhwtl8xC36RhyntPtbPD5J13mJ3P0dEITk457yJRlWWMfNKs+Wyx4GS1pGsatOsw5/FFYDKdsru3y2w2y1i2KRfUW7aRanvzEyviHEVVUe/vMrlzh+n5ORYThILzwxNSUpap59O18oe+5WnbEk0JGwUwIBQFk8mEnZ0dnHMX0NwwBPFCuocXtTfo1v2vxDEVz0ry7rvyWb+7FQWXqXQZFM00GMlB76ApjS4oox5mkjVkpuKpcLSkXEz7U5W+W2CyIcYc/KMM1cPScmZrQ8FCDVyA0ju8Dh1VCpSCrxwNiRZl2SdCZ8x6R4kwdo7KOTqBZC/RurKFFGyKBbOdHd750Q+ZTWe0779Pc/8BRw8f8tGDh3xxcsinR0d8sZxz0qwvONjEco9zWVeMx2OqqsI5R4xxSN7kCg3ABZXvKxj35WfsqmcsAqOdXW69C66uWN65w/zWJxx+9DEfPXjAw/kZXxyvedA2HK5WdLGn8J5uuA3OO8qypK7ri3wixkSM8cLAX2ZBbod3w1wM3nKy7gfVgXWvNA4sQOlclkaWjOcblikaJRt8q4qLxrjP0silZcL67LQMHUYNuu+a286JmKiYOTG2Zh0yd13WvSt6KNrM5hNMqIKj2PTHumHTK6BJjvM2sl4n0trQ1vJN81l3UDY9il/XnbT1gDax9HQypX6/Jr39DovTE+5/9hlPflvyu9Njfj0/4bNH91m1DVKUFN7nEvKG6kr1ChWYc5cd39fx5zf5ytzYnmoy5vZoxMHNm7Tf+x6f7O5wP0U+nJ/y0eP73HvyhPlqlWFMM4oQsjGLDFopOcH1Pi/CjWd+3Z3FDUbtYlbFlWCZQbV0jL2n3Dwzt2kdFlozljHRtwnXgOsyLOvtso9+4HKznmQ93XdP1igDQmZy2VHogYpMSG5dlto1FUKfqVndVlO/H+R+2whVA6u1UDQwipLV+dyrS37ZsAi8DzmTryrGkwlLJxRnp8z/+BH3Y88X8zO069mbzihDoCgK/OCNl8sl5+dz1s0a7z3eh5cxxdc36KGP1vsCDxRlyWg8Zue9Ne7ePZaffcqhCI+XS1bn55QhUNcVfuDtExFijDTrhvV6zXQ6G4y5pCjK1z5vEQgItQrTziEr0CQU0TEKHu8uefr8sMCiZtSqa8CtYdo7Ks168Nt9Wm9CqvCVDNqLM1ExTHSbQC1IllTYU8+4zxCNdVnValOLULlkFg0OVB2zXug7j48wMk8gx4NxS0TzmxiEDWzyztkVbzSejJnt7DKeTChCgRokzTSxIkIYeOO6ruPo6IgHDx7w3nvv8fZbbxO+tiixTRVpr2TImx9VRZxeGckKITCeTJjNptTjEf6ibK0XMf0GRmyaluOTY548eUJd119z7l9FGGNc0vgO7bmSFa2m5vFJ2G8MjYJrsuTGBbE7g0EPoHZUQaNHIlQpAwhhIxktl1iiqlrXd999Urghl9f8k3sPyCXMmXks5jbSjRZdGrq2dGvrysMqQ4fbELPZMK+XLHfeXWd6fynY65mkJ3eHhaRMvWevrNgrSmahYN3Hi63Rh0yb1bYtTx4/4ZNPPuHmzZvcunWLH/zgh1eT4RQvDPByO391AprthScb3cCNxJspru8Zm7HnA3tlZnbqy/KyNco5QigQcSwWC7744kt+//vfE0LgrbfeesY7x9hfhCVilsexnuu59VkPLcIYz0g9ooZFuXjGtpVAuosfwQ20jo5BmcFxOa10za6+++akfJLPDDBsEsOC3BftU7Z6HUjOoz3LSOmHFe9dDrr6wTPrKyzRDSWKc/4ZL+QAp4qkRFClFEfl81iSbNCFokBVaduWx48fU39UMxqNGI/H9F1HVdV474fJkiUxJYqiYH9/nxs3blCW1dXQ4aWTxcvYXIYhhas5S57sJka0bXExUjhHGQJxgA+dy4mgiHB2dsZHH31EVVd0XcfZ2Rm7u7vZkPtI13eYKkVRMBqNmIzHTK+1wn5VjiAIgQzBFpaTzKj5+eq1Z+yGZNtvnvHQJpye/70Dgv0dJ4W+60yremAgySdwyRhvV+SEN6NpMrC5b9a+bLYwBJOsW5enuOQ1PNxXT9wuV2tOzs44my9o+i5rDDi5IPiWEIgx0vc9x8cnudvMjBQjjx8/Zn9vj7qu6fue+WJBjJHZbMaPf/xjZrPZlkHLlQamN/FKqpzN5zw9Pubk/Jyu668kwt57yoGiYGPQy+WCs9NTHj16xM0bN4fdp6Hve8qhGnr71i3u3L7NeDJ5Jg9Rs03WL88PRnJ5nKEbz20pxG7XbS7k7CSTaOpzYvdszJgTsfZS8eK7EQ1qgSibx311KesAwTBoqHxdvLYt+sg1MOObqSlkgvJNUpRDj0tPd3R0yOdffMEnn37Kg4cPWa5WeUsfYKxtKMvMaNuW46NjYh9ZrZZ88cUX3Lp5k53dXUIIpJQoq4rbt2+zXq+fGWfafN83wDYuziPGOCR5fvi98ujJE768f597Dx5wdHxM07WXQcFQht/eYR49esjZ2Smnp6c8fvyYW7du5WLTEI/fvHkT7z27OzvPH8Wyr6s/bCQ4rj+oZ8PDy2TPLv5fnt8/YgrfvSRFL2JJVZMzNTG9Pql7ZXnxLMHM9huvRP/bOiTP3l1jQ0r1HLaijLVGVqsVx8fHLJdLBKEoS7z3PH78iN/85jf8429+wxdffMFqtcJ7Twghe+GBuch7P/yZcduTkxOWywX3799nf28vl8Nv3eL27dvcun2bu3fvsrOzi/dhq/jy1dDYpbHLRTy+Wq1YLBY06zVN2w73LCvNtl3Lb3/7Wz7++BMePXrMYrGg7+NFkXEzRV6Wl3HwarVkvVqxWi45OTnh4OCAnZ0d9vf3uHPnLlVVsbe3x87OLvVo9Pxii9nlzws8tMhVNPV5l709IQ7PDMdcSXSiqtLJq0Scr2PQnUWpkwoRI3Gh73RNK/DrZG/t+ZQNzyhcG2a5b9cspef6j41XPTw85NNPPuHBgwfElBiNRngfePT4Ef/4m9/w+9//nqdPntI0DWGIG1NKFzGj99kwRBwx9qxXa5r1mvV6Tds0OO85uHGDO3fv8Nd//df86Ec/YjbboaoqVLOk26YA81VoxnYCmVJiPp/z6NFDTo5POD8/p+/7DL+1Lefn53zyySd88vHHnBwf07btYDxu8OKG944QAiF4vHc4J6xWK9qu4+nTp7Rti5lx8+ZNbt++zQ9+8AN+9OMfc7B/wGho9r/uZ8XIZHOmWxNBV3dQeQlp4+ttIxvsWbYIegyzJKZRRFtpv2uDFktmybCI2AvnLeUbbGFf3SpoQ26smSjhBQmLqtJ1Hefn5zx69Ijz83MQoe97Hj16xEcffcSXX3zBarUmhJD5kwf81szouo6+z/HoaDSiqiqKshgkLTyTyYS7d+/y3vvv8+Mf/4S//Mu/5K233r6CfGwneN8kXIox0rYd8/mcJ0+ecHJywnw+5/TsjMPDQx48eMCD+w84OztDVYdF5y8+33UdKSXCgNRUVTUsMqUsS/b29njn3Xf5wQ9+wI9//GN+9KMf8d57722hH/rs89AXe2heVOeyb/aMr3SymGQP3X7HIUfyXnuTTfk95dzhW3xtPEVMWEwXfQkXBDVDF9nGEGezGaPRiKPjIw4Pjzg6POTBw4fc+/JLjo+Pc392UeTy+GxKUZT0fc/5+TmLxQJVZWdnxs2bNzm4ccDNGze5cfMmN27c4J233+Z73/8+P/zhjzg4OLhilDrooWzHz89DCzaZx2YyxQzKsmJnZ4fVasXh4SGLxYIHDx7w8NEjHj18yNOnT/O5x0RdV5RlcbHDdF3LarWm6zrABgPe59atfM63b9/m7ltv8c477/D+++/zve99j5s3b16B8lIcJjAGyFBTEktJJGU/Im/8gW6r/14SXnRiiab51jy0AXxw7Zdd21oKPqkSTSUNyOJ2Z+LrG/jFdmTIJm5OimTBTbuQ4R0MxDtnVVXLwcEB77//Pl3fk1SZz+e0bUvfdZRlye7ubp6OBqqqZGe2w3Q2pe97zIz1ek3f96gao/GY995/n7/66V/xk5/8hHfffZdbt24xm82YTmcX1UMzvTDm7bDiq72yooMH9N6zs7Nz4VWb9Zrj42OePL0c43fOUZUVdS2MRiOm0ymjIfbNIcgpfdfTx54iBMbjMe+99x4//elP+elP/5L3v/d9bt26xWQyZTTKcOQGiYldl1lU/KXSpyYVsfwDSNYIkmFym0tlrVesMj/LK5QDnF5FkUavppjGL8F+/hIHfDWUQ8RUXEoiCZGE8K17aKeqztDgg5ZleeGdti8jBKiqirrO+iij0SiHFlXNnTt3aNqWFCOLxYL5Yo53noODA/b39wG4e/ctjo6OaNqWndmM7//g+/z0r37Kv/u3/46//Mu/5J133n0BZOjwr8nqXxQF4/GY2WyKah4gKKuK3d1d3nn7bc7Ozliulmi6ijtvoLrFYsHZ2Tld1zEej/n+97/PX//1X/Fv/+2/5a//2/+Wt99+58XHDiEz+2+9yrIYQjEVNH3rG3BuoxZNThPy6lXwVzPoprFmXGi0IqltZwxv9iVyQaJmJFVR1cJ7rav6K483GuUHOplM2N/f5y/+4i9Yr9a54UgTTw8PeTRAd6O6Zm9/n/F4PMTBebcbjUYcHBzw7nvv8sMf/ohbN2/xXbxCKPje975HWVW8/fbbnJ6eslouaZqWru8uUIvz8zNUjel0ymxnhkimBtOkjEYj7r51l+9/73t87/vf5+bXnftz+lSqqsJlTFPMTDbSb9vb7+uGIZmEaeDoQFVNk1rsWREvDHoLMfgF2C9f16D/6vZt+/BazNE7p51ailhUIVluCL0oUdvrLtfrv1ITU0WHokfftl97L70P3Llzh93dXWKMF0supcjDhw/5/PPPOTo6IqXEdDrlYH+f/QHa2k4Wq6piOp3yXb6qqub9997jrbt3L0IhHYRAT46P+fyLL7h/7x5N07C7u8vdt95id3eXoiiunPN0OqV6RfmM2PeisRdU3dAdLF8VC3+zkOMiZs4tDqZENQX63qwjF4z1+kHl2/LQRyJ2w/qULCUT/dZDDgNRM982TTg/OfGPnzyRsq4viiIae9TytIdwOcZUhIAPgclkclGkUE20bctqtRoaeRqqoRtvd3eXmzduMJ5MBjbQlj5Gzs/Ocq9C0iEB5Y20Ym6w5kvV1wznbYokIYQBdvRsE6Ofz+csFwvW6zU7OzvMZnm6ZhNibeA7M2MxP6frIyn2FzTGV859qIQ653ChuOj3Pnz6VFbLlTMz74J37lsSaExctDtoVO1TSs3wq28H5fjgA/gpH1wx2CPvtVWNPdLZZetFvkmvuRHZZbxxKYwuuKhaLhbLqrt3r5QPP/RHJ8eEIpCSojkBFD+oPbkB8Qghq68WocjGLoKqDvHmKYvFkr7raJqcCK5XK44ODymrCjPNSVbfE1NGVVJMqOnFonmTUysb6QcnQigKiiIQQm5n9X7QW4RcODo5Zj6f0/fdRbx9fnZGWZY47y4Xb0r0fZ8T5BhzV+HWuV9gwDGrYLmyxHlP6noOP/5YTo+PvZj5MhRevHObneJVIsxtX76lnUM06DLZZ1KjWUm/5mJOVuhVxYm89BFfyUMv12tdlGWMKfWoRTI5/BuX0LHBayHi1KxcrZfV08+/qE5SdLPPP6OoSmJM6LAty1Dtuj5Fct3wNl1mOkxxX3j1rSb462jF9T+/rddzz3mLkWlT1dRhwnv7vC8YX59TyDHsxbHgAHu6ssQXgdi0tI+fOH3y1O9CKMvSe+/lol31NZzVpgPPhpaHzpS1Kg0WI9Z0HWvI01gJFf7jf3R8gwm8V+u2WyxsvrcXI/SGRZHcnCRvPtTYlNBFjNA2TfXk8aOiXZw7XxaEsiAmhRgv1r5d64vY9ip5VCy3iYYQhjK3G3LOXOpOg262yFUvfHVRyBu+yquDstsx86Yd4LKS6S9Cks2OE2Maqp2XEzabr83VRPnK5i03tKpaUeCCx9qOYrGUg9Oln5mFUBTZoHkzYqOXHtro1VhotCalTk3WiUzVd/HmJ0++0c1+CYP+4NlKj3N6pNo3SAvSZWxnu54vr3HZdhWtzDS54hGvfQzn66V/cvjUtaYXTdnOhi4uu+qZLv+0K5Vb5/2VCQ/bGlnaiL5fYKCy1VAl8F0o9l0/Z7aUqJxzBO8verdV7YJg5tKYt5oH5Op1PNegh7f2TsA5QlL2DKnNe1+UviwK772TbQqHVyN1NxOcbDac3oylJhYpxpXG9dpi0+3uNpv3/wKR/9Pi34teCgS8WQ99gWCcn6fFZNL2ao2adCZvStzs+R5aRCQI4jVJv17L2WLOedcRh2JGMXhP/QbLSJ4xTvvOFZveRGjyJsIgPyzqZqBRq8QhRenSaBrKorLSOe9wYq/Yp/4cOxLAkhmrlFim2K9SWkejuSvS/WbrvZ+s1/Lvv20cGoh90zT9WNdq1iYzVXvz6kfbOi0BKAxcjKS+p21buphwTtDB076sQV+GI5dZp/B6E9zf1Ws7lOKF5yzfiKo6DPdunZReExo8ariiJozy9H3weWbjzRi0IKYD4ZAqjWq/Ttp2qt1eiq9F5P/1KMfVUHaz5URWq+X6QOcJVr1ZbNWIdklM7i67Br/R9iRXY+ccUgAVYhWZTsxvYlrnECfD8K1DXrK4dJn0XTXof06eeXPur7v4hmkdETXEDVmGCJXzbupdOXXO1yLBiWSUYzi4fEN29023griLLNeSKq0arWm7MutWSPc/fPhhlG/ToF8IH7btsrF0rrBsjW5lymRLkkHewEq+iAOHkfcarPbe6qKgMkNUswzFAEHpP6eY4c9lgQwhR26VSWBK7T3TqnKzoiimzvsKvLc8UqJv6JgMzRudGdGk7aCLZpHvSAVrcMu6STESsFrHdNqh52tNzUIjU/M5QRvChJfZnr7Kg2+XWIMZhSIlQpAMU7kBt3UXumL/YtCv5KGH+5hVq7zVPjAuS7cTQhg7b5V4ERG3abCQ1z+mbFCOpBoj2nSm7dpS/C5k3S6HT372Hx0fXPSSNCv6kw49W2lanSaxHS0YCeIlzw9u60TLVxjsdaOWa0lhJrAxV4DzipPc1kFUzd5FEMRZet7w27+8Xg5VyYUsc85Z6b2NfZCpD27snJQZ9pO01RstL3iWL0RAhi5MGeINydVfTKyNyrpPtu5N4/XN/bfAz761kCNjgpuDpabv550yXzpdBk39SlNZOk+Q509cvQAueuaGPMdDiwdfmPiAeVNcskHvMEeSA8PSPy+h9T8H72zbKIkIhfNa+8ImRSEjH/zIeSm3JJVftKO6r3iOL9oVhq62TrF1b6lt9PXb+tzLemiAXy8WV67lwPtlMpsvNS1XmtbLlIgZQ930bT9DPbs9Q5iVluS5kwybbUAxRHCFOF97CaVIcJioaea8sw1zB//inV8HOcl4vxXe66go0qQodOK9VQPD1WWy/qzT2IxibZSzvmI3vurezdSwtjdbdca6VYvfhUGT4x3kk/VafrH1u//jvXttZ6xXyZaN6nKV0sB2NGgVsxmYen5wawIm129QrsnZoANolqXtaufC1Pty4n0oB5X1a/Q6/2LP38g7y0VjlA6oiROx2vk09iFNXEi1eC1Erui4v2gn3YgI2wuC4O1fu0uMJKlJ25ut1prW87Z9xqAfjkbfiBHudbrS5b8H7frYNxabPum61ywl5S0TzrjnZdRc8nMkjG5gVNqczPbWpYPFBhM3ER/2JBT7PoSx965w7s8aL/5n5aEvjUEL59LI+zh2Po7EabCsvvAVaEV+lgLJ5T9fmDcNHjwzvYJHFLMumjVLTd2p19f20K/DD20GLIm9S2Hdia1VtUOpvRPxdtmdrwOpVSDThSHQYKwlE2kXJlQmlAOZuA4VKUVxWYFJKgg3JNgtX3BQlIxjp6tc9hY2Cca/2ObLe6OtZsYLsk0Rq5ykqZM4Fecqk9Jvh4T5ieY+sK2AOklW/dWBOKhQIehl3eKi4juEmN6EhOBEkgptq7paaFo/WqcrBv0LsP/Nt2nQJ6PRM+HTsm1TVUvTW1ip0CSsNBmI9u1a0DTcwkSWB1uERO+MOjl8n3U83IY9V7ZQDhEpRfzYe9srCtsvSt2NpS7NJKXkbOBAtWF7fNnk5F9eW6L02UPbyEgTXF+LBO/QJEbKSNJg+Vfj6SjQeaMJWRk2qEA/cIHbVhOGbELMy/jD0JSMpsMW56qr07h6pko4nU7tu/LQANKqxT5Z0wdbtd6adaHTdWGuFrC4CT3yVZhlLcK1NxZBWUwMDSAt1GqkHrxmtlLvhpKuE9TlfkIVpDCvO1WVdjXquapbqgmWLhmX/iUMefkYenDVG5+TlUQkVVgUZ7IOaguXVd1NIWz0Uwav3GE0khn8m9ogCGWEoEbZQ0gDZa4XzBnqIXroBLpkrM2sNW2WmhanMS4WKbXwekW58PI3AH7+nNWyMuv6qPN1pecrp6vzIqVRmYJaZqZ0ivnByhozFily5pRlYfQzwVeCX0LTGoVCpYZzw312YskjjcvreSlKwulEi3iQynQWo+9T9N2GR845XKaRkn+pGn5dyJEN2ekgZSFYwNlISSMkeW9uVauduoSoULXgFHMG5qHDWDtl7hOLUulGgg+OUQ9Fp5Q4Qsp64M6DeLFYYBoUVWGtibnF2JjO16pHhzEdN32/3kBkG3jwZ7/61XfqoW3R0i7R5cx03oo1a69pFQZKLDUKNWoTcyD9YJTnPrEsDa0cZeXw0egK6IIQ+gHkF6PxWV8alwO3lSp4SSMf+mko4si54IwiczxnwRpz1/o0/gX8eC6yv1nvqkoaOjicqowUGSviHdIFYx4U0cQkCWWfJdjEhH4Qs18UyrxU+lIoPThzdAF6b5RDzKHOaIKyKs3UQxmNlUUWSbsu6Ulj6fGjtjtmsWiuJ5M//YbO+rVFg47jui8LWe1pscRovGXnuPRKExI+GbNojMyRnGEjQUfQFYM6Vmc4g1Ht0ShYC1HzVnYiytLpQKSeY7KRiU5EuhH0Ps/DWd/3JFO8pOwOuEoA+V3YyfU+YfsKr3jZ7vlNW3ze3MsN6GpnSodROofhXWX4CYQxeI9Ih3HuEqsgVIUxNkeFEIHeQ1dAG6DPZCmUGMlDqgbkw4ymMOY+cS5KMqjV0aVIq7oUeKomD2jSEZkHlLgVev/iJSa9X9mgP/zVr+yDa/WPR3HZqTJ/L1VnRc+y7iQ6j628ypMQUU12kAL75ikKh584ipFHgtGuE61lEce+dKQdiGuIjXGikYcSWaA2My9VCuwmj4tiO5piZRZFU0CT6oXw/BVXJN+dPdszPvCq4b4otv8T7iBySTmXTEkK3plMhHJPPHvmQx3FN2Tn1AbFlSY75tjT3EujwbAi5zgpDdPbCVLwxDprE/ZqthCVM4mcayIlGHWO1CaNyc4Lk0cF7h7r0QnD6BU/+5mzDz5QeQVVktfRWMGAR33fGPG4Vz0KHfOyJfqAxMKYu0Tj0oUMxdSDVg5fSWb3b4zUG32ldLXRVgYKbUyc9okTi6xUB03wrCleJJhEo05GteFwc45oRhChGEgQ9V9wjq98+YvKhhLNUYpjUpTslkXY994dqHNV792pKQuMU0n0Xml8pjAfueyhzQ+lsGRYysWwJEZbGJ03VkmZa2KpSZqUhE6Qzuhj6kU5d+KO36pnR/DhenNuv/7kE/fvvwuime145mIsZrlcr3brh30f77uKQ1VrUm95EN1DVGyFyplLJJen/vrLgkleGGq0KXE+iKB3kphLLqOHlOO2Uh3BBNPk6qhhB7G9euQPqtJ5oNOES0oxLLa0aWV9w0141zPwzezhxQS1XspU+K0RrxRjHjDdSGYMv7fN0Ol3MN4lQ8YlQHAeMyUk1RLVCU5vF5Xdqkb+RlGEA4L4KL43ODbonbK0zFhROQbEIssiM2DMWL7vrRiK0pmysESrikWk6sW5KKTerE26TmrzpDb/Cx0tt+/rN51SeVWD3sQyeRv4+c+FX/7SgNW85UHj7fMkPFprWqSYscg6OZIY4o1GjORTXskdqDOKQgg+g9Ztr/SmWBwgIWdU6qSKwp4F6gvwT90YK2/44N6uK9+NCh0XQZMXCYqUgwbApUG/SQm25/Nnirvcv3UgknID/YC7GGS9atAi7oITb7MyroQlz0QjryLl/tzFaCIiIQRIyZYxatN33ahP/TsmcoeyPAih2BEvzoQmwciclSKyksxZsQ6GCxmG6xhaSgd8OQGNKU0ylinZMiWThOwmL7sp4KJxHlM8T3p+bum8Mdb7O23avqrf5p9XQu9ePSn88MNL5dyjo7nVu486s8OF6byIUas2uFEQfOFJpWGl0Lq8com5fc55yUB1hLbPalQqgpSZp3naCjudY1c9xaD5LYKfiKtuSuHfHlWp35+1O7s7sZhNiklZlmMf3Mag1ew7waWvxMouN0dutAIvqAaGodyN5vaFYoDbVBquTVRfspm/yRPNu8cwOY6qzJvGTo6P1/b0cH1z2fg9dTJ2oRp7L6ZGaUIdhbETWu9wpaCloy+yUmzcKpzgICo0aiTLHr0RZeQcU/XctCCY0an20Wxxaml5TuyXFK8LP7+yQdsm3Pj1J59c4UvwI3famh6dpnRaG0slzQJCUYrTiZN+LLSoaaeivQ39sTK0EebRLfVACUXhqKJjmoSpOeooOI+ZQwonbqcoituoW4dyGceT5frOHQ7ef2905/btnWlVFVnFFEkpmcsz/G/MIC4NbaOHaMSBm9k7R1lVuOBp2jYzn3Y95ahmd3+f8WiMxki7XJJizLriAwOUxojGNIQrWVc8hzD6xnYYS7mj2XsvbiCuOTs/109+//v10Xx1Nj1ZFJW5sqhLgvMkU5wqlYiNxElXgNVCOfK4IqubWdri/ZBNL0eWRE4+Dw7U6pl1gZ0YSF1KXmmS2dlc0/yxds1eVV25wA9zePudGPSmxi6s11etpCzX56ZnbeJ4qpw5s/FOUfjKe0ftpB0bImI46E1FUx6SvFiWXvAF+JGjKhyjzjEKwoihL0DMcCKlD672SNIkS7N+rXKu01n83g9/tPuTf/UXo+l4WmzUpFJKNtAWyBvzcDCECxfFW9bnc9rlEi/CeDrBFwXz5ZInT56wWK+Y7Ozy1nvvsre/j3Y9i9NT+ralGo2Y7OzgQ5ENeuDWcM4h/pLM/E0ZdIox0/cWxYV6wvHRoYb5sv3933+4LhfL6HzZU41MnZOkionhRRgFRyoFrYUwckjI8teuyxJnG3ZdG+JrBAovVnhnk+SZEqg7RyOqTli0picLjSdPumb9h6dPrww37T/bYvHd4dDDViHv7uyk07NmXSc/d475zMnNEnEjcZJw9CiI4YIQihxb2+X8T+7ECjkMcYOccmmSK05bzYd5fXhZ9biq7aI/O1sVfd/d3Nmt33v3PQk+fOc4WFNWnJuhbUdtRhic+ZrcjDM2mCpMccSyRCcTYlVSVzU70xm+LP9klZZRWeqDycTu91FstRIr8+BEdNAPM4RBYCQOFRs4uuRCLMj5TRPSZUFr81zF5ZCqVke90W7Pa3TdqJ0dp3R6FuOK6s1d0Csb9IfX/v93e4e6d1w2N/BnU6ozJ6xrpKhU/Lo1Oqf0PidAhRPKUjAnF11Ym21LI2hnaKO46CkQyi0HKxuut75Hlytdt+vonzxN1qwtyDMChd/JS7uO5eERq+MjnBnBh8yZt1igXUd7fMLx6TnL3R3iqKYNAfOO2nv6+ZzZdIfRrRtXs8DN8Mabv6TtVvLozGJYd6Hu+jFJy6BWgkniUvU3mFCpECOkzuh8xq11kLh2PhuuDYiSxQG+MzPXi/kefI+JIgqqwmKpevhFtz58ulrMn37+QLcD6Ld+/etvVEx5Iwa9tS0YYH/8I/zQ2qbQ6uy219PSyaIWN6mT893KsNaw0qwITqoghEEf0C5mcYw+Kn2v0BuuBd/l/ungHMql/mHKyIH1TePadlXYg4cyf/TELc7Odbq/t20Uwrds47FpOLl/n+OP/sDZl1/SL1cZ4RDBq+Es63asxKGjiri7Awf7MB4RgJH37O7usf/+e8zeuouvCuK6Ia3WeTea7eDL13Zh28P4QxpEF6E9fPCwbZ8+LSZdX/hQVmUIIzfMD25mNIOJVMnRd0pnmb5LCgiFUHgheHfR7LTRLtRkpF6xzix0oMmkjUqD9p3ZSSPp4Wf96vHy6fkC0O2h6g+fmd34bkOOi7t0vF6vq9KddpSnTty6wDFRj7bKEsF3zspSGJWeEATcZVOuqRF7oesE7Yyqh0KHYYCtSYFEppBKgGmqfNfvxNPzdvH4kTs9PFz70QjnXRCzoDEiTvCDdNuLK3YvEzcPSIQa4kPmwus6lo+fsPj4M/p/+gj99DPSfIF6jy9LqlBSOcFSoul6Wu9gNsFuHKDTKV0ZWJUly50Z8/mc3cNDRrs7hKrEh0BZ1UjbUQyY2FUZN3kpE97Agk4G7DsTjFtr1hw9frx4+tFHsXn4qJwlnYxGo2oaikKAzraUwRBKFUZRMBV8Ag3gSqEoHIV3g4DpIJQyFFlSJ0JnlL1k4dBonKW4XqNPO0tfLlP7GFiRGzM3SLz99DXQjlcy6F+C/c2vf23bBv3HP/5R393Zmadp+eTHyZ6aMfeQRubwSSxpkD6ZhYQVnRPvZGg64gLdTkmI0WHJ8Aq1uaw0y0UOg5oNc4vIKJTjm2VNm3Rph0f9+dPDk/LmjdVkOpkUTmY+eJ8l1hygJvJq1Qu5qKC4oQvQ0XUti9NTlg8eoF/eY3rvIaOHT+jXDRQl5ahiHIpMq2VK2yc6S6SzOenojHY6ZrW7w3pvh/VqTX92TnvvAbt3brP/g/cZv/MOo9lOJtO5YEfajATD1yI3chGiGWYmzonlFynGdHh0tPr8d787P/vHD7V69GTvwKTaG0/KynkREXpVI1MHS8joOV6F0nJV1hJIzIL1fgtIMgYyUxNITlwySRGiKsddbycaT09TvB+T+5T14hEbcsa/+ZvAf/gPyi9/ab/8U3jo29tUzoPzvHd+vuiL2ePG9H6HHbdqnaqNxuLELKDRINlXcEhfemzYng4eSHUBVSOS8CIyqevRbe+LlXjh8OTo8LPPz+XWzVCMKqvKauJ9CBeyvIpl1u5X8NAXVL3Dd5nRxY75+Rmrp4f4o2Pq+ZKyixiO5DKrqYsJs4gNcPvIBNd2aNPRNS2lQfCe1XJF17UsfSCcnTPdmeHffZdiNLrYyE31gi74Cnz4dYtRM0m79x4zE48nxminT560X/zmt6vVb37L+48P066IvzWZOlNlhRFTskHJeBCfz9NGtYGlYauMz6czUMstw4WIOENWphymnsPYd09jf3xq8ZGz7iFnnG6e9gdPnzo+/NBet3r0ygb900uqr41elQHLE5qHnehna9VHp7Ffjr3f3XfF5TSKQrLnV+82cJsMXSnp0pgvw+JcsDAB2atqqauqOBIZHT9+ovd/++GiO9iX8Wwynd68KRIu42fTPNDl3Gsrg6Cpo+saVusli/US1/UkHFqNCU6Io4omCPN1w6rtiBjOB0bBMwsFE3LcOe4jnJ0jXYdbrjDv6UNg9fSQxdk5fncva50gz5AzfqPQ6Vq45Zxz69Mznn76WWo++UTutpFi50DGZUmfEqu+e+40/cawZQO3btR1ro2/paHhrHaCcyKdqXWqOk9peZr64yONJ4XI+XZXwsOuk7f+hCjHRRb6i5//XH7xt3/r+NWvIrDuzB6aymxp+vlh6g/p3S0JFFPnKBEnIhsWS7tO8mpb25ZtLdSr4oxmyoCNliVjJ3Rt44+fHNqx/a7v9nfl1p3b6c7eHkUoLmJJM+ObkfxeECxfjJIJiqWevl3Rt0u61NBKoh8VLHenlN7jvdCWgRPteZg6nvaJRR9xRPaKmrujMW9XNTfFM01KdXqOW64Y9R39aESaL1g8eox98SWpqrhx5zZ1VW9I4S44oOXlG/WercCpiS2XQQ+Py3h0LMmFwGwPE4eJbl++XL8ntpX8bT+zrbuWwQ5BJOdIopj1ak1r6Wiu6fBM4/k6xn77pI7L0t764IM/HWy3uaa/+vBD+eDpU3dRBj8+Pi/ee+/h0xQf9aYnEdqxuKJyBcXA2XVBEvMMSaK91EEH2j8pBp7num2E83lYrZdl+sMtW/2bf42mHyegeP1+nyHZUUVTT9fMWcxPWZ0d0a5P6FxLuxPQu1PiItB0kXM1jpLyZFLz1BsnqxV93zNJHXek4oeV50eu4P2m42aM7PYRZ0Ijwvm6Yf3gEeeTMVIE6rKgvnuX19AE5IomoBnNctnZ2dzvtt14JM6VIqWa0Zpu8wPK9e6RCyfzFbzvG4dtkqELh0hSLGJNNA5XKR2dxX61vPa5h6ORffinxKG3X/+vrrtyaQ0sn3b9WfLutEIWK6+1FpazDRm2KjAVxD1vlXz1PbuMIZ2Ycw5n4lzXjWTZ78Snh6k9OXUpz6dl5gTw2x1x39wqhBQT3XrJ/PSQ0+PHHJ88Zn5ySN/NkWmL4FkWwhdP1txfNCzFI5MxfmeKm885f/qER+dn3F+veNqsmY+mOBfYKQIH4zGlKUEc/XxJfPCI5ISmqpjv7BBGI8rJlDKEvOXLS7BP6OAxNmFAStb0XbNerprTe/ea+OCh3Wj7WahGYd+5WgRpNeXhQXju3bLn7KbPeY/ZQHjvRPDIRvSpaUxPjmN/9DA283C+0u0Z2t9Pp3b7z8GgP/jgA376059eca2/Wy71HQ3LoiwO2+CPkunEjEkQEeWy1VAuuxmfgQC/1mXmWG24deZKZDoVU1k3bTo+8cvz89V4Z1c9VIrW4jzObQcuyFfFoXnNXO67mhJNs2J+fsrZ8VPOjx+znh/jrWE8SliROGlbnjbHfPp0jlUz3pntcnN/RuWExePHPDo64v56zdFkTLt/g8n+TW5Od9mva4o+4Zo1xXJF3bV0IujODou7d5CDfWZVhQ8+9zEPYM1GCfb6dQzlcnOIGGKIM9Vkp/P58uFnn5+efPhhFz/9fHKz7fZ2JpNqjAvBORdzP4LYC+DNK/yDX63pvaHOsvycTCM2X6X05FHsHn++WpzURye9214jv/oVH/wJmpOemxw+vFp7l6QqZ3272PX+aVKemNlNjInfUhaQ1xjX2HyoNyUmw0RkUlfjOxpcl3TdPXmyPPzy3jzs7fU705krcONLp2aDhso3PbRdaLmLOIIvqIqSwhmjKpJiixsZyUW6vgPNaIbrOoq+J7QtOl+wODujmc+ponKrLLk9nTEqPbcBt1K69ZpupbRFQE9OcYslVddl2rNvsOw3SbdzOTMDTYvFov3sD384P/r1f4k3P/+iuqVa3ZnMam9GOyAbr0urtv253kySKivVrjeOOuHLY+3uP+7W55tiij0HNfuTGvQvwf5maxrcgH+zXMaz5I4Oy+Jeo/X7Cd6NZre2DVlENpVV+eY3LFtjHNR9gnNyMBoXtZk76ZMtP/vi/NO/+4d5U5X993784/Gtnd2rD1vVhib7lzv20HJZVSOmuzdwzlGPpjTLM0znqJ3Rrk/xo4b93R3e2hUWa2F+csJyccJiec7Z2TmmRiGCxsR50/DpaslsOSdh/CjBvkZiisxTZNW1FJoYeUddVZTesyF3sRcSnV/mfzl5vMSwPc73yxWHn3yWHv/j79L4+JwwnbnZdISp0cX+UrLu9UqR4hCJGMuUOE09T1N3vkjpnhkfnaOfctKcASmB+8XP4Ze/RH/6GtjzG4+hbw+j5kPAavr4cbdf10/jePTZX9X6Tmf6w7Wm9xvTKmygo+fcjG8Q0g54Z24CCSIyGY1lhnli748/+Uzvpb5bB2+z6bS/OZtdiTtfpntNtlIiI/dnjCc7WX977yZ927Jezlkun3J8+iXdKhKk4b2dm4xvjbj/dMGnJ0c8XJ5wtl4R1w0ueHan09xl6IRHqzWcHNP2Haka8+OqxO/tcKaJxd4us71dir09dnZ2GIcyx6MblMO/mAotDxuYDOxIMsTQ2p+cuvjgccnjp17bPlCPTLenbV6xfdy2grgwYM8ROEuRB33bP+m7w9Okn/fO/aFp9EvyMKz+7d/8jf+rD28bfMAv3wyX+psx6O1SZcq3J500zfnYtfeS8HmDPj5LaT2OsZoOrZxuq6nglWGWAf8MItRlyUiQ877zcnhaLJt1OLp5g8W/+ou++9732qqqw3DYVwtznCc4TyhKGOXfzW69xcnxLvOmheYpo6VnFgvulCPGRcdpStxbrVk1a0hKGQJVCDBMTc/XKzpTCjHe3tnlvTt32a1qaueIOzMmP/ges9u3mE4m+CEBsK9Ag8x0G2sWcMSUNKUYz09Om/WDR2l8cjr2arLjfC0IraptVLBeh+hQueSsK5yTqIm1Jg5j3z3s+9PjFJ8uvT55/PjxBcDx0WIhD95wb82bMOgLTFp//nPHL38pQx1pdb9JD+LMPj6L8Z5XOXYiE6MsaucsgMs+xDbYpbyiVQ9NTrn44NVC1XeTyXnfh8dP+vbpYWxWq3NXlZXDVcRYbkc6L7M7vMhteaCeHVC5KcVSCIcts6cLwjJSmHE+m7F0hlvNOVvMSes2T6c7h2mia3qSKuv9fezObcb/+q+5+9bb3AiBdVkwunHA7q2b+BCupcTy/MWtauJdro4PNr9cr9aHT56eH336+er0nz5i7/x8Mq7H1YFZWfng+6FIpbwcp/dXOmky2X0QESdCNOtWKZ2fxHh8v+/OnmRRzW/1Fd7Q92QOhb/9W/f2v//3wq9/nQujjx6duJvv3HvU9/cbb4+Bm5VzReVKvIhTEYlmJmZm8s13OwGxYV9tNW1K536nKGd3Hc6W62V772H39P7905tFMRqNx7tBpJQsv7oJWl6SHcO2YtStqt16jVt21Gc9HK4ZPzpj1PUUVUm3tw839qjOT/nD/Xs8Wa5pYqQsy9xonxI1MJ1O2f3ee9z8d/+Gd3/8E3xR0msE5xhPJlfO4avgRzMzVRMRB8O41/nZ2frj3//++NF//vWq/O3vZndW64O7k+lsIiLJORc1K/LZa46rXbDpSn6QeQSOVTQeLVQfHvbx7H57foXQ/MF0ah/+6rb9ORr05UleTrKYQHwcl2cOd7jW4qgWt7yjOi1EXO08Hblh3F6Bf2ETK27aDltNqGHOe3cwGdelOHe+bvX0jx8vu1sHq76u43vvvz+pyuriUKq5WWcz7vS1u4Bq5lJFSOs1cb1kff8R+uAx4XyNaxJh3VG0DXs+8L26phtVLFR5XFU8AtoYkRConGMcAjvjMTf397nx1lscfO977Ny+/dzFdFmj+/owzDnZTLxos173jz7+uPny1/+lu/vgCeOyKm9NdkKJcNZ1NINBv26TrWSMMzdZgzRmRNPzZHyxUv10Ze2TJoT0c3C/GKqJf3X79hsppnyrBv3hNZ/2XxeLdreoT6yQJ3cKPVJsx8GkcGJJB7rzjCa/Utix+UxSNQwrnZfpeOJGqmU/X4T7v/kwrmK7jKORP9jf1+mN6rpLy11L8gLe0k1z0hDSWO7JpHv6lMWnnzL/7B7Nk6fIcoX3AcoS6zuCKbOo7HeR3ZiohmShVyOkxMh7RuMJN27e5PadO9y8eZPZ7s5LwWHX/uWKP7DNbNhwuHi+0O7BwxDvPag5nVfu4Kb3M4eTIam015cB2lI9s85MkilnKfYrS0/WFj96nPoPTxs+o2D9i5//HP/L/97A+O0HH9gHb9j+3Bs15l/9yp7cuqXbF/rhfK73U3d8rumLaHypcNqZ2XXp2detUOtgeIX3TKqKnbJyVdv5/st7/ul/+Xv/+B9+o2cPH/XXvJnkIu1FGPh8r7dNNeA8mNEeHnH6j7/j+O/+gcUnnxKXcwgOq0tSETCD0PdUTUvVdgTNOHJmF8oDtbPdXe689RZvvf02B3t7jK4NIyRNF8WTr0rH8oJz+OBkEDdQoD0/OVmuHj7S6uhkfCum3ZnI2KlJk5RGNct9vHr9dFvUSTayFa0qxzFyFPvVWUyPl2Z//CI2H/3+8ecP79271/CLX1ysvl++IajuW/PQPwVj2DJ/Du6vQP5jCGmu+qQ1/Z1i+63a3bMUb1fOFfJtrKrcr2wBkbG44kbS6frszPo/fmZHn3y6PLp796iaTMqqLMdFCF77yGUMyfP7Mq/BfGbGej7n9N4DTj77nKSJ8d4uI1/gLWU6WXJTfeUdxTDBkuGzSBxoDXZ2drhz5w5v3bnD/s4OhQgW4wX9QkoRQsB/RUikScmT3E7Ai3PQgx2fHi/u/+6fzh99+KGNDk+q3aIa7018VYXCdSlpMttoaL9e4sSF0q+AsDbluO/Tk9jND1N8cm7dvY84fTRAdfCLX7iIbqqIb5wL7Y0a9C/B+OCD3LT0s5/JrSdPhF/9qjl5990HR6nrO3R2pP0PXM/7Cjd2vMcb5iS33uqgb/4qycmFgKIZTcrTFqNQVO/Pdg9qZ5P14fHZ4//69wsbT5a3fvj92Vt3794uxuOJDAO6ampe5Er/A1/hPnoz1qYsU0T7SGhaCEpoe4qUCCKkELCyQMsC89mzkxKJzA03nU25desWt27fZrazk9EM5wYixczk+eJ7IUP1MucB3l8axnqxSH/87Yfz3/0///ZU/+s/lm8fHk3eKUfTvdp5E/HJTFTTZqHK9nV+kztvlrc355wEEUko6xjTceoXT/ru8ZPYPzqO/RM+P5tfbuMfCj/7j8IHFxQYf9ZJ4cXJ3coScAA99+4d/2dY/O//aveth117vwt6msxuFFXN7iAbFnPNf5NMfmOLdkODg5qx7nsMrKrK8FZZ7IxQHi/X6cHf/+b0SNOyc6TdnZ2d6Xg8ET/cgphJ092LKjnX/t9Np/jbt3AnJ6T5gs57mjR4Zu+J3nEmxmNLnGii5YK/FpOEOMdkMuHg4ICDg4OMZjh32cTP1xn0RlfKnulZPH3yeP3lf/mv6y//3/+fuP/lw2pST8qbewf1TlGw7DtWqjkXBl5TFNouIjERZwatWjOP8fgwdl8+7LoHh6k/Y4u/5ReXdvFnDds98/posZC/uEqQ3j3W7mnT6ZPO7KTGrd8qra7ESe3E1pf8E1e2oZfCibceTB7RUhCsCl5mZUWZEvOzs7r7/IvyVNNq/7330vpHP2r15s3W5XvgNw3tW6DGCw1anKM+OGD3Jz+iD5754SH9fMHpYk2bEiF4eu943DV83EY+X8xZ9F3GyUWQIaSo65q9vT329vao6/q5sODzbChXSDHnnHjnRbwnga3Wq9Vyvmgf/PGPi+VHf2D68Mn0xrqdzOpJNQoBHwKSIpbSpSVe85HfEG7aplclYrSm65Xpg+OUPr7fN1+szeZDUC8C/O0/Exz6mdevgetVoA+Xy9W7vn5SIA/uFOXtpHrHCVUQhzNFrmboVwTsv2o73FTPNpMUcoHYDsTeqriuK+vFYqd8+pT288/d4smTvrlzZ1nV9UigzBqeNsAtm0xxAFWvjiEgzjG9dZNb/+q/IezvER494ujhI86eHnFcFbTzOafrFV8uzvhiueDRcsHZutkiScwrJhQlo/GY8XhMVVXPeuNn+KQ3SaptJE5k49Fj3/WHx4en9//w8fnjv/v7VD94NPphUc1uzfxoUpR1l/IOFFVxkgsp3yTMsKu34LI0ZZDAGlM510SDHrfGJ0ea/unj5fLz6Nx8gAUN4FffkJH/z8agfz+d2snt21fu2bFqV2l8eKPSj3qzvdZ00phV5Yt2+m8QYRlXe3Rl8NaNJjpVq7wLt4pqN/RW+vuP1ke//6ir9/bmN99/z8ZVHYqy9FdQje1RreseWoTR7g6uKij29+Bgn342o5lMOAmOx6nl/vyUz89PeXx2xqpp8kygCN65i0lscY6iKCjKkhBC5r7DXrpSuTkvBVuv1+2jL+8tP/nPv573//CP5Z2z5fjt8exgX1wQjLWmzCUw+OUr2oMv07NrV12yQE5ic6goC0scxb6bmx72pp8skv3x/nL5mNWqlRd59X9OBv2r//AfLvCm/+FnP3MA/+f/9J+W4+nBxyexHzWq07Oktw/7fpdMLJX7aDcOUQeBRtviepAt4HmLRkwx1F0+GDdo66kZXUqqZkyL0hdjNx4hxfEXD/TL/+l/PltiTQL3g5/8aAyuuMCdUzI1J25AF+SZ2Y2MSVejCa6q0bKEskKrkifLOY/vf8nn8zPun50yXy7RlFUI2JS97VnfuNllUkoXO9KmN8N7j4i7+JD3Dk0qA09J38e+m5+eLk8+/qQ/+81vXf3xF9VUpb4xnoXdomAVO+aWDDCPiOf5904AP0xsb7PRbCvEah6vIrjciKSKLTRxmHp92HfzwxgfnWn/6Sq5T1mtzoD03+ViCnJJvfHPz6AHql0T4GdPngj/4T/of/zgg0UI4dP7YdrP02zvaep+ZOhbCXZ3vbdqSO4GrcJNVCvXwepNlKlk8c7kjN5nskcxoVSkNMHUSKgoxriq3E41cr5pwvLp0fLL0xN7ulp11XjS3L15sy329yvA2TAlLe6r49hNt1/hPHs7O9RliQh8+sknrJqG07MzVoslliI+a/KRhsWxMequa1kul8wXC/bWa+p6dEmvu2XQ2ai3AU6HYJJS1E61mZ+dL44//2K1+uhjP7r3aGd2vpiWRV1JrbmHO+sHyiY3sC0nEUXpQxbNDCaUUSjS1XLNNbD4IsUoxNGJyiole9K3q/td+/Bh33/xuO++fGCrxwNUZx+C+8Xf/I2Qw41/pgb9/BOPR0dH86Oy/Px/G+OnXrjfeD33uNnE127qvAuIdYPCrDpj4C/BmeAUNpuyDlzFnVNab7SlYQGCZlK2ELO3sWQ4D2VZUFeldN78aHVWlUdn9dlv0ftvv9Xffved8x/+m39NVZVj730ZvHcXCZjqEApcfaWBXNGXFaUPlJMp/cEB46qGlIjrNdb3BJErk+Z+oNVNKXFycsK9e/e4c+cOO7MZ4/GYEMoXFlD6rkW8x/sgubSd4vxsvr738ceLR3//D33642fV/rqZ7pRhHCpfrIMZLoEiQS89gQr0m3sXlK7ITPxlcoiBTxuCzNyFr+5SNzJlQncLeNmIzLaq3XlKh0/67uMv2vaTJ7Z8/PnnXzbb07kPr+nE/3M06E18J//3xUL2L/mk4eHD1e9Go6dLGz9IRfn4wIddNZ1WUrgCsV6TdaLWFKABKRTGvWOUhJCE5IzojNYrS68sS6Wt8+x8pYrgKDoj6LCFStYBaQojijDZqep3dLLr1qvy9O9+s/i7yeTkPHbrH/3Fv7px5+bNm1tFC0spmnkvfmh7VdVLAvOUCL7A+cxKFELAB4/IVX6Rbe7nosiRTYyRR48e8bvf/Y7JeMzBwQFvvfXWixyDpJg4Pz3FxDHd3aUuS1O1/vjBg+aT/9//3Bz9f/8X2fv8frEfdbI7HVdWOT+vEj1idUIqzRJu0WfV17VTVkFZlUpX5/7sUa/45CiwzJA07CgajL6A3hlxoPdCxQSRtSZa03mj+tlxSv/4Zb/+/XHTPR0CswsSit9/QwHNP1uD3ugbPhgKLpvXH5pmDny5g3y8LtOsNXtPYeSGvq3osGVI0pVQaKYKC72/0J1ObpAVC8p5qbSV4QKk6CiiUjuoxFEMk4Eroi2JdGWSsD8u361He8XZevzZF/e6f1r8p7Oj5XxVulDc2N/bDT4UG3RBM0nLs3z6IohzmT9ELdMXtC39BhIb/u0icBygum2DfvLkMeKgKkvu3LnDO++8w507d597H+fn55ydnWHOQRHU+dAtTk7WT3//Uffof/nP9L/5XbgbrTrY363q6SjM6yTnElklY693eIRiYHKIYjReOS+URaX0JQQnGI6yN2pnKIK3HGN3wVhWWVXBUEISazolKpxppwvVpxH7/VrSb36/XH48Pzw8284QPgDlV79yfAcCv+G7WDW/vH3bfj7Ywc/4mfspH9j/tWmWh859uvSjg5XazlmKeycujkVyRdwNRI6NU9YCKeSQYqJ5kjiSKV/bYKxDDju8ZOrX3nlSAC0yV3HnjVOJHGuy6OHGqHa3ysq53hXrs8Xk9LN758flf+4+f/u91e1bt4737t6djUajsggh+ODlQuBaMOe8OOcJ2eavVg+7jna9pl2v6bsOVSV4f0UWww3UXn3fc35+TkqJUT3i1q1bTKZT/uIvfkJV5QmC9XpFs14TYySEIPWopixr7VNqHj64vzz86A/Lx//l79R/9mW9u1zVN+tpPSnK4KpSYmhZmoIl1HssOEY4EIgovYM2KOug9D7zPRQC0TlSkcli0LyrnYXEqUuog4l31ILziCw08qTv22OLh3OLn6rZH+fp8DHQ5666/w7hl3q5WX/7r++Cflb48EN+NcDJ/+O/H/n/28OH+n8Zj83B+o6v13tFmHrc3WQchJzEI160c+ZaUVmJspasXRgHXmkV6AOsS2Nd5G3ULMv3Vuqocbgg9AUsgnLkI8fSs1aVUr3MUmDSwygmqREvMYV133dnqqs2+G66MytGVVVtyGb6rjURUXHuhWN3T5485u///u/5zT/8Aw/u36ftugtI7joEl1IixkiMiRh7mnXDyckx9+/f5/PPP+cPH/2B//LrX9v//D/9T/aHP/xBxDnefudd9m/eTM1ief5Pv/716e//x/9Hs/67fyhuny93vhfq6Z3xaFRVVegDrIgsLbImsRaV1hlp2C3UZX3Bdak5DMu1f0rNXM6lOMQLXWGclYmnrrcj6y2qyY46u5GC+STuKPbdZ11z+FnXfPgwtv//9r6lOY4zu/Lc+32ZWW+gAALgS3yJLbaotuxuetr2eGIkhzfjCC/N3jhmOfZmwv/AltTzDyZiImY2E97ZY8xuYsK7sehWP/SAJFIk+AJAvFFAAVWod2Z+jzuLzAJBtbpb3epukRK/DQBGgKjKOnnzPs495ydbKr71aP2wDcDfAPD6a1B/v7Y21qrDVyLl+FSRTNfyWQkODvq7QHxwvjDatfoMgCuJ96dIMPsCh4iEueYUJeLRU0662uMwEBqAMW00yqIyI0/Od4c8AJ9tdHsIjBYMtSCFR1ccWs7SwDlSjjByqQw9UBJNZ2qTpYnaRFROh8PlB0vb94aDdmc4iKulUjR19eoRn9N7jzRNUSh+9iXr97rY2tpCY2cHB60WBsMBvMcT1NOsg5HdeMyMKIoAEPr9Ae7dv4+NzQ1Uq1XU63UEQYBer4d+t0vTJ06gWq/jW6++6rUg7u80Bqs/eGe09f/+lc6P4vDs5HTt3EQ10kQ09A4mNsLKIdKggQL12KEXOgzhUYdGRATHWdOEgSwcS6Yb6FiQBIANPGLxaMOi7awMjJUJr6GFKBSmkTg3dLa9a5JHy+lweTNNtptaHx6LxPIP/T69jt/u0fhyzrh7ZG6sre1VL39zdeBkxYucr7Eqz0hQqoFZ+4yOeOitHCCL1IY8OBfXhia4vKnKyPaIRHKnJhYMYdEXj563GDoHNkDREgIn2fSEiSulIpeV4kHX1zqdVrfzYHnULhbs2vSJXqlYaE7OzBYqlUoxLBR0r9tVe3t7GA5HAiJEhYgUKyTJCBsbG7j58U2sra9jMBhgPPMcR+JxOy7reRGCQEPrAESMOB6h3W5ja2sTIoLJiQnU63VUqlWq1WqYmp2FKhSSw05nSMN4sPP+h4m9c09P7bfDk4VCaSqKCrVymVJn4UcjwHoUPcGKQl88WsojIYElATGhwgpeZU+5o2Zg3sozEAw5s9TriUPPOyTWIzQkJccIiSh1gr53o573G23nbm4m8a2HxjSaq4/S3GUZBMjClwCs3yqgCcBfLSyMP9ej4Hd7GDcuFfSDCaVODcWXYnEveNHFSJSUhVH27EMP7rGnIRF67KA0QQWENP9giAikMr566gUumxCi4yxi66EsULMK005jUhQKYBIIxT7z3isT0ZkgmiRrdXt9q7tx4wfDfjzcuvida5WXvnllrlYsVkfDAW7dvInl5WUBkUxNTXEYRtRqt7CysoKVlRVsbmzAe0GxWIKzLlNLtTbjK6uMZcfMKBQiFIslKKUwGGgkSYLBoC/WefQ6HZqs1XD+wgV85zvX8MKFC74yMdHfWl5pP7i/HNPdu3qu1ZmYnqyXZqKoGDHT0KZIc851AVn7jR2hKw7KASl59AmI2GeFhkJW5PHjQYqDIBZB6jPD0461MN6jYJnqVmHSKRGAdmyCtjUHbW8We15+3DDmw/04bhBI5M03joq/SwsLnvDbdUD9rUfo9rHiYKy7tpr2u0VdWrpExakB/GTL28mKVcUTmqkMhbpotLxFnx2MBlItiEMBB5mbqRvrn3JG60pEIOIx8g595+C8oOYVJr3GjASoQhETkYOga9O8KcE0U61WCt6Vd5I0WLp1O7532O445+1ktVJKJyer+602Nra2cO/+fUrTFNVqlUQEW9vbWF1dxf7+PtI0BSHrZmgVQOCf8Ph2ObXVOQdAEIYBmCtQilGrVclai3KxiIsXL+I7167h3/7xv5Ozp0+P9re2Bzd/8MPB9r+848/1+6UztcnJcydPljIwW7TjWAQCBaKImFTecK54hSIx+uwhQVZEB6EAeRvOZRV4ti8HZMR/ydxfe95K4ImqouikBFwRTQPrsG2Swa5LN1vW3BkauvVoe3tpDOK3336bX8/ZdfO/pULwSwX0Mbkn/k/Xrqn7CxW54T8ZbDq3NoAv7jpzKkjkotKYLpFSVaUwxRotaHSUxUCLUIHIRQSvsvTv+PbLWILXk2S+iBqZdC0UKl6h6hWKwnAEpBA48cKABMyqHIRUEJAxabXTG9SS5VXbnfxQHpRK3akXL3EKFArVauXUmTN6f2+Pms0mGjs72NjclJ1GA/3+gACgVCyiUqmgXCojCDP3gCRJMRqNkKaZVO1wqJEkMWq1mlQqFTo5N4tabQIT9TpOnJjG3KnT/uLly8nk5GRse73B8MHDWD1Y0jP7+8FJDipTOijWSyVykgm6GGclo4coEChv0xEKxCgRo6QYPiBQRHBBVhi6XA1qrMttx5NXEiRK4AmiPWOCNU37kAMHHFoz2jem8cglK3smXhnYcCufcx3NHF7/LfE2ngZAH73BNwAsjkZ0AwsWLQx6c3PbW94D8fD8SAVXJKDpKuupAhe4qBUmWEkl0OJDjzBSosJMJ09yRD/mqmecD1GZ41ZECpEwyqxRyNwEMlaeHNkaH1k4WefgBagqHbxYqkyXxJcPlx4dLsVJP7y62T959Wrt3LkL+uzJU5XVlWW89+676HW7Muz3hYko0FqMtZQkCSqVMiqVMmoTNUCAbq8Lay2SJEaa2qxnbYwwsUxMTODMqdP0jZdewuUrV/DCuXMoV6ppPBr1GkvLnf1btxJ7+54+sbc/MVufrkzrMCoGAfeTJNPPznzOSY5RaMemP5qAUqBQDQFbIOhCppHt4TPh8nHhOq7mKGt1KiIpsPIlr1Q5VT6wxNZ7G3vXPLTmwSMb33sY9zfvrW8Nj9ER5I1KRQhf3vkyikL5jJ8ddncH+9PTO73hYCkJC3eK0LVTOuDJQE2WEehIK1S1AjR50pxrD0u23XH8EgoBJCDJjIkiJhSFUXIMlT9ajzNkeMyFEsHIZQ69xTDicqFYK1hb2+z3ae/jW8ODVqtfKpTohT/6w35p5kQJBNrZ20On28XE5CQZ52g4GKLdbiFJEpTLZdQma5icmITWGqVyRhHt96tIkhTWWoRhQCdOnMC5CxfopZdfxtVXXpErV6/6s2fOGEnS4dLNW4Otn7w32nr7hp9td0sn61OTl2ZmK0Wl0U8THKaJZB8ikSY+0k/0yApjh2wDvKgYFQ2kGhCVpWYsBBZ6UlE0lxljEIqsoBVQdoqUY+rDY+jMqOfsRs/ZTxomvnvP95rHuB0CAIuzs4KvGaCPzkou8sh5ZF1cXOxXz5xZDin5ySmO9L4zQc3oK1OayiJASRQE7BMvyvqsoNGKQGqscJ9F6tweJGuPCRDmI/MxX02OdZKP8+hyOSswM5WCAIoYsVDlQuomdnf2KL35CfZq1e7EpYsSlUrBy6++Wj599mzBDEdkjUGv28XubgP7BwcyHI0gAIVhiFKxiLm5OVy8eBHWWknTFOKFKpUyZmdn6eyF8zh34aI/e/ZsPDFZj0e9ftx59Cg9WPjY8N0H4cl2JzzjuVJXQbEahtkU0hr4XDNcWD2xsz5m0Qkyw59IOLOS8ALjAKGMIxNozt0BMiBbDzjnhYQQgaQkjNARUuOxlRh3aG37wJqlnnMfrZv4Abrd1muAf/Pq1eD6K6+4+fl5N/8pJdqvDaDfAgQLC/7Tjbye91uN0cgMw4q0nKkGhqd65MuTFEALSRF5vzQQqIAQaEaQG9cICbzPXJicSObpYgCdAioVkDlWjR5DMh2behAA5z2GxsB7wUQURVf01MyU9xM7D1c69zudnn75yuDstW8XL7/0Dcx897uB8qJNmmLQ7aB5cICdRkPWNjawtb2FeBRTtVrFmTNnMDszI4VCQcQ5ISKeqE3Q1FQd5clJFKvVlAT99k6jt3n7k2T/gwXIg5VwttWqzU7NlOs6CItBqHpJkpOEPPKoTE9QPY+Rl8ekrsASQiIYyfwFJRSogKEV5Z2X3FbPixiQiAVVHKFkGS72aPVT1xiZdjNJHrVSs3jg7J3d2K+hg/gGYK8Ui8GnWZZfxwg9th2Ew9/x2//+bX79xg3HOzvDNWAV1enSnrHnUuD8SKTGTJOTpDkAa/IAOwCOJIqYQuGfBrQRuDTzRwxSQWDz38Hj/utntBWzXFoEqbWeiVAuFvlEEFSKcYLkoEnbu43hQbMZV6ICrpy/ODg9Mxew4ii/lvySt7y/f8AP7t/H3Xv30TrYR6VSwYuXLuHSpUs0OTlJY/N4pbUH4I1zxsTJ4HCnMdj88OPk/tv/4oa3PgnmhknxRK1eOz89XSqFEYZpioFJvROBYmb+HNvEKn9CFUDwnjIGoicozwgChpbHgHZe4KxADBBZYpUIHw4dDgamsxXHS+tJ8nHTuLsjN9qkZrP/M6q+r2/KMX7zbwI4nVfH46vRMoPtNvy9k4jOKebStAkuQ2GqDIXIEwoGngxJYFjpPMoAlPF/fWb86A0gRqBsNtbVOYv9uA4R/aLRZr4HWFGMWR2UvsFqcr/TC9Sdu3Z7aioOnDuoXzwX1GdmIgaKinV5bnaOS4UiKuUK9ptNBIH2J0+epAvnzlNUKh3/U3G73+vu7TRGnfUN07u/jN7txaD8aL0wOUgKc0qVJrQuFHI+iMdjotNxqVD5OYhSQojyfmboAZORyKEMQWmGUvmQSvK8GuStESUW6McW3aFNuqlda6X2vXUzemcrwf1H2+sdzp+qIoL7lYq8/iWnGsdbwU/D4TcAfivrHI1P9MILL1z+vULp37xaqPzRS1HpD0+HwTdOBGGxoEi8iFeKBIAG06eMhcaUzTyl8GPZMPrcGiDjx7fKx9YQgVjnrUi6b43ZEd/bn6j25NKF+MS135NvfPf3CxeufKNWiYpTAAoAMBoMMBwO4Z1zURRSrT71xJ8/HHT3H95e3F3+4KNh/+49HW02KtVOr1IdDUuTnqJqoHWktVZaQ4iyCDqW06XPt6/9RE5NjzsZxI9dx5yIEAEhCAGxS5znjnO8Ho+wmaQbuya9sZGM/s8Pm50fL9eCJtbW4rHaQ+7JwvgNiMY8reSkz3VjzSKzxH0D4Ndfe03dWFtDt1j0VQ7tTKC5SFRXoOlIqDTBmqtQXHIM7YjEAGwg2oKUBQILaE8IPCEQhqac2/sr3OlWRFLvRYhQKRR4qlzWASiKux3a29wyjY3NdNjvemsMpWmK1FnxRGCtfbFQQKlc5nKlwlGxKB7wwzi2w8EgbR/sx49u3e4++tG7g+a7H/j09t0wauxVp1JTmwujyslaLaiWSuyJMHJWUm8zu0SiT0uH/ML3MfYYDMAIhRBKViQrCygLsMmuWdEpCjyxtULtxNrtND5opGbxwJgf9Z3/0Q+2Hy1Rp2MFoFPXrgX/8Y//mOYzb0F5SnD0paccR4FkPHB5G6/xlWy7wWJ39/D+SVp5QasCO3869X5OCaIa6/pEoDkiwkicj73POhpE+ebUZ7QwiD7lePjLg9sjG6lHRKjroHAhKtQq1mm3tWvShY/86l7Trp893Zp88cX2uZdfKl66cKmugTHBiSwwbDSbva3l5VFnddX37z108mitdHKvGQSpiYrg8iRzVM42d45WpfLWzK8kDnrcfGm8FT+OpZL/rD15JogwdOoFXevQdbbVtW5x36TvrSejm4c+3R7/KgPy3/F0nqcG0ONvZjEr7Uuzgsx6Oe03Gq2PT2K1XSrdHsKdKDCHU0GoKqImJ5QmT8glRAHPY8l6+Smb21+1WlGZ1jERgNgYJNaAvKBaqahLxWJtzrlyn2Damzvx/najs3e70AtefDEedXvlogrCF154oZRfZ9M5OBhu3L97eOdffzho3/oEk3ut6gVSU7NaVyu1CSbvWRETs8LQGoi1QD7O5mNplfwKF9Yf636MW5eSL80yC4QzGa+ud9j16WBfzGrHuw+aSfLu++3ew93OY6FyAbC9sOC2FxaeA/qX+CAImTOZW2009tunTjwMmcsnXBDUjS6DUT7DCMrMFGaOsc5CYHMFdToe0OSLFQzjBDH1Xrx40ayoVChQTSk1YZ1qx6PQDQZ6MOibyNmhOey5FhMeMSXJQatXqFYC8pI2lleSg4UFP/rgY1YPHnLFSbE2Wa/O1KeiShDCQWC8h/UeqXcikg1+mAhfdN3jCQJNlktnW+AEYrCyEPQzMI+2Xbq9Z9P7e87eXEn7d3c7u3sA3L+89ppu3rgh3wP8m4C8+fTUYE8voOcx7zGf91bffHPcYh12+vure6WJeNsYUojrPecmU+/mXiyUuMYsTCwDZ33qJfd6ofGs5deW4FEOrsxz3MPlposhEaqsAh0WKiespVFsUrO06vZSK/v3ltpSCEhbT77ZBG1uly8e9qpRVAgmoUu1IAyUCLzPV7dyx9uxkAB9gafLzw7ZmWuCInK5ihM7EfStjfeN2d5IR4sbJr25lph77zu3BWAAAM0bs+pOzi+gz3KofQqOenoDNGin39d/vrODG4AgRfzNV19t9Q570rOmasRVIuZiGVwsK600ETkB+dyYTB0L0F/EpuyJSJ0VZEQAee+ROg/jPTwgQRBQrVgMZ0rl0kQQVimOw/burtl8uDxs3Lsf9+4/RLDVKEz3htWzWk+eLVdrU+VyFAYBWyKk3sP5zMlVHt88RES/rpvx2BOHwCDhbJmXjQi1bOr3nW2sp/Gdh/Ho/U9Gg48+7HWWTLPZHv/eVSzSYl68P61P9qcV0ABA106f5hs7O7wNyH8BZHVtza8WT6TajliT+AKrQkhcY6JyxIqizIUJGiQeoFz88dcGiicKRBFxEEgmjSuB1lwJI5qIIiowc5LEQdzt2uSgZdFuc7E7iOrWl2e0Ls8Vi4XpUpmjMCRLBJPJrstjP1L6jTzLc3VXUUQImYmJlIVQy6Zu26aHWyZ5sJ6m7z2IB++90+8vmf39JgFG3niDceMGAUAToKcZ0PopBrScWlhwf55X1UdaPtv3Dm9PTd2s8URvQiWeQKWBdyUHqZwPC6gwiwfEOCdWssVZ/gztoy/y6MiJfWMto8zZ2ntYWIwcwXmPIiuaiwrlMrGy3noFVqUwjCpaBySCxJpsUOI9KFcQpS+o1/yLMw0RyRa6afxE6zuHbWM6S2m8spbGt9aN+eBRHH+CZvMAwEgAzC8u0jHbNXmKMfNUA/rIu04A+utr39H/Y2HBEuDQam0dKtV5xFxOISdSkXLEfKnIqhIEIWsizt2YPP0GPoUjtdPxsFEy7sjI+yObJxUEOBEEhROgwuNEOEsnUgCpMU/k5r/JR+XxIVEWG0gZEXSdw4E1nW2bri6l8ceLw+FHa97cW9ve3jx2yejOYwncpxrMTz2gj5/1Tof/Gtc0Y8EIgMVmsz95Ut015CY0IJNKxSWlvglgYi4IETEjAGWDEWSAy7sF9EWqmZ9l6J5xYGXsBAVNzBEphJT5cwtlVs6pd0i9Eyc+H5SA1M9IMX4d6PEZh1AUERWYQdksBS1rsZXGg22Trq8n8UfrafKjm/Hgk06SbCNr58FC6Hu/ZoOF54DOQXR9acme+pQ5xGGj0dg9ffpHNQ56E2mSekY08u6bHlI4oyMUmYSFJHGWM4lvMH8x67LP/IexSlJGFpLMisd7JN4jhc2KsLxH7klAIsSgzLcA9IRq6qfbjPIFI3P+UHAkwmMlpG4G5uGDeLSxFo9ubaTJD/d88uNOHDfQbvcAwIonEGEe8HLjBt56BqLzsxShJd9PkzcAxtWrGouL9vtAsry9vVG+dCkJTVqKCZPWi46Iz2tQZS4IlAIjIAYhN7LOe7K/qp/18U+VPtXfZskWTkEgC0HMApvjVYGgcw1ROqrGH98R8qlo/0X65sfvt2yTBYpAnAIYOSu7Jh1uJenGchzfvBv333/UH91c3t9ZonwX8O8A/t4rr+irGbfG0zMUoRWesfM6QK2JCfX7f/qn8r8zHgF22+0BFSKrmJIA5BUoskI1JoqKrFBgRsgsDBIrHi7XQ6fPCYrjo+9c/iPz687ZPpyDWY219EAQDZgSJC0Btgh4ne/xCkgJQXmCknzpQLJtdS8ylsh4zG/Oex6fl4g0rv0AeM67GRExCwFta7GexMO1NFlfS+KFZZO880nv8MPV/d0lAMOxZvTM9etc7/epurMjN56RyPzMAvoGIH/Wasl/Xlx8InLsT/SGRVNslgjDkUgYe6kn4idLzKqSgdp7ESQi5DOzm8+VftCTbS84kaP1plw2JhdXyrjG7LOpg4sISRWUVkGmBHIqd6LwGaVT5V+zfCDTmzMk2aLqOAXJJ4Sfl1l3HNs5sY7CTHqMet5jPY1HD+LRxt148PFqktxopPTO7bi7jExn8Ai4dxYX8cHOjv8TPHtHPYOvGTfyr9cBdf3q9eDt5nX5fveG2e31uopoYLXSKShSRGHIVAiIw4BZcWZBTUw09vP7hY4m9BkphwGQskeqM908rwFSfORqa5RgFAn6ZUG/kKl8xuLhXLaYyj6L6GCCVUASeIy0YKQzSTNCDnpQZiTw+QEtDJDOblVFzOwJ1HHW79q0v5bEa/eGo5t3RsN3b4667905NX2HVlaGDJK/wHX1wuU0/HarhUXAv4Vn8+hn9HULkPsi4g6A+aNi8ZNWq9ENww89sylbbhdS/gMjuDoUXz8bRigzgwSSivcjGVP986Hczwl3kAxkWfGX6VsMIw+vMmGXkidYSxAnSEjQVx595zFIPWL2gAMKBNSUgtMKlhiaAKcEo8BjwAIDgXaEWgwUU4b2+ZidHr+Gn+5iYLws7JnJs4ACIkXMiL3HvjFomLS7bZLVjTj5aDtJfnw/Ht3a2m2soNEwx2/ocxMT/tQzlmJ8VQD9uE+9uGjfesKgAnat0VipXpxtVUxwmAh8x/lo6NxLTKidCyIUmXOngM9wvPoZheCR6LoQDHs4DfRDjzQQRBBYBxjFGaAh6JNDL82EbkaUzSSKnkGULfaCMn/5VGf61iOdtSNKhiGWoSxBe3pi4ZV+7nMky1/yrRZx4qltjeyYtPMwGT5aSZKPNuL4na0k/vEW7AaQt8wJcOKJQA4Lv3nLiOeA/pxzg7/Cd/Qr/2GG/+af/zllouT2o73dvdnyzctlW429DwMgLqd0mUSmZ4IwKLJCNDbDzHLjY6PnzwYPH8vTSLIiLhEPO+6e5GA98khxAusz2wdwtnEyBqYjICaPHjIRRSdABIYSgj5m6iM/p/ijTHgfTESKiDyg09x4tO9t0jKmtW3itUfp6Nbt4fCDO/3Rx/v720sALAP4x+vX1f/86CP9+tLrDllHQ551MHxlAN3GJd8a7QlnViIAgL29R81k9txHYUn6JaZ9hvR71r561hVPXygUMakCaAKsePEiDhDxAjVmn/JnBMLxKlfoCGFCSCxg2GOYa4RUkAnbgDOprVhlBSQRUBBGESqT+hVC3zl0xSIWjwIpFIVQcYTI5qkG/wwRk+zmkwAQzSwqe73KCzD0DvsmTfZturNrk4dbJr25EScfLA+Txf397U0cW3O7Mz8vzWvX/CXMCr4iR39F3ofMY96Nq0XJ7JUFgO/sra+35uZ2nUKvZVMcmEgnIqKJZzmioMrqiOGGo8EIfaY/stDYgAcoWEbNMkQJusojVR4xZfl0AYKQCZFiFEKV/Y4HilCIPGe2GhAk3mHoHJwRVEWh5hk1USjkuboj+anXQHlEzjcmWQBvRciKoOc8mjYdNWy6sx7Hd1eS/kebafz+ehzf3NxrbB8H899mO5weCwt+AQtfFTw/m12OX9hpm58PXp2Y0O+2Dvz38ZbfHQzMuvd9A4ZibwusfYEUmERZj1AytimHTBwSU/a4JzmWUNKn+Btgl3mWkACGgTTIHAXG2LMMJIHABtlYg4mgwdD5fxCLoAuHoXgoT6hbhRmnUXUKgRA8PR4AHW1M5eP7kIjy/jJZIu45R/smdfvOtA+sebRr09tLyfCD26P+hzcODxf3Dw5WgUx5+H9dv64ABO82m7TzFUgxvhaAnn39dfzZO+/438Fbj3UckyRuR2Y0wdFepFTXEdzQeR2LL1rxpYgVFZmhwWCIA8E7Efhs5kFjDYwxs10LIQABTDARMIoEsRaMyGMEjxF7JEFmNScqi/vCucIneQzIYwAPT0ARjCnRqItGSRicOxT4Tw1KROADIikwS0gZMaPrHTbSBJsmae0Yc2/XJu/vGPOT5aT/wbvd7j3b6ezmnUYAwD8tLmK+2cQfZyr7zwH9LJzFxcXx0i299tprejWzRQBGtrfd7TaKhah36LwfeEdC0BpQTAgABJzpX1DuT8nIC66jSJmnHAEImrKJYBwK+lFmYDSgrE2XaoELAQoIFBCgMnH2JHegGpGHIYEGoQKFSdGoiUKYPwXGwsrjFCN/CczZy2Ergo6z2EwTs5nGe5smefAoHX3wKE1+spoOP1w4PHzQb7ebx8BMbwD8J5lvoHwVwfxVyqF/ZrQubm2pN69eZVm8YzgzgfLvbm2tlufmot8tigkUH5aZd8nwxZH4FyreTdWVDuo6QClXr3EiSL0XKyJZGpBtrfCRGX02QTQkSIOcg60IOiKQZqggYzo78XAOMCZvMwgQuiyqqMfNN4iI+NzDlQEERKyJaTx6H4hHMzXoONPetWZjz9oHu87eWTXxJ3fS9O56f30TbXSRF7Du+l+o319Z4bezxeOnSnbgOaB/yWKx+u1vW8zPSw7m8TGD3d215Tn06wFvFxWtdsS+UjL8u3UdfnNW61OKiIs6REAEB+8skeTr5SQi7HP4iWQqTWIExIIwJEQBIwoUopARqky5XxFghaDYQwmQOgchQEuWj5N7nCg7ZHYxRBBF425gptAwyFOM1Tg+3Hfpw7a173fEvdeFu7s5Gu2sp1uHaGH4ePAi4Pl5l6sWCH2Fwfy1APT8/LwbR+vrAF/NtpWFgd7u7m7v/1p78N16/WAW6EzrQnxGXOJRGBST5BSJlKtKsyJSNrsdRIMoV6TN7ZEzQBZSQo0UhAkFxQiFoVzWV2ZHUGNTGc+wFjAW8FagraAwlikbpxfZU0B5CNL8Nkycxch733Im2UqT1ko6Wl1L45tbxvxwj+W9O2fOrOHOHXtkk/54XkTjeQvhq380vj6H6gDvXLsGfPCBPfq4Dw567x0crFVPn7DfLtSGikuNiNTLAfCt2LtLVa1n6zoIyqxQUIrCPMUQL97kmr2RI9RFUTlbFqPAElTCGPs6Z3kw8mEewTvAGYKkIsoABUcSCYkCoIg4yHkmKQRdZ3FoDQbOpkPvD7rithomWW6YdHE5HS0ujEZ30kZjDUtLx2TURHD9e3x9HpjHPPCUbmj/ZlpcXyNA/3Rn+egwgOjyyZOVb01P1yY9XZmA+sO61n8wrfU3TwXhyVNhIZzRAcqsQBCQwBrxiCFIxJMlkGeQsOTdbMCB4PM/RZkOe4bxx5p7wp5ECyQA+4AIJVIUECmBUN97bKUJNtNkdGDNbtuZpa63d7rWfLRp/Ce3zHBnfX29AzxOMY6/UQGemdWp512OLwjuNwCexXX+b6/NqL9f+y6+j8W01e8P7zWb7Y8D1Ubq/UgceYgPiE3AbLwIGS/KiLAjYSZmRcRFUlwhRWVhKlpGYCjXpM60qcffBykQGULRMsqOURFFZWgqkuJs0kfKQTgWT11r3b41cdOaxo5JVtbS+O5SPLp5b9T/8P5wsPDO+upip9PpEmAEQri6GF5pFvnPsYO38fU9Gl/ncx3AHgD8k5fjAXxnZ//dev3O76hqGilan9TBxcCY8x1nLkTg8wXFc7M6DGbDEFXKjCzhBal4seI9eQgJg8STIGsDZmr6kGx5F4iYEEKxpkyQL4ag5xzazuLApug72+47t95z7u6BMw+bqVldTYdbtweDzf7BwTry7ZIsKrH8LV4GrhXxFRr6PU85fs0nrNfrxWv1+sTZYvFsBcFlFnyrxvRqXeuXzgbRqQuFUmFOh1l7TwDrnXOAWJfxNyTfFvTHPEgYLExAwIQg7y2DQH3v0bQGO2lsG9bs79l05dCamwPxPzbi7uyNRtv3VHe0stJOAcSfkUZ87dKL54D++SkILV69qk+mKf3Xv/xLo956yx/pwc3MVC4Xi+emoV5+IYy+dT6KrlwqFC+dCwqnplVQKzAVQRRqAhVYIWKGyr1Lnlh4pcdrVlY8LDJmnBGfDp0Mus4dNkzSWk1H249M8nAtGd28G3ffbzVaD5HzMBiAE6G3X39d/UO/T/cXFuRGFq3l+cf4dU85PnWuzsz4nX6fngAzADSb/SVgfWlmJvVemmWt7leN+QZAV/ZsekETnQ2IZye1Lp6NiphQCiVWUMgAPA7DkncuBs5hYD32rcGeSUzs3e7IYy32stzxZr1hzfq2TTduxt2NfqO1gWOkIp91QuQfr1+X+wBmnwP5eYT+AtdJnz17Vv9uvV6YsvYCI/hW4N3VkuKXpoLwwqwOz5wNopmLhSLXlUaQC/vmZCc4ACPv0LIWO2mCPZd0dk261XZ2qW3ldg/2duyC5cZouNVNDjuLzWaKbHT9HLTPAf3Frst1gK++9hqd7vfpGoDvLmQCN8eQFU6fPn1xmujSmSi6cCksnT4XFU9fCMLTc0F4YoJVRRNHBNHZHIbEASYRG/eddPetOWyY+GDNJNubSbz2MDEP78T9Fezvb43bcOPIbq//hfqbjz7SycSEby8s+Hk8+5slzwH9JVyXNzJhQvqnnGv/GQiqYKo4cTmcmHi5Mll7qVipn1B0pkZ8tqz0XACaJqIKkQQO4rygZ8k3jcNOR3yjmcbN9WR4sDwatT8YjVo4OGgDR2piRy/EA/S969cZ8/PH/bOfA/o5oH/tZ7zUQq8BVPzud0v+8PB01eJiPQjPTyp9pqBVnUiKqXiTwh8MvV/vizzqEK3vWttsW9tfyqZ8z6Puc0D/dlOQ+rVrHHU6PGXO0JsXYMMbN+xnLOJphOHlc/UTl18qlS/MhOEJJimPYNOOs7vrLll+mCQPsLX/6HixN1ZXMq/9e/33WNUfbgXyPMX45c//Bx7ztB6lIX+6AAAAAElFTkSuQmCC">
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
     {label:"Trainers",v:"trainers",term:"trainers",none:true},
     {label:"I don&rsquo;t have a word for this",v:"none",term:"no word for this",grid:"none_shoe",excl:true,none:true}
   ]},
  {id:"singerfinger",text:"Do the words <i>singer</i> and <i>finger</i> rhyme for you?",
   tag:"real data",real:true,metric:"pct",grid:"singerfinger",
   info:"singerfinger",infoLabel:"velar nasal plus",
   opts:[{label:"Yes, they rhyme",v:1,word:"rhyme singer/finger"},
         {label:"No, they sound different",v:0,word:"keep them distinct"}]},
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
let _rsz=null;
window.addEventListener("resize",()=>{ clearTimeout(_rsz);
  _rsz=setTimeout(()=>{
    const onIntro=document.getElementById("intro").style.display!=="none";
    if(onIntro) drawMini(); else render();
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
  version:"2026-08-07"      // bump when the question set changes, so responses stay comparable
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
