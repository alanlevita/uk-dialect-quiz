# Real foot-strut percentages ("% who RHYME foot/strut", i.e. NO split),
# taken directly from the prose of MacKenzie, Bailey & Turton (2022),
# "Towards an updated dialect atlas of British English", J. Linguistic
# Geography 10(1):46-66. These are the authoritative published figures —
# used instead of my color-decoded Gi* estimates, which had a registration
# error in Scotland. Where the paper gives a county/city number we use it;
# otherwise we use the stated regional average. Regional groupings and the
# two Welsh / a few Midland splits are our reading of the paper's text —
# easy to adjust here in one place.

_SCOTLAND = 3       # "97% ... foot-strut split"
_SOUTH = 5          # "just 5% in the south"
_NORTH = 79         # NW / NE / Yorkshire "79%"
_NWALES = 45        # Wales 22% overall, but N. Wales "resisted the split"
_SWALES = 12        # ...so S./mid Wales splits more (lower rhyme)

COUNTY_PCT = {}

def _put(pct, names):
    for n in names:
        COUNTY_PCT[n] = pct

_put(_SCOTLAND, [
    "Aberdeenshire", "Angus", "Argyllshire", "Ayrshire", "Banffshire",
    "Berwickshire", "Buteshire", "Caithness", "Clackmannanshire",
    "Cromartyshire", "Dumfriesshire", "Dunbartonshire", "East Lothian",
    "Fife", "Inverness-shire", "Kincardineshire", "Kinross-shire",
    "Kirkcudbrightshire", "Lanarkshire", "Midlothian", "Morayshire",
    "Nairnshire", "Peeblesshire", "Perthshire", "Renfrewshire", "Ross-shire",
    "Roxburghshire", "Selkirkshire", "Shetland", "Stirlingshire",
    "Sutherland", "West Lothian", "Wigtownshire",
])
_put(_NWALES, ["Anglesey", "Caernarfonshire", "Denbighshire", "Flintshire",
               "Merionethshire", "Montgomeryshire"])
_put(_SWALES, ["Brecknockshire", "Cardiganshire", "Carmarthenshire",
               "Glamorgan", "Monmouthshire", "Pembrokeshire", "Radnorshire"])
_put(_NORTH, ["Cheshire", "Cumberland", "Durham", "Lancashire",
              "Northumberland", "Westmorland", "Yorkshire"])
# East Midlands — paper gives several county-level numbers directly
COUNTY_PCT.update({
    "Derbyshire": 79, "Nottinghamshire": 76, "Leicestershire": 43,
    "Northamptonshire": 7, "Rutland": 63, "Lincolnshire": 63,
})
# West Midlands — paper gives Warwickshire/Worcestershire, notes Herefordshire
# + S. Shropshire as strongly split; region average 47%
COUNTY_PCT.update({
    "Warwickshire": 24, "Worcestershire": 31, "Herefordshire": 15,
    "Shropshire": 47, "Staffordshire": 55,
})
_put(_SOUTH, [
    "Bedfordshire", "Berkshire", "Buckinghamshire", "Cambridgeshire",
    "Cornwall", "Devon", "Dorset", "Essex", "Gloucestershire", "Hampshire",
    "Hertfordshire", "Huntingdonshire", "Kent", "Middlesex", "Norfolk",
    "Oxfordshire", "Somerset", "Suffolk", "Surrey", "Sussex", "Wiltshire",
])

# City -> real % (city-specific where the paper states it, else the region).
CITY_PCT = {
    "London": 5, "Bristol": 5,                 # South
    "Manchester": 79, "Liverpool": 79,         # North West
    "Leeds": 79, "Sheffield": 79, "York": 79,  # Yorkshire
    "Newcastle": 79,                           # North East
    "Birmingham": 47,                          # West Midlands (region)
    "Nottingham": 76,                          # Nottinghamshire
    "Edinburgh": 3, "Glasgow": 3, "Aberdeen": 3,  # Scotland
    "Cardiff": 22,                             # Wales overall (mostly N. Wales)
}


def county_pct(name):
    return COUNTY_PCT.get(name)


if __name__ == "__main__":
    import json
    with open("britain_pixel_data.json") as f:
        names = json.load(f)["county_names"]
    with open("britain_pixel_data.json") as f:
        cg = json.load(f)["county_grid"]
    used = sorted({cg[r][c] for r in range(len(cg)) for c in range(len(cg[0]))
                   if cg[r][c] >= 0})
    missing = [names[i] for i in used if names[i] not in COUNTY_PCT]
    print("counties on map:", len(used), " classified:", len(COUNTY_PCT))
    print("MISSING (would need a value):", missing or "none")
