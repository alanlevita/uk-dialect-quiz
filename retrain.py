#!/usr/bin/env python3
"""
Update the quiz's answer maps from collected responses.

The published-research maps are treated as a PRIOR rather than replaced. Each
location's rate is pulled toward what respondents from there actually said, in
proportion to how many of them there are:

    posterior = (observed_yes + ALPHA * prior_rate) / (observed_n + ALPHA)

ALPHA is a pseudo-count: "how many real responses it takes to start overriding
the research". With no data the maps are unchanged; with a lot they become
empirical. There is no threshold to pick and no cliff edge.

Usage
  python3 retrain.py --responses responses.csv          # write empirical_maps.json
  python3 retrain.py --simulate                         # Monte Carlo validation
"""
import argparse, csv, json, math, os, random, re, sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA_DEFAULT = 25.0
SMOOTH_SIGMA = 1.0

# Modern council/ceremonial areas (what the hometown picker stores) -> the
# historic counties the map grid is built from. Several are genuinely one-to-many
# (Powys, Cumbria, Scottish Borders); we take the dominant/most central county,
# which is accurate enough to place a respondent for county-level aggregation.
MODERN_TO_HISTORIC = {
    # Scotland
    "Aberdeen City": "Aberdeenshire", "Argyll and Bute": "Argyllshire",
    "City of Edinburgh": "Midlothian", "Dundee City": "Angus",
    "Dumfries and Galloway": "Dumfriesshire", "East Ayrshire": "Ayrshire",
    "North Ayrshire": "Ayrshire", "South Ayrshire": "Ayrshire",
    "East Dunbartonshire": "Dunbartonshire", "West Dunbartonshire": "Dunbartonshire",
    "Falkirk": "Stirlingshire", "Glasgow City": "Lanarkshire",
    "North Lanarkshire": "Lanarkshire", "South Lanarkshire": "Lanarkshire",
    "Highland": "Inverness-shire", "Inverclyde": "Renfrewshire",
    "Moray": "Morayshire", "Perth and Kinross": "Perthshire",
    "Scottish Borders": "Roxburghshire", "Stirling": "Stirlingshire",
    # Wales
    "Blaenau Gwent": "Monmouthshire", "Bridgend": "Glamorgan",
    "Caerphilly": "Glamorgan", "Cardiff": "Glamorgan",
    "Ceredigion": "Cardiganshire", "Conwy": "Caernarfonshire",
    "Gwent": "Monmouthshire", "Gwynedd": "Caernarfonshire",
    "Merthyr Tydfil": "Glamorgan", "Neath Port Talbot": "Glamorgan",
    "Powys": "Brecknockshire", "Rhondda Cynon Taf": "Glamorgan",
    "Swansea": "Glamorgan", "Torfaen": "Monmouthshire",
    "Vale of Glamorgan": "Glamorgan", "Wrexham": "Denbighshire",
    # England
    "Bristol": "Gloucestershire", "County Durham": "Durham",
    "Cumbria": "Cumberland", "East Riding of Yorkshire": "Yorkshire",
    "North Yorkshire": "Yorkshire", "South Yorkshire": "Yorkshire",
    "West Yorkshire": "Yorkshire", "East Sussex": "Sussex", "West Sussex": "Sussex",
    "Greater London": "Middlesex", "Greater Manchester": "Lancashire",
    "Merseyside": "Lancashire", "Isle of Wight": "Hampshire",
    "Tyne and Wear": "Northumberland", "West Midlands": "Warwickshire",
}


def load_geography():
    d = json.load(open(os.path.join(HERE, "britain_pixel_data.json")))
    cg = np.array(d["county_grid"])
    land = cg >= 0
    filled = ndimage.binary_fill_holes(land) & ~land
    if filled.any():
        idx = ndimage.distance_transform_edt(~land, return_distances=False, return_indices=True)
        cg = cg.copy(); cg[filled] = cg[tuple(idx)][filled]; land = cg >= 0
    return cg, land, d["county_names"]


def load_prior_grids():
    """The surfaces the live quiz is using, read straight out of the built page."""
    src = open(os.path.join(HERE, "index.html")).read()
    raw = re.search(r"const GRIDS_JSON=(.*?); let grids=null;", src, re.S).group(1)
    return json.loads(json.loads(raw))


def county_of(hometown, county_names):
    """'Leeds, West Yorkshire' -> index of the historic county, or None."""
    if not hometown or hometown.strip().lower() in ("", "notgb"):
        return None
    part = hometown.split(",")[-1].strip()
    hist = MODERN_TO_HISTORIC.get(part, part)
    return county_names.index(hist) if hist in county_names else None


# ---------------------------------------------------------------- core update
def update_surface(prior, counts_yes, counts_n, cg, land, alpha=ALPHA_DEFAULT,
                   sigma=SMOOTH_SIGMA):
    """Shrink each county's prior rate toward its observed rate."""
    out = np.array(prior, dtype=float)
    for ci, n in counts_n.items():
        if not n:
            continue
        mask = (cg == ci) & land
        if not mask.any():
            continue
        prior_rate = float(np.nanmean(out[mask]))
        post = (counts_yes.get(ci, 0) + alpha * prior_rate) / (n + alpha)
        # move the whole county by the same delta, preserving internal texture
        out[mask] = np.clip(out[mask] + (post - prior_rate), 0.0, 1.0)
    if sigma:
        v = np.where(land, np.nan_to_num(out), 0.0)
        num = ndimage.gaussian_filter(v, sigma)
        den = ndimage.gaussian_filter(land.astype(float), sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(land & (den > 1e-6), num / den, out)
    return out


def tally(responses, key, positive):
    """Count yes/total per county for one surface.

    Only "1"/"0" rows count toward the denominator. Question 6 now has a third
    answer ("neither"), which is neither a yes nor a no for this surface -- adding
    it to `tot` alone would count it as a no and bias the map downward.
    """
    yes, tot = {}, {}
    for ci, answers in responses:
        if key not in answers:
            continue
        v = str(answers[key]).strip()
        if v not in ("0", "1"):
            continue
        tot[ci] = tot.get(ci, 0) + 1
        if positive(v):
            yes[ci] = yes.get(ci, 0) + 1
    return yes, tot


# ------------------------------------------------------------------ real data
# Only questions whose stored answer is literally "1" or "0" belong here, because
# the tally below counts a row as positive iff the value == "1".
#
# "tvd" used to be a yes/no and is NOT any more -- it stores "tea"/"dinner"/"supper".
# Left in this list it tallied zero positives out of every response and dragged the
# surface toward 0 everywhere. Removed.
#
# The other 14 questions are lexical multi-selects (their answers are word lists,
# scored against one surface per word) or 1-5 sliders. Retraining those needs a
# different update per answer shape, which is not written yet -- so they keep the
# published maps, and that limit is worth stating rather than glossing.
BINARY_QUESTIONS = ["footstrut", "singerfinger", "youse", "thfronting", "scone",
                    "trapbath", "bookspook", "stirstare", "northforce",
                    "forcecure"]


def run_real(csv_path, alpha):
    cg, land, county_names = load_geography()
    priors = load_prior_grids()
    rows = list(csv.DictReader(open(csv_path)))
    kept = []
    for r in rows:
        if not (r.get("result") or "").strip():
            continue                                   # no final result -> skip
        ci = county_of(r.get("hometown", ""), county_names)
        if ci is None:
            continue                                   # no usable hometown -> skip
        kept.append((ci, r))
    print(f"{len(rows)} rows -> {len(kept)} usable (hometown + result)")
    if not kept:
        print("nothing to train on"); return

    out = {}
    for q in BINARY_QUESTIONS:
        if q not in priors:
            continue
        yes, tot = tally([(ci, r) for ci, r in kept], q, lambda v: str(v).strip() == "1")
        if not tot:
            continue
        prior = np.array([[np.nan if v is None else v for v in row] for row in priors[q]])
        new = update_surface(prior, yes, tot, cg, land, alpha)
        out[q] = [[None if not land[r][c] else round(float(new[r][c]), 4)
                   for c in range(land.shape[1])] for r in range(land.shape[0])]
        moved = np.nanmean(np.abs(new[land] - prior[land]))
        print(f"  {q:14s} n={sum(tot.values()):5d} counties={len(tot):3d} mean shift={moved:.4f}")
    json.dump(out, open(os.path.join(HERE, "empirical_maps.json"), "w"))
    print(f"wrote empirical_maps.json ({len(out)} surfaces)")


# --------------------------------------------------------------- monte carlo
def make_truth(prior, land, rng, strength=0.35):
    """A plausible 'reality' that differs from the published map: a smooth
    spatial distortion, so the prior is imperfect but not nonsense."""
    H, W = land.shape
    noise = rng.normal(0, 1, (H, W))
    noise = ndimage.gaussian_filter(noise, 12)
    noise /= (np.abs(noise).max() + 1e-9)
    return np.clip(prior + strength * noise, 0.02, 0.98)


def simulate(n_respondents, alpha, cg, land, county_names, prior, truth, rng,
             counties=None):
    if counties is None:
        counties = [c for c in np.unique(cg[land]) if c >= 0]
    # county-level truth rate a respondent from there answers "yes" with
    rate = {}
    for ci in counties:
        m = (cg == ci) & land
        rate[ci] = float(np.nanmean(truth[m])) if m.any() else 0.5
    yes, tot = {}, {}
    for _ in range(n_respondents):
        ci = counties[rng.integers(len(counties))]
        tot[ci] = tot.get(ci, 0) + 1
        if rng.random() < rate[ci]:
            yes[ci] = yes.get(ci, 0) + 1
    return update_surface(prior, yes, tot, cg, land, alpha)


def county_mae(a, b, cg, land, counties):
    """Error measured where it matters: the per-county rate the quiz reads."""
    errs = []
    for ci in counties:
        m = (cg == ci) & land
        if m.any():
            errs.append(abs(float(np.nanmean(a[m])) - float(np.nanmean(b[m]))))
    return float(np.mean(errs))


def run_simulation():
    cg, land, county_names = load_geography()
    priors = load_prior_grids()
    counties = [int(c) for c in np.unique(cg[land]) if c >= 0]
    rng = np.random.default_rng(12345)

    print(f"Monte Carlo: {len(counties)} counties, "
          f"{len(BINARY_QUESTIONS)} binary questions\n")

    Ns = [0, 50, 200, 1000, 5000, 20000]
    alphas = [5, 25, 100]

    print("=== 1. Does it converge on the truth as responses accumulate? ===")
    print("    (mean per-county error vs the simulated reality; "
          "prior-only is the no-training baseline)\n")
    header = "    " + "N".rjust(7) + "".join(f"  alpha={a:<5d}" for a in alphas) + "   prior-only"
    print(header)
    results = {}
    for q in ["footstrut", "youse", "singerfinger"]:
        prior = np.array([[np.nan if v is None else v for v in row] for row in priors[q]])
        prior = np.where(land, np.nan_to_num(prior), np.nan)
        truth = make_truth(prior, land, np.random.default_rng(7))
        base = county_mae(prior, truth, cg, land, counties)
        print(f"\n  [{q}]  prior is off by {base:.4f} on average")
        print(header)
        for N in Ns:
            row = f"    {N:7d}"
            for a in alphas:
                errs = [county_mae(simulate(N, a, cg, land, county_names, prior, truth,
                                            np.random.default_rng(1000 + s), counties),
                                   truth, cg, land, counties) for s in range(3)]
                row += f"  {np.mean(errs):11.4f}"
                results[(q, N, a)] = np.mean(errs)
            row += f"   {base:9.4f}"
            print(row)

    print("\n=== 2. Does it degrade gracefully with almost no data? ===")
    q = "footstrut"
    prior = np.array([[np.nan if v is None else v for v in row] for row in priors[q]])
    prior = np.where(land, np.nan_to_num(prior), np.nan)
    truth = make_truth(prior, land, np.random.default_rng(7))
    for N in [0, 5, 20]:
        got = simulate(N, ALPHA_DEFAULT, cg, land, county_names, prior, truth,
                       np.random.default_rng(99), counties)
        drift = county_mae(got, prior, cg, land, counties)
        print(f"    N={N:5d}: moved {drift:.4f} away from the published map "
              f"({'safe' if drift < 0.02 else 'CHECK'})")

    print("\n=== 3. Realistic uptake: respondents cluster in cities, not evenly ===")
    d = json.load(open(os.path.join(HERE, "britain_pixel_data.json")))
    city_counties = set()
    for nm, col, row in d["cities"]:
        r, c = int(row), int(col)
        if 0 <= r < land.shape[0] and 0 <= c < land.shape[1] and land[r, c]:
            city_counties.add(int(cg[r, c]))
    weights = np.array([(float((cg == ci).sum()) * (6.0 if ci in city_counties else 1.0))
                        for ci in counties])
    weights /= weights.sum()

    def simulate_weighted(N, alpha, prior, truth, rng):
        rate = {ci: float(np.nanmean(truth[(cg == ci) & land])) for ci in counties}
        yes, tot = {}, {}
        picks = rng.choice(len(counties), size=N, p=weights) if N else []
        for k in picks:
            ci = counties[k]
            tot[ci] = tot.get(ci, 0) + 1
            if rng.random() < rate[ci]:
                yes[ci] = yes.get(ci, 0) + 1
        return update_surface(prior, yes, tot, cg, land, alpha), tot

    q = "footstrut"
    prior = np.array([[np.nan if v is None else v for v in row] for row in priors[q]])
    prior = np.where(land, np.nan_to_num(prior), np.nan)
    truth = make_truth(prior, land, np.random.default_rng(7))
    base = county_mae(prior, truth, cg, land, counties)
    big = sorted(counties, key=lambda ci: -weights[counties.index(ci)])[:10]
    small = sorted(counties, key=lambda ci: weights[counties.index(ci)])[:30]
    print(f"    prior error {base:.4f}; showing city counties vs the thinnest 30\n")
    print("          N   overall   city-counties   rural-counties   counties seen")
    for N in [200, 1000, 5000, 20000]:
        got, tot = simulate_weighted(N, ALPHA_DEFAULT, prior, truth,
                                     np.random.default_rng(2024))
        print(f"    {N:7d}  {county_mae(got, truth, cg, land, counties):8.4f}"
              f"  {county_mae(got, truth, cg, land, big):14.4f}"
              f"  {county_mae(got, truth, cg, land, small):15.4f}"
              f"   {len(tot):3d}/{len(counties)}")

    print("\n=== 4. Sanity: if the prior is already right, does training hurt? ===")
    truth_same = prior.copy()
    for N in [200, 5000]:
        got = simulate(N, ALPHA_DEFAULT, cg, land, county_names, prior, truth_same,
                       np.random.default_rng(5), counties)
        err = county_mae(got, truth_same, cg, land, counties)
        print(f"    N={N:5d}: error vs a correct prior = {err:.4f} "
              f"({'ok' if err < 0.03 else 'CHECK'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", help="CSV exported from the responses Sheet")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if a.simulate:
        run_simulation()
    elif a.responses:
        run_real(a.responses, a.alpha)
    else:
        ap.print_help()
