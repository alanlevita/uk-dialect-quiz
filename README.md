# The Great British Dialect Quiz

An interactive dialect atlas of Great Britain. Twenty-five questions about how
you speak, each answered with a map redrawn from published dialect research —
and a final map placing your accent against twenty-three named places.

**Live at [www.ukdialectquiz.org](https://www.ukdialectquiz.org)**

Built by Alan Levita during a research internship at the Intellectual Forum,
Jesus College, Cambridge.

---

## What makes it different

Most dialect quizzes show you a static picture and give you a verdict. Here the
map is the point:

- **Every map is interactive.** Hover any city to see the word actually used
  there and roughly how common it is — on every question, not just at the end.
- **Nothing is invented.** Each surface is redrawn from published research or a
  large-scale survey. Where a source gives counted percentages, those are the
  numbers used. Sources are listed on the results screen and in `SOURCES` in
  `build_quiz.py`.
- **It improves with use.** With consent, completed runs are recorded and can be
  folded back into the maps by `retrain.py`.

## How it works

The site is a **single self-contained `index.html`** — no build step at serve
time, no external requests, no dependencies. Everything (map data, images,
favicon, styles, script) is inlined. That is why the file is large, and why it
is committed rather than gitignored: GitHub Pages serves it directly.

`index.html` is **generated**, so do not edit it by hand. Edit `build_quiz.py`
and rebuild:

```bash
python3 build_quiz.py        # rewrites index.html
```

### Repository layout

| file | role |
|---|---|
| `build_quiz.py` | the source of truth — question set, dialect distributions, map rendering, all page markup, CSS and JS |
| `index.html` | **generated output.** The deployed site. Rebuild, don't edit |
| `britain_pixel_data.json` | the 94×145 land grid, county lookup and place coordinates the maps are drawn on |
| `decoded_maps.json` | per-cell surfaces pre-decoded from source research figures, cached so the build doesn't re-read images |
| `retrain.py` | Beta-Binomial retraining of the maps from collected responses, with a Monte Carlo validation harness |
| `COLLECTION-SETUP.md` | how the Google Sheet response collection and traffic funnel are wired up |
| `CNAME` | custom domain for GitHub Pages |
| `*-pic.*` | question photographs, inlined into `index.html` at build time |

Source research figures are deliberately **not** published — see `.gitignore`.
They are third-party material kept locally for reference; the maps in the quiz
are our own recreations built from them.

### Requirements

Python 3 with `numpy` and `scipy`. No JavaScript toolchain.

```bash
pip3 install numpy scipy
```

### Deploying

```bash
python3 build_quiz.py
git add build_quiz.py index.html
git commit -m "…"
git push
```

GitHub Pages serves `index.html` from the default branch.

## How the maps are built

Each answer has a surface over a 94×145 grid of Great Britain:

1. A value per historic county (`mk()`), from the source data.
2. Gaussian smoothing over land only (`surface()`), so borders don't show.
3. Where a word turns over faster than counties resolve — Bradford's *snicket*
   against Leeds's *ginnel*, ten miles apart — a localised `point_blob()` peak,
   and sometimes a subtraction where a county's headline word bleeds into a city
   that does not share it.

Answers are combined by z-scoring each surface before averaging, so a sharply
localised answer counts as much as a broad national one, then smoothed and
min-max rescaled. The result names the place with the highest neighbourhood mean.

## Data sources

- YouGov, August 2025 (n ≈ 38,000) — school plimsolls, county-level
- YouGov, February 2025 (n > 12,000) — [names for the door-knocking prank](https://yougov.com/en-gb/articles/51544-is-it-knock-down-ginger-or-knock-a-door-run),
  the only source with Scottish coverage for that word
- *Our Dialects* — L. MacKenzie, G. Bailey & D. Turton, [ourdialects.uk](https://www.ourdialects.uk/),
  © George Bailey, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- MacKenzie, Bailey & Turton (2022), *Journal of Linguistic Geography*
- BBC Voices, via Grieve et al. (2019)
- Survey of English Dialects (Orton et al., 1978)
- Tweetolectology Twitter survey (2020–21)
- Starkey Comics dialect surveys

## Privacy

Responses are recorded only when the consent box is ticked **and** the run is
complete — every question answered and the result screen reached. Partial runs
are discarded. No cookies, no third-party analytics, nothing persisted between
visits. See `COLLECTION-SETUP.md` for exactly what is stored.
