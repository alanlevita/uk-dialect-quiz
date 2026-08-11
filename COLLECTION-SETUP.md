# Collecting quiz responses into a Google Sheet

One-time setup, ~5 minutes. Nothing is sent until step 4 is done.

## 1. Make the Sheet

Create a new Google Sheet. Name the first tab **`responses`** (lower case).
Leave it otherwise empty — the script writes the header row itself on first use.

## 2. Add the script

In the Sheet: **Extensions → Apps Script**. Delete whatever is in `Code.gs`
and paste this in full:

```javascript
const SHEET_NAME  = 'responses';
const TRAFFIC_TAB = 'traffic';

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);                       // serialise concurrent submissions
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const d  = JSON.parse(e.postData.contents);

    // ---- traffic events (visit / start / complete) go to their own tab ----
    if (d.type === 'event') {
      let t = ss.getSheetByName(TRAFFIC_TAB);
      if (!t) {
        t = ss.insertSheet(TRAFFIC_TAB);
        t.appendRow(['received', 'event', 'run', 'referrer', 'viewport', 'version']);
      }
      t.appendRow([new Date(), d.event || '', d.run || '',
                   d.ref || '(direct)', d.vw || '', d.version || '']);
      return ContentService.createTextOutput('ok');
    }

    const sh = ss.getSheetByName(SHEET_NAME);
    const a  = d.answers || {};

    // Header is derived from the first row written, then reused. Any question
    // added later appends a new column instead of shifting existing ones.
    let header = sh.getLastRow()
      ? sh.getRange(1, 1, 1, sh.getLastColumn()).getValues()[0]
      : [];
    if (!header.length) {
      header = ['received', 'version', 'run', 'client_ts', 'hometown', 'result', 'match'];
      sh.appendRow(header);
    }
    Object.keys(a).forEach(function (k) {
      if (header.indexOf(k) === -1) {
        header.push(k);
        sh.getRange(1, header.length).setValue(k);
      }
    });

    const base = {
      received:  new Date(),
      version:   d.version || '',
      run:       d.run || '',
      client_ts: d.ts || '',
      hometown:  d.hometown || '',
      result:    d.result || '',
      match:     d.match === '' ? '' : d.match
    };
    const row = header.map(function (col) {
      if (col in base) return base[col];
      const v = a[col];
      return Array.isArray(v) ? v.join('|') : (v === undefined ? '' : v);
    });
    sh.appendRow(row);

    return ContentService.createTextOutput('ok');
  } catch (err) {
    return ContentService.createTextOutput('err');
  } finally {
    lock.releaseLock();
  }
}
```

## 3. Deploy it

**Deploy → New deployment → ⚙ → Web app**, then:

- **Execute as:** Me
- **Who has access:** **Anyone**  ← required, or browsers get a login redirect
- Click **Deploy**, approve the permission prompt, and copy the
  **Web app URL** (ends in `/exec`).

## 4. Point the quiz at it

In `build_quiz.py`, find `const COLLECT=` and paste the URL:

```javascript
const COLLECT={
  endpoint:"https://script.google.com/macros/s/AKfy...../exec",
  version:"2026-08-07"
};
```

Then rebuild and push:

```bash
python3 build_quiz.py && git add -A && git commit -m "Enable response collection" && git push
```

## What gets sent

One row per completed run, and only if the consent box was ticked:

| column | meaning |
|---|---|
| `received` | when the Sheet recorded it |
| `version` | question-set version (bump `COLLECT.version` when questions change) |
| `run` | random per-page-load id, not persisted — dedupes accidental double-submits only |
| `client_ts` | the visitor's clock |
| `hometown` | typed town, or `notgb` (never blank — the run wouldn't be sent) |
| `result` | the place they were matched to |
| `match` | accent-match score 0–100 |
| one column per question | answer value; multi-select answers joined with `\|` |

A run is sent **only** when all of the following hold, checked in
`runIsComplete()` at the moment the result screen renders:

- the consent box was ticked
- `endpoint` is set
- the result screen has actually been reached (Finish was pressed)
- **every** question carries an answer, with multi-select questions holding at
  least one selection

Fail any one of them and nothing leaves the browser. The run also stays eligible
— the send is only marked done once it succeeds — so a visitor who backs up,
fills in what was missing and finishes properly is still recorded. Every row in
the Sheet is therefore a full run; there are no partial rows to filter out.

Hometown is *not* required to record a run — it's research metadata, not part of
the dialect record. Filter on it in `retrain.py`, where it actually matters.

## Site traffic

The `traffic` tab is created automatically on the first hit and gives a funnel:

| event | fired when |
|---|---|
| `visit` | the landing page finishes loading |
| `start` | "Start the quiz" is pressed |
| `complete` | the result screen renders |

Columns are `received`, `event`, `run`, `referrer`, `viewport`, `version`. Three
formulas at the top of a blank tab give you the whole picture:

```
=COUNTIF(traffic!B:B,"visit")
=COUNTIF(traffic!B:B,"start")/COUNTIF(traffic!B:B,"visit")
=COUNTIF(traffic!B:B,"complete")/COUNTIF(traffic!B:B,"start")
```

Group by `referrer` for traffic sources, or by `viewport` for the mobile split.

**Privacy shape.** No cookie, no localStorage, no third-party script, nothing
persisted between visits. `run` is regenerated on every page load, so two visits
by the same person are indistinguishable from two different people — which means
these are *page* counts, not *visitor* counts. The referrer is cut down to its
host (`reddit.com`, not the thread URL, and never a search query), and viewport
is rounded to the nearest 100px. Direct hits record `(direct)`.

Unlike the quiz answers, these three events fire **without** consent, because
they carry nothing about the person. That's a defensible line, but it is your
call rather than mine — if the Intellectual Forum's ethics position is that
*nothing* leaves the browser before the box is ticked, move `pingEvent("visit")`
inside `startQuiz()` and you lose only the visit→start drop-off rate.

## Re-deploying after script edits

Apps Script keeps the old code live until you redeploy. After editing:
**Deploy → Manage deployments → ✏️ → Version: New version → Deploy.**
The URL stays the same.

## One thing to check before you launch

The consent text currently promises that **no IP address is recorded**. Google
logs request metadata for Apps Script executions, so that promise is stronger
than the setup can strictly guarantee. Either soften the wording (e.g. "we do
not store your IP address with your answers") or confirm the position with
whoever signs off ethics at the Intellectual Forum.
