# Collecting quiz responses into a Google Sheet

One-time setup, ~5 minutes. Nothing is sent until step 4 is done.

## 1. Make the Sheet

Create a new Google Sheet. Name the first tab **`responses`** (lower case).
Leave it otherwise empty — the script writes the header row itself on first use.

## 2. Add the script

In the Sheet: **Extensions → Apps Script**. Delete whatever is in `Code.gs`
and paste this in full:

```javascript
const SHEET_NAME = 'responses';

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);                       // serialise concurrent submissions
  try {
    const sh = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);
    const d  = JSON.parse(e.postData.contents);
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
| `hometown` | typed town, or `notgb`, or blank |
| `result` | the place they were matched to |
| `match` | accent-match score 0–100 |
| one column per question | answer value; multi-select answers joined with `\|` |

Nothing is sent when the consent box is unticked, when `endpoint` is empty, or
before the result screen is reached — partial runs are never recorded.

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
