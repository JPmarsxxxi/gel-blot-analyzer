# Gel/Blot Analyzer — SPEC (V1)

Reference: `BUILD_BRIEF.md` for market/competitive context and the patent analysis.
This spec is the implementation contract — someone with no memory of the interview
that produced it should be able to build V1 from this file alone.

## Goal

A web app where a scientist uploads one or more gel/western blot images, gets
bands automatically detected and measured (position, intensity, % of lane, and
optionally molecular weight via ladder calibration), can correct the detection
with a visual overlay editor, labels lanes with sample names, and exports a
clean table/image/PDF — without installing anything or drawing a single box by
hand unless they choose to.

Core loop: **upload → auto-detect → review/correct on the image → label
lanes → export.**

## Non-goals / out of scope (V1)

- **User accounts / login.** No auth. Projects are accessed via an unguessable
  URL (long random token), not tied to an identity. Anyone with the link can
  view/edit — same trust model as an unlisted Google Doc / Figma link.
- **Diagnostic or classification features of any kind.** The tool must never
  store a reference band/lane pattern and compare a new sample against it to
  output a diagnosis, phenotype, or classification label. This is excluded for
  two independent reasons:
  1. It is the specific mechanism claimed by US Patent 11686703 (see
     `BUILD_BRIEF.md`), and this project has no legal clearance to build it.
  2. It would move the product into regulated medical-diagnostics territory
     (FDA-adjacent), which is out of scope for a quantification tool.
  This boundary is about the *runtime, per-sample* comparison-to-diagnose step
  specifically. It does NOT restrict training the band-detection ML model
  (below) on a general dataset — that's a one-time, offline, non-diagnostic
  step and is required, in-scope work.
- **Non-gel image handling beyond a warning.** If the uploaded image doesn't
  look like a gel/blot, show a warning; do not attempt to classify what it
  actually is or block the upload.
- **Any file type beyond common raster images** (see Edge cases).

## Files & interfaces involved

Stack: **Python backend (Flask or FastAPI) + vanilla JS/HTML frontend.**
Image processing and ML via established libraries (numpy, scipy,
scikit-image, and/or OpenCV; a small custom-trained model — see below) —
nothing hand-rolled at the pixel-math level.

Suggested structure (names indicative, not prescriptive):

- `app/server.py` (or `app/main.py`) — routes: upload, project view/edit,
  export.
- `app/models/` — persistence layer: Project, GelImage, Lane, Band records.
- `app/detection/` — lane detection, band detection (ML model inference +
  classical fallback/post-processing), background correction, ladder
  calibration math.
- `app/training/` — scripts to generate synthetic gel images, pull in public
  dataset(s), and train the band/lane segmentation model. Not part of the
  runtime server; run offline to produce a model artifact the server loads.
- `app/static/` + `app/templates/` (or equivalent) — upload UI, overlay
  editor (canvas/SVG over the image), lane label inputs, export controls.
- `storage/` — uploaded images + generated exports, referenced by project
  records.
- Project persistence: a database (SQLite is sufficient for V1) storing one
  row per project, keyed by an unguessable random ID (e.g. UUID4), plus
  associated image file(s) on disk. A project holds one or more gel images
  (batch = multiple images inside one project), each with its own lanes,
  bands, and labels.

## Key decisions & tradeoffs (from interview)

- **Detection approach: custom-trained ML segmentation model**, not the
  classical peak-finding pipeline alone. Chosen deliberately over the
  simpler/faster classical approach, accepting the extra upfront cost of
  building a training pipeline before the product feature works at all.
  - Training data: **both** synthetic gel images (generated programmatically —
    bands of varying width/intensity/position on noisy backgrounds, with
    known-correct labels) **and** whatever public gel/blot datasets can be
    found, combined for scale + realism.
  - Investment level: **higher upfront accuracy bar**, not a bare-minimum
    "good enough" model — expect real time spent on dataset diversity and
    training iteration before calling detection "done" for V1.
  - The classical intensity-profile method may still be used internally as
    part of post-processing (e.g. refining a box to the actual peak) but is
    not the primary detection mechanism.
  - **Model choice (revised after benchmarking):** the default runtime
    detector is a YOLO11n box detector (Ultralytics, AGPL-3.0) trained on the
    same synthetic + GelGenie data; the U-Net segmentation path remains as a
    fallback when ultralytics isn't installed. Chosen by end-to-end band F1
    on 25 unseen gels from other labs: YOLO 0.848, U-Net 0.449, GelGenie's
    released models 0.66-0.73 (app/training/benchmark.py,
    compare_gelgenie.py). A run with more data, 1024 px input and longer
    training scored higher on familiar gels but lower on unseen ones, so the
    simpler model ships.
  - Low-confidence bands are still shown, visually flagged (e.g. dashed/
    different-color outline) rather than hidden — the scientist decides
    whether to keep or delete them via the overlay editor.
- **Lanes: auto-detected, user-adjustable.** Vertical lane boundaries are
  guessed automatically and rendered as draggable dividers the user can nudge.
- **Overlay editing supports all of:** moving/resizing band boxes, adding a
  missed band, deleting a false positive, a global sensitivity slider that
  re-runs detection at a different threshold, and dragging lane boundaries.
- **Quantification: intensity + % of lane.** Each band reports a background-
  subtracted intensity value (sum or mean pixel value inside its box, minus
  local background) and that value as a percentage of its lane's total
  intensity.
- **Ladder/molecular-weight calibration: included.** The user marks one lane
  as the ladder and enters the known size (kDa) for each of its bands; the
  tool fits a curve (typically log-linear against migration distance) to
  estimate molecular weight for bands in other lanes. This is optional per
  project — if no ladder is marked, bands just report position + intensity +
  %, no size.
- **Sample labeling: simple text label per lane.** One text field per detected
  lane (e.g. "Control", "Treated 1h"); shown in the UI overlay and included in
  exports.
- **Image adjustment tools (match GelBox): crop, rotate, brightness/contrast
  sliders, and automatic background correction** (rolling-ball or polynomial
  fit background subtraction, applied before intensity is measured — this
  affects correctness of the numbers, not just appearance).
- **Batch upload: in scope for V1**, implemented as a single project
  containing multiple gel images. The user uploads several images at once;
  each gets its own detected lanes/bands/labels within the same project
  record, and export covers all images in that project together.
- **Project persistence: server-side record, re-openable by URL.** A project
  (one DB row + associated image files) is created on upload and can be
  revisited via its unguessable URL to continue editing. No accounts — the
  URL itself is the access control.
- **Export formats: all three** — CSV table (lane, sample label, band #,
  intensity, % of lane, size in kDa if calibrated), an annotated PNG (image
  with band boxes + lane labels drawn on), and a PDF report combining both.
- **Explicitly excluded: diagnostic/classification.** See Non-goals. This was
  discussed at length during the interview; the decision is to leave it out of
  V1 given patent risk and regulatory scope creep, and revisit only after real
  legal review if ever pursued.

## UI: progressive disclosure (revision after first hands-on use)

First real use found the V1 editor intimidating: every tool visible at once,
delete/resize controls on every band, an unexplained color code, a ladder
button on every lane, and a long table per gel. Same features, same API;
the UI shows what the common path needs and reveals the rest on demand.

- **Upload page:** choosing or dropping files starts the upload immediately
  (no separate button). While detection runs, show a progress state that
  says what is happening. One line explains what happens next.
- **One gel at a time:** a project with several images shows a tab strip of
  thumbnails; the selected gel fills the workspace instead of stacking every
  gel on one long page.
- **Layout:** image on the left, results panel on the right (stacked on narrow
  screens). Top bar: project name, "Copy link", and a single Export menu
  (CSV table, annotated images, PDF report).
- **Summary first:** a one-line summary ("Found 31 bands in 7 lanes") and a
  short, dismissible hint on how to correct the overlay.
- **Clean overlay:** band boxes only. Delete and resize controls appear on the
  selected band; Delete/Backspace removes it. Confident bands are solid,
  low-confidence bands dashed amber (not red, which reads as an error), with a
  legend.
- **Lanes:** a header strip above the image, aligned to the lanes, holds each
  lane's sample-name field. Positions are in image-relative percentages so
  they stay aligned at any size.
- **Adding/removing lanes:** "+ Add lane" then a click: in empty space it adds
  a lane of typical width and detects bands in it (a missed lane); inside a
  lane it splits that lane there, moving bands right of the click (two lanes
  detected as one). Each lane has a remove control (on its name field and in
  the results panel), confirming first when the lane has bands. Lanes are
  renumbered left to right; sample names stay with their lanes.
- **Toolbar:** only the everyday controls: a detection slider labeled
  "Fewer bands / More bands" and "Add band". Rotate, crop, brightness and
  contrast live in a collapsed "Adjust image" section.
- **Results panel:** bands grouped by lane with a small bar for % of lane;
  selecting a band in the panel selects it on the image and vice versa.
  Molecular-weight calibration is a collapsed, optional section with a
  "Ladder lane" picker, replacing the per-lane ladder buttons.
- **Errors** show as an inline message, not a browser alert().

## Public deployment

Released as open source under AGPL-3.0 (required by Ultralytics YOLO) and
hosted as a Docker container: on an Oracle Cloud Always Free server behind
Caddy for HTTPS (`deploy/oracle-setup.sh`), or as a Hugging Face Docker Space.

- **Server:** gunicorn, never the Flask development server or its debugger
  (the Werkzeug debugger allows remote code execution). Port from `PORT`
  (7860 on Spaces). Client IPs come from `X-Forwarded-For` behind the proxy.
- **Upload limits:** at most 25 MB per file and 20 files per upload; larger
  requests are rejected with a clear message before any processing.
- **Rate limiting:** per client IP, 10 uploads per 10 minutes and 300
  requests of any kind per minute; over the limit returns 429 with a message
  saying when to retry. In-memory, per process: good enough for one Space.
- **Retention:** projects and their files are deleted 30 days after their last
  change, checked when the server starts and then hourly.
- **Privacy note:** the upload page says images are stored on the server,
  that anyone with a project's link can view and edit it, and that projects
  are deleted after 30 days of inactivity.
- **Storage:** `GEL_STORAGE_DIR` sets where the database and uploads live
  (`/data` when mounted: the server's `/srv/gel-data` on Oracle, persistent
  storage on a Space; otherwise the container's
  disk, which is wiped on restart).
- **Licences and credits:** repository `LICENSE` is AGPL-3.0; the README
  credits the GelGenie dataset (CC-BY-4.0, Dunn Lab, University of
  Edinburgh), Ultralytics YOLO (AGPL-3.0) and the vendored design skills.

## Edge cases

- **Zero bands detected:** show the image with no boxes; user can manually
  add bands via the overlay editor. No special empty-state error.
- **Image doesn't look like a gel/blot:** show a warning to the user before/
  after upload (best-effort heuristic check), but still allow them to proceed
  — don't hard-block.
- **Unsupported file type:** reject with a clear message. Accept common raster
  image formats (PNG, JPG/JPEG, TIFF). The public deployment caps uploads at
  25 MB per file and 20 files per upload (see Public deployment).
- **Overlapping/touching bands in a lane:** the model may merge or split these
  incorrectly; this is expected to require manual correction via the overlay
  editor (add/delete/resize), not a special algorithmic case.
- **Saturated (overexposed/clipped) bands:** intensity values may be
  underestimated since pixel values are capped; no special handling required
  for V1 beyond the low-confidence visual flag if the model detects this
  reduces its confidence.
- **No ladder marked:** molecular weight is simply omitted from results/
  exports; intensity and % of lane are unaffected.
- **Batch project with a mix of good/bad images:** each image's detection and
  editing is independent within the project; a bad detection on one image
  does not block viewing/editing/exporting the others.
- **Revisiting a project URL after edits:** the project record reflects the
  latest saved state; reopening the URL shows the current overlay/labels, not
  the original auto-detection output.

## Acceptance criteria

1. A user can upload one or more gel/blot images (PNG/JPG/TIFF) in a single
   action and a project is created with an unguessable URL that persists and
   can be reopened later showing the same state.
2. On upload, lanes are automatically detected and rendered as adjustable
   vertical dividers over each image.
3. Bands are automatically detected per lane using the trained model and
   rendered as boxes overlaid on the image, without requiring the user to
   draw any box manually.
4. Bands the model is less confident about are visually distinguishable
   (e.g. different outline style/color) from high-confidence bands.
5. The user can, via the overlay UI: drag a band box's edges to resize it,
   click to add a new band box, delete an existing band box, drag a lane
   divider to a new position, and move a global sensitivity slider that
   re-runs detection and updates the displayed bands.
6. Each detected/edited band displays an intensity value and its percentage
   of that band's lane total, both background-corrected (background
   correction is applied automatically; a manual brightness/contrast
   adjustment and crop/rotate are also available before detection).
7. The user can designate one lane as a ladder, enter known sizes for its
   bands, and see estimated molecular weight (kDa) populate for bands in
   other lanes in that image; omitting this step leaves weight blank without
   errors.
8. The user can type a text label onto each lane, and that label appears in
   the on-screen results and in all exports.
9. From a project, the user can export: a CSV containing lane, sample label,
   band index, intensity, % of lane, and kDa (if calibrated) for every band
   across every image in the project; a PNG per image with band boxes and
   lane labels drawn on it; and a single PDF report combining the annotated
   image(s) and the data table.
10. Uploading a non-image file is rejected with a clear error message before
    any processing is attempted.
11. Uploading an image that doesn't resemble a gel/blot still completes
    upload and detection, but surfaces a visible warning to the user.
12. An image with zero detected bands still loads successfully in the
    overlay editor with no bands shown, and the user can add bands manually.
13. The band/lane detection model used at runtime is a trained artifact
    produced by an offline training script/pipeline (not detection built
    entirely from hand-tuned classical thresholds), trained on a combination
    of generated synthetic gel images and at least one public dataset.
14. No code path in the application stores a reference band/lane pattern and
    outputs a diagnosis/classification/phenotype label based on comparing a
    new sample against it. This should be verifiable by inspecting the
    codebase for any such comparison-to-label logic — there should be none.
15. The deployed server runs without debug mode and rejects a file over
    25 MB, or more than 20 files, with a clear message and no processing.
16. A client exceeding the upload rate limit gets a 429 with a retry message.
17. Projects untouched for 30 days are deleted along with their files.
18. The upload page shows the privacy note; the repository carries an
    AGPL-3.0 LICENSE and credits GelGenie and Ultralytics.

---

# Usage tracking — SPEC

Added 2026-10-13 on branch `feat/usage-tracking`. Self-contained: the V1
sections above describe the app; this section describes only the tracking
feature. Criteria are numbered T1–T15 so they don't collide with V1's 1–18.

## Goal

Let the owner measure whether outreach is working: how many people visit,
upload and export each day, where they came from, how often uploads fail, and
whether projects get used again on a later day. First-party only: no
third-party tracker, no persistent visitor identity. The numbers are read with
a command run on the server; nothing about usage is shown on the website.

## Non-goals / out of scope

- No web page, endpoint or API that returns stats. CLI only.
- No recognising the same person across days (no persistent cookie ID, no
  fixed-salt hash). Unique visitors are counted per UTC day only.
- No third-party analytics scripts, pixels or services.
- No `?ref=` / UTM campaign links. Attribution is the browser referrer only.
- No storing raw IP addresses, user agents, or project IDs in the usage log.
- No charts, dashboards, alerting or email reports.
- No tracking of what is inside images or projects (band counts, names, etc.).
- Tracking must not change any existing behaviour, response or status code.

## Files & interfaces involved

- `app/tracking.py` (new): record events, daily-salted hashing, bot filter,
  referrer cleaning, notrack check, rollup into daily totals.
- `app/models/schema.py`: new tables (created by the existing
  `Base.metadata.create_all` in `app/models/db.py:init_db`; no migration tool):
  - `usage_events`: raw rows. `id`, `created_at` (UTC), `day` (UTC date),
    `kind`, `visitor_hash`, `detail` (string, nullable), `project_hash`
    (nullable).
  - `usage_daily`: rolled-up totals kept forever. `day`, `metric`, `key`
    (string, "" when unused), `count`; unique on (`day`, `metric`, `key`).
  - `usage_salt`: `day` (primary key), `salt` (random 32 bytes, hex).
- `app/routes.py`: new `POST /api/visit` (beacon); record events in
  `create_project` (success and rejection paths) and in the three export
  routes (`export_csv`, `export_png`, `export_pdf`).
- `app/server.py`: record `upload_failed` for 413 (too large) and for 429 on
  `POST /api/projects` (rate-limited); handle `?notrack=`.
- `app/maintenance.py`: the existing hourly purge loop also runs the rollup.
- `app/stats.py` (new): CLI, `python -m app.stats [--days N]`.
- `app/templates/index.html`, `app/templates/project.html` (or
  `app/static/js/upload.js`, `project.js`): send the visit beacon on page
  load; update the privacy note.
- `deploy/oracle.md`, `README.md`: how to read stats on the server
  (`sudo docker exec gel-app python -m app.stats`) and how to exclude
  yourself.
- `tests/test_tracking.py` (new).

## Key decisions & tradeoffs (from interview)

- **Unique visitors are daily only.** `visitor_hash = sha256(salt_for_day +
  client_ip + user_agent)`. The salt is random per UTC day, stored in
  `usage_salt` so a restart mid-day doesn't double-count, and salts for past
  days are deleted, so hashes can't be linked across days or reversed by
  brute-forcing IPs. Chosen over a fixed salt or a cookie for privacy; the
  cost is that returning *people* are invisible.
- **Return use is measured per project instead.** When a project page is
  visited (beacon from `/p/<id>`) for a project whose `created_at` is on an
  earlier UTC day, record a `project_return` event with `project_hash =
  sha256(salt_for_day + project_id)`. Project IDs are edit-capability links,
  so they are never stored raw. The daily metric is the count of distinct
  `project_hash` values that day. Shared-link viewers count too; accepted.
- **Visits are counted from a JavaScript beacon**, not server page loads, so
  most bots (which don't run JS) are excluded. The beacon endpoint also
  ignores user agents matching a bot list (bot, crawl, spider, slurp,
  headless, preview, monitor, curl, wget, python-requests; case-insensitive).
  The beacon sends `document.referrer`, the page type (`upload` or
  `project`) and, on project pages, the project ID. The beacon request's own
  `Referer` header is not used (it would be our own page).
- **Referrer: full address without query or fragment.** Store host + path
  without scheme, e.g. `www.reddit.com/r/labrats/comments/abc123`. Query
  strings and fragments are dropped because they can carry search terms,
  emails or tokens. Referrers from the app's own host are stored as empty
  (internal navigation). Truncate to 300 characters.
- **Excluding the owner: `?notrack=1`.** Any request with `?notrack=1` sets
  a cookie `gel_notrack=1` (1 year, `SameSite=Lax`, `HttpOnly`) that holds no
  identifier. Requests carrying that cookie record nothing (visits, uploads,
  exports, failures). `?notrack=0` clears it.
- **Failures are counted** as `upload_failed` with `detail` = reason:
  `not_image` (file fails image validation), `too_large` (413),
  `rate_limited` (429 on `POST /api/projects`), `bad_request` (any other 4xx
  from `create_project`).
- **Retention: raw rows 30 days, daily totals forever.** The rollup writes
  `usage_daily` for every completed UTC day (before today) not yet rolled up,
  then deletes `usage_events` rows older than `RETENTION_DAYS` and
  `usage_salt` rows for days before today. Rollup is idempotent: re-running
  it for a day replaces that day's totals, never doubles them.
- **Tracking never breaks the app.** Every recording call is wrapped so an
  exception is logged and swallowed; the request proceeds exactly as it
  would without tracking. The beacon endpoint always returns 204.
- **Days are UTC.**

## Events and metrics

| Event kind | Recorded when | Daily metrics (`metric`, `key`) |
|---|---|---|
| `visit` | beacon from upload or project page | `visits` (key = page type); `visitors` (distinct `visitor_hash`, key ""); `referrer` (key = cleaned referrer, non-empty only) |
| `upload` | `POST /api/projects` succeeds; `detail` = number of images | `uploads` (projects created); `images_uploaded` (sum of detail) |
| `upload_failed` | see failures above; `detail` = reason | `upload_failed` (key = reason) |
| `export` | an export route returns 200; `detail` = csv/png/pdf | `exports` (key = format) |
| `project_return` | project-page beacon for a project created on an earlier UTC day | `returning_projects` (distinct `project_hash`) |

## CLI

`python -m app.stats [--days N]` (default 30) prints, for each of the last N
UTC days that has any data, newest first: visitors, visits, uploads, images,
exports (by format), failed uploads (by reason), returning projects. Then
the top 10 referrers over those days with counts. Then all-time totals (sum
of `usage_daily` plus not-yet-rolled raw rows); visitors are summed per day
and labelled "visitor-days", since daily uniques can't be deduplicated
across days. It reads both rolled totals and raw rows, so today's numbers
are current. On an empty database it prints "No usage recorded yet." and
exits 0.

## Edge cases

- Missing `User-Agent` or IP: hash with empty strings; still counted.
- `document.referrer` empty, malformed or not http(s): stored as empty.
- Beacon for a project ID that doesn't exist: record the `visit`, skip
  `project_return`; still 204.
- Beacon body missing or invalid JSON: 204, record a visit with page type
  `unknown`.
- Beacon flood: covered by the existing per-IP request rate limit.
- Server restart mid-day: the salt is read from `usage_salt`, so that day's
  uniques stay consistent.
- Two requests creating the same day's salt at once: insert-if-absent, then
  read back (there is one worker, but don't rely on it).
- Rollup after downtime spanning several days: each missing completed day is
  rolled up once; existing totals for a day are replaced, not added to.
- Project deleted by retention: its past `project_return` counts remain in
  the totals (only hashes were stored).
- `gel_notrack` cookie present on an upload that fails: nothing recorded.
- Tests must be able to fix the clock/day (pass `now`, as
  `purge_stale_projects` already does).

## Acceptance criteria

T1. `KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/ -q` passes, apart from
    failures that also occur on `main` (currently
    `test_reject_non_image_file_before_processing`, caused by a missing
    `pi_heif` package inside ultralytics).
T2. The upload page and project page each send one `POST /api/visit` on load
    (present in the served HTML/JS); the endpoint returns 204 and creates one
    `visit` row; a request with a bot user agent (e.g. containing
    `Googlebot`) creates none.
T3. Two visits on the same UTC day from the same IP + user agent give
    `visitors = 1`; the same pair on the next day gets a different
    `visitor_hash` (test by injecting the day).
T4. No usage table row contains the raw IP address, user agent string or
    project ID used in a test that records every event kind.
T5. A referrer `https://www.reddit.com/r/labrats/comments/abc?utm=x#top` is
    stored as `www.reddit.com/r/labrats/comments/abc`; a referrer on the
    app's own host is stored as empty.
T6. A successful upload of 3 images records one `upload` event with detail
    3; each successful export (csv, png, pdf) records one `export` event with
    the matching format.
T7. A non-image upload records `upload_failed`/`not_image`; an oversized
    upload records `upload_failed`/`too_large`; a rate-limited upload records
    `upload_failed`/`rate_limited`. Status codes and response bodies are the
    same as on `main`.
T8. A project-page beacon for a project created on an earlier UTC day
    records `project_return`; two such beacons that day give
    `returning_projects = 1`; a project created the same day records none.
T9. A request with `?notrack=1` sets the `gel_notrack` cookie; afterwards,
    visits, uploads, exports and failures from that client record nothing;
    `?notrack=0` clears the cookie.
T10. Rollup produces correct `usage_daily` totals for a past day, deletes raw
    rows older than 30 days and salts for past days, and running it twice
    leaves the totals unchanged.
T11. If recording raises (e.g. a usage table is missing), uploads, exports
    and page loads still succeed with their normal responses.
T12. `python -m app.stats` prints per-day rows, top referrers and all-time
    totals as described under CLI; on an empty database it prints
    "No usage recorded yet." and exits 0.
T13. No route returns usage data: `/stats` and `/api/stats` return 404, and
    no template renders usage numbers.
T14. No third-party script, pixel or external request is added to any
    template or JS file.
T15. The upload page privacy note says that anonymous daily visit counts and
    the referring page are recorded, with no IP addresses stored and no
    cookies other than the opt-out one; `deploy/oracle.md` documents
    `sudo docker exec gel-app python -m app.stats` and `?notrack=1`.
