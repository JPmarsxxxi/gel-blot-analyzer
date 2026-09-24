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

## Edge cases

- **Zero bands detected:** show the image with no boxes; user can manually
  add bands via the overlay editor. No special empty-state error.
- **Image doesn't look like a gel/blot:** show a warning to the user before/
  after upload (best-effort heuristic check), but still allow them to proceed
  — don't hard-block.
- **Unsupported file type:** reject with a clear message. Accept common raster
  image formats (PNG, JPG/JPEG, TIFF). No explicit file-size cap for V1.
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
