# Gel/blot analyzer — build brief

Point a new session at this file. Everything below is checked, not assumed —
sources noted throughout.

## What we're building

A tool that automatically finds bands in a gel/western blot image and measures how
strong each one is — replacing the manual "draw a box around every band, click
through menus, repeat for every gel" process scientists currently do by hand in
ImageJ. Every molecular biology lab does this constantly; it's genuinely tedious.

## Who's already in this space, checked directly — not assumed

| Tool | What it is | Real weakness | Adoption |
|---|---|---|---|
| **ImageJ** | Free, the default | Fully manual — you draw the box yourself, every band, every time | Massive. This is the habit to beat, not a competitor to out-feature. |
| **Phoretix 1D** (TotalLab, 10 employees, founded 2007) | Paid, automated band detection + batch + PDF reports | Desktop-only .exe, no public pricing (sales-call model), a separate program to install and learn | Real institutional trust ("premium product on the market") but small company, old distribution model |
| **GelBox** (Univ. of Kentucky, published 2024) | Free, open-source, image adjustment + background correction + band-fitting + metadata linking, built for scientific rigor/reproducibility | Needs MATLAB or a standalone build; still manual band-fitting workflow, not automatic | **Only 3 GitHub stars, 1 fork.** Real, legitimate, academically sound — but essentially unused in practice. Not a market threat. |

**⭐ GelBox is the reference for what a rigorous tool should be capable of — match
everything it does, then go further. It is NOT who we're competing against for
users; almost nobody uses it. The actual competitor is ImageJ's habit.**

## The patent — checked, and it does not block this

**US Patent 11686703, assignee Quest Diagnostics** (a large clinical diagnostics
company; inventors Saratkar, Naides, Cleveland). Read the actual claims directly
(not just the description):

1. Convert gel image to grayscale
2. Detect lanes
3. Compute an intensity vector per lane
4. **Calculate a Pearson correlation score against a reference lane intensity
   vector stored in a database**
5. **Assign a classification/phenotype** based on that correlation score
6. A "drift corrector" normalises for uneven migration across lanes

**This is a diagnostic pattern-matching method — comparing a lane against a known
disease signature to classify it.** It is NOT plain band detection + intensity
measurement, which is a simpler, different, and much older technique (ImageJ,
Phoretix, and GelBox have all done it for years with no patent dispute found).

⚠️ **The one rule that keeps this clear: never build step 4/5 above.** Detect bands,
measure their intensity, stop there. **Do not correlate a lane's pattern against a
stored reference/database of known patterns to classify or diagnose it** — that
specific mechanism is what's claimed. Plain quantification is a different, unclaimed
method as far as the claim language goes. (This is an informed reading of the actual
claims, not a certified legal clearance — flag for a real check if this ever becomes
a funded, high-stakes product rather than a demo/pilot.)

## How to actually beat GelBox — the product decisions

GelBox's feature list is the checklist. Match all of it. Then go further on the
thing GelBox and Phoretix both get wrong: **making it something a researcher would
actually love using, not just tolerate.**

**Match (from GelBox):**
- Image adjustment (crop, rotate, brightness/contrast)
- Background correction
- Band-fitting / quantification
- Metadata linking (which lane = which sample)
- One unified file that preserves the whole analysis — traceability matters for
  real scientific rigor, not just speed. Keep this even though it's "boring."

**Go further than GelBox or Phoretix, on intuitiveness specifically:**
- **Bands detected automatically by default** — the tool does the work, the
  scientist only nudges/corrects if something looks off. GelBox and Phoretix both
  still make the user do real manual fitting work; flip the default effort.
- **No install, no MATLAB, no licence call.** Upload an image, get a result. Removes
  the single biggest adoption barrier both real competitors have.
- **Immediate visual feedback** — see the detected bands highlighted on the image
  right away, adjust with a slider, not a menu tree.
- **Batch upload** — drop in every gel from an experiment at once. Phoretix does
  this but hides it behind a sales call; make it the obvious default.
- **One-click export** — a clean table/image, ready to paste into a lab notebook or
  a paper, no reformatting.

## V1 scope, kept small on purpose

Take one gel image → detect bands automatically → measure intensity → show the
result with an editable overlay → export a clean table. That's the whole first
version. Batch, accounts, anything beyond one image comes after someone real says
"I'd use this."
