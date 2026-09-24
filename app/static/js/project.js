const SVG_NS = "http://www.w3.org/2000/svg";
const projectId = document.body.dataset.projectId;

document.getElementById("export-csv").href = `/api/projects/${projectId}/export/csv`;
document.getElementById("export-png").href = `/api/projects/${projectId}/export/png`;
document.getElementById("export-pdf").href = `/api/projects/${projectId}/export/pdf`;

const workspace = document.getElementById("workspace");
const tabsEl = document.getElementById("gel-tabs");
const panelTemplate = document.getElementById("image-panel-template");
const toastEl = document.getElementById("toast");
const exportMenu = document.getElementById("export-menu");

let activePanel = null;
let toastTimer = null;

function toast(message, isError = true) {
  toastEl.textContent = message;
  toastEl.classList.toggle("toast-ok", !isError);
  toastEl.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toastEl.hidden = true), 4000);
}

async function api(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "Something went wrong. Please try again.");
  return data;
}

function jsonBody(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

// localStorage can throw (private mode, blocked storage); the hint just reappears then.
function hintDismissed() {
  try {
    return localStorage.getItem("gel-hint-dismissed") === "1";
  } catch {
    return false;
  }
}

function dismissHint() {
  try {
    localStorage.setItem("gel-hint-dismissed", "1");
  } catch {}
}

document.getElementById("copy-link").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(window.location.href);
    toast("Link copied. Anyone with it can view and edit this project.", false);
  } catch {
    toast("Couldn't copy automatically. Copy the address from your browser's address bar.");
  }
});

document.addEventListener("click", (e) => {
  if (!exportMenu.contains(e.target)) exportMenu.open = false;
});

document.addEventListener("keydown", (e) => {
  if (!activePanel || e.target.closest("input, select, textarea")) return;
  if ((e.key === "Delete" || e.key === "Backspace") && activePanel.selectedBandId) {
    e.preventDefault();
    activePanel.deleteBand(activePanel.selectedBandId);
  } else if (e.key === "Escape") {
    activePanel.setMode("select");
    activePanel.select(null);
  }
});

function screenToImageCoords(svg, evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}

function pct(value, total) {
  return `${(value / total) * 100}%`;
}

function formatNumber(n) {
  return Math.round(n).toLocaleString();
}

class GelPanel {
  constructor(image) {
    this.image = image;
    this.mode = "select"; // "select" | "add-band" | "crop"
    this.drag = null;
    this.cropRect = null;
    this.selectedBandId = null;

    const node = panelTemplate.content.cloneNode(true);
    this.el = node.querySelector(".gel-panel");
    const q = (sel) => this.el.querySelector(sel);
    this.imgEl = q(".gel-image");
    this.svg = q(".gel-overlay");
    this.laneHeaderEl = q(".lane-header");
    this.lanesListEl = q(".lanes-list");
    this.summaryEl = q(".summary");
    this.warningEl = q(".gel-warning");
    this.modeHintEl = q(".mode-hint");
    this.ladderSelect = q(".ladder-select");
    this.ladderBandsEl = q(".ladder-bands");
    this.applyCalibrationBtn = q(".apply-calibration-btn");
    this.addBandBtn = q(".add-band-toggle-btn");
    this.cropBtn = q(".crop-toggle-btn");

    const hint = q(".hint");
    hint.hidden = hintDismissed();
    q(".hint-close").addEventListener("click", () => {
      dismissHint();
      document.querySelectorAll(".hint").forEach((h) => (h.hidden = true));
    });

    this._wireToolbar();
    this._wireAdjustDrawer();
    this._wireCalibration();
    this._wireSvgEvents();

    workspace.appendChild(this.el);
    this.render();
  }

  // ------------------------------------------------------------- toolbar --

  _wireToolbar() {
    const sensitivity = this.el.querySelector(".adj-sensitivity");
    sensitivity.value = this.image.sensitivity;
    let timer = null;
    sensitivity.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        this.el.classList.add("working");
        try {
          this.image = await api(
            `/api/images/${this.image.id}/sensitivity`,
            jsonBody("PATCH", { sensitivity: parseFloat(sensitivity.value) })
          );
          this.selectedBandId = null;
          this.render();
        } catch (err) {
          toast(err.message);
        } finally {
          this.el.classList.remove("working");
        }
      }, 300);
    });

    this.addBandBtn.addEventListener("click", () => {
      this.setMode(this.mode === "add-band" ? "select" : "add-band");
    });
  }

  setMode(mode) {
    this.mode = mode;
    this.addBandBtn.classList.toggle("active", mode === "add-band");
    this.cropBtn.classList.toggle("active", mode === "crop");
    this.svg.classList.toggle("adding", mode === "add-band");
    this.svg.classList.toggle("cropping", mode === "crop");
    const hints = {
      "add-band": "Click on a band in the image to add a box there. Press Esc when you're done.",
      crop: "Drag a rectangle around the part of the image to keep, then press Apply.",
    };
    this.modeHintEl.textContent = hints[mode] || "";
    this.modeHintEl.hidden = !hints[mode];
    if (mode !== "crop") {
      this.cropRect = null;
      this._drawCropRect();
    }
  }

  _wireAdjustDrawer() {
    const toggle = this.el.querySelector(".adjust-toggle-btn");
    const drawer = this.el.querySelector(".adjust-drawer");
    const rotate = this.el.querySelector(".adj-rotate");
    const brightness = this.el.querySelector(".adj-brightness");
    const contrast = this.el.querySelector(".adj-contrast");
    const applyBtn = this.el.querySelector(".apply-adjustments-btn");

    // Live CSS preview only; the real pixel adjustment happens server-side on Apply.
    const preview = () => {
      this.imgEl.style.filter = `brightness(${1 + brightness.value / 150}) contrast(${contrast.value})`;
      this.imgEl.style.transform = rotate.value ? `rotate(${rotate.value}deg)` : "";
    };
    const reset = () => {
      rotate.value = 0;
      brightness.value = 0;
      contrast.value = 1;
      this.imgEl.style.filter = "";
      this.imgEl.style.transform = "";
      this.setMode("select");
    };
    const setOpen = (open) => {
      drawer.hidden = !open;
      toggle.setAttribute("aria-expanded", String(open));
      toggle.classList.toggle("active", open);
      if (!open) reset();
    };

    toggle.addEventListener("click", () => setOpen(drawer.hidden));
    [rotate, brightness, contrast].forEach((input) => input.addEventListener("input", preview));
    this.cropBtn.addEventListener("click", () => this.setMode(this.mode === "crop" ? "select" : "crop"));
    this.el.querySelector(".cancel-adjustments-btn").addEventListener("click", () => setOpen(false));

    applyBtn.addEventListener("click", async () => {
      const body = {
        rotate_deg: parseFloat(rotate.value) || 0,
        brightness: parseFloat(brightness.value) || 0,
        contrast: parseFloat(contrast.value) || 1,
      };
      if (this.cropRect && this.cropRect.w > 2 && this.cropRect.h > 2) {
        Object.assign(body, { crop_x: this.cropRect.x, crop_y: this.cropRect.y, crop_w: this.cropRect.w, crop_h: this.cropRect.h });
      }
      applyBtn.disabled = true;
      applyBtn.textContent = "Applying…";
      this.el.classList.add("working");
      try {
        this.image = await api(`/api/images/${this.image.id}`, jsonBody("PATCH", body));
        this.selectedBandId = null;
        setOpen(false);
        this.render();
      } catch (err) {
        toast(err.message);
      } finally {
        applyBtn.disabled = false;
        applyBtn.textContent = "Apply";
        this.el.classList.remove("working");
      }
    });
  }

  // --------------------------------------------------------- calibration --

  _wireCalibration() {
    this.ladderSelect.addEventListener("change", async () => {
      const chosenId = this.ladderSelect.value;
      const current = this.image.lanes.find((l) => l.is_ladder);
      try {
        if (current && String(current.id) !== chosenId) {
          this._patchLane(await api(`/api/lanes/${current.id}`, jsonBody("PATCH", { is_ladder: false })));
        }
        if (chosenId) {
          for (const l of this.image.lanes) l.is_ladder = false;
          this._patchLane(await api(`/api/lanes/${chosenId}`, jsonBody("PATCH", { is_ladder: true })));
        }
        this.render();
      } catch (err) {
        toast(err.message);
      }
    });

    this.applyCalibrationBtn.addEventListener("click", async () => {
      try {
        for (const input of this.ladderBandsEl.querySelectorAll("input")) {
          const val = input.value.trim();
          await api(`/api/bands/${input.dataset.bandId}`, jsonBody("PATCH", { known_kda: val === "" ? null : parseFloat(val) }));
        }
        this.image = await api(`/api/images/${this.image.id}/calibrate`, { method: "POST" });
        this.render();
        const estimated = this.image.lanes.some((l) => !l.is_ladder && l.bands.some((b) => b.estimated_kda != null));
        toast(estimated ? "Sizes estimated for the other lanes." : "Enter at least two ladder sizes to estimate the rest.", !estimated);
      } catch (err) {
        toast(err.message);
      }
    });
  }

  // -------------------------------------------------------------- events --

  _wireSvgEvents() {
    this.svg.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    document.addEventListener("pointermove", (e) => this._onPointerMove(e));
    document.addEventListener("pointerup", () => this._onPointerUp());
  }

  _onPointerDown(e) {
    const target = e.target;
    const pt = screenToImageCoords(this.svg, e);

    if (this.mode === "crop") {
      this.drag = { type: "crop-draw", start: pt };
      return;
    }

    if (this.mode === "add-band") {
      const lane = this.image.lanes.find((l) => pt.x >= l.x_start && pt.x <= l.x_end);
      if (!lane) return;
      const h = Math.max(10, this.image.height * 0.03);
      const w = (lane.x_end - lane.x_start) * 0.6;
      this._createBand(lane.id, { x: pt.x - w / 2, y: pt.y - h / 2, width: w, height: h });
      return;
    }

    const bandId = target.dataset.bandId;
    if (target.classList.contains("band-delete-hit")) {
      this.deleteBand(bandId);
    } else if (target.classList.contains("band-handle")) {
      const band = this._findBand(bandId);
      this.drag = { type: "resize", band, start: pt, origW: band.width, origH: band.height };
    } else if (target.classList.contains("band-box")) {
      if (String(this.selectedBandId) !== bandId) this.select(bandId);
      const band = this._findBand(bandId);
      this.drag = { type: "move", band, start: pt, origX: band.x, origY: band.y };
    } else if (target.classList.contains("lane-divider-hit")) {
      this.drag = { type: "divider", leftId: target.dataset.leftLaneId, rightId: target.dataset.rightLaneId };
    } else {
      this.select(null);
    }
  }

  _onPointerMove(e) {
    if (!this.drag) return;
    const pt = screenToImageCoords(this.svg, e);
    const drag = this.drag;

    if (drag.type === "crop-draw") {
      this.cropRect = {
        x: Math.min(drag.start.x, pt.x),
        y: Math.min(drag.start.y, pt.y),
        w: Math.abs(pt.x - drag.start.x),
        h: Math.abs(pt.y - drag.start.y),
      };
      this._drawCropRect();
    } else if (drag.type === "move") {
      drag.newX = drag.origX + pt.x - drag.start.x;
      drag.newY = drag.origY + pt.y - drag.start.y;
      this._placeBandShapes(drag.band.id, drag.newX, drag.newY, drag.band.width, drag.band.height);
    } else if (drag.type === "resize") {
      drag.newW = Math.max(5, drag.origW + pt.x - drag.start.x);
      drag.newH = Math.max(5, drag.origH + pt.y - drag.start.y);
      this._placeBandShapes(drag.band.id, drag.band.x, drag.band.y, drag.newW, drag.newH);
    } else if (drag.type === "divider") {
      drag.newX = pt.x;
      for (const line of this.svg.querySelectorAll(`[data-left-lane-id="${drag.leftId}"]`)) {
        line.setAttribute("x1", pt.x);
        line.setAttribute("x2", pt.x);
      }
    }
  }

  async _onPointerUp() {
    if (!this.drag) return;
    const drag = this.drag;
    this.drag = null;

    try {
      if (drag.type === "move" && drag.newX !== undefined) {
        this._patchLane(await api(`/api/bands/${drag.band.id}`, jsonBody("PATCH", { x: drag.newX, y: drag.newY })));
        this.render();
      } else if (drag.type === "resize" && drag.newW !== undefined) {
        this._patchLane(await api(`/api/bands/${drag.band.id}`, jsonBody("PATCH", { width: drag.newW, height: drag.newH })));
        this.render();
      } else if (drag.type === "divider" && drag.newX !== undefined) {
        this._patchLane(await api(`/api/lanes/${drag.leftId}`, jsonBody("PATCH", { x_end: drag.newX })));
        this._patchLane(await api(`/api/lanes/${drag.rightId}`, jsonBody("PATCH", { x_start: drag.newX })));
        this.render();
      }
    } catch (err) {
      toast(err.message);
      this.render();
    }
  }

  // ------------------------------------------------------------ bands --

  _findBand(bandId) {
    for (const lane of this.image.lanes) {
      const band = lane.bands.find((b) => String(b.id) === String(bandId));
      if (band) return band;
    }
    return null;
  }

  _patchLane(updatedLane) {
    const idx = this.image.lanes.findIndex((l) => l.id === updatedLane.id);
    if (idx !== -1) this.image.lanes[idx] = updatedLane;
  }

  select(bandId) {
    this.selectedBandId = bandId;
    this._renderOverlay();
    for (const row of this.lanesListEl.querySelectorAll(".band-row")) {
      const on = row.dataset.bandId === String(bandId);
      row.classList.toggle("selected", on);
      if (on) row.scrollIntoView({ block: "nearest" });
    }
  }

  async _createBand(laneId, box) {
    try {
      const lane = await api(`/api/lanes/${laneId}/bands`, jsonBody("POST", box));
      this._patchLane(lane);
      const newest = lane.bands.reduce((a, b) => (b.id > a.id ? b : a), lane.bands[0]);
      this.selectedBandId = newest ? newest.id : null;
      this.render();
    } catch (err) {
      toast(err.message);
    }
  }

  async deleteBand(bandId) {
    try {
      this._patchLane(await api(`/api/bands/${bandId}`, { method: "DELETE" }));
      this.selectedBandId = null;
      this.render();
    } catch (err) {
      toast(err.message);
    }
  }

  // -------------------------------------------------------------- render --

  render() {
    const img = this.image;
    if (this.imgEl.getAttribute("src") !== img.url) this.imgEl.src = img.url;
    this.svg.setAttribute("viewBox", `0 0 ${img.width} ${img.height}`);
    this.warningEl.hidden = img.is_gel_like;

    this._renderSummary();
    this._renderOverlay();
    this._renderLaneHeader();
    this._renderLanesList();
    this._renderCalibration();
    renderTabs();
  }

  bandCount() {
    return this.image.lanes.reduce((n, l) => n + l.bands.length, 0);
  }

  _renderSummary() {
    const bands = this.bandCount();
    const lanes = this.image.lanes.length;
    const unsure = this.image.lanes.reduce((n, l) => n + l.bands.filter((b) => b.low_confidence).length, 0);
    if (bands === 0) {
      this.summaryEl.textContent = "No bands found. Move the slider toward “More bands”, or use + Add band.";
      return;
    }
    let text = `Found ${bands} band${bands === 1 ? "" : "s"} in ${lanes} lane${lanes === 1 ? "" : "s"}.`;
    if (unsure) text += ` ${unsure} ${unsure === 1 ? "is" : "are"} dashed — worth a quick look.`;
    this.summaryEl.textContent = text;
  }

  _svgEl(tag, attrs, parent = this.svg) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    parent.appendChild(el);
    return el;
  }

  _renderOverlay() {
    this.svg.innerHTML = "";
    const img = this.image;

    for (let i = 0; i < img.lanes.length - 1; i++) {
      const left = img.lanes[i];
      const right = img.lanes[i + 1];
      const attrs = { x1: left.x_end, x2: left.x_end, y1: 0, y2: img.height, "data-left-lane-id": left.id, "data-right-lane-id": right.id };
      this._svgEl("line", { ...attrs, class: "lane-divider" });
      this._svgEl("line", { ...attrs, class: "lane-divider-hit" });
    }

    let selected = null;
    for (const lane of img.lanes) {
      for (const band of lane.bands) {
        const isSelected = String(band.id) === String(this.selectedBandId);
        if (isSelected) selected = band;
        this._svgEl("rect", {
          class: "band-box" + (band.low_confidence ? " low-confidence" : "") + (isSelected ? " selected" : ""),
          x: band.x,
          y: band.y,
          width: band.width,
          height: band.height,
          "data-band-id": band.id,
        });
      }
    }

    // Edit controls only on the selected band: on a busy gel, per-band controls bury the image.
    if (selected) {
      const r = Math.max(5, img.width * 0.007);
      this._svgEl("circle", { class: "band-handle", r, "data-band-id": selected.id });
      const del = this._svgEl("g", { class: "band-delete", "data-band-id": selected.id });
      this._svgEl("circle", { class: "band-delete-hit", r: r * 1.3, "data-band-id": selected.id }, del);
      const text = this._svgEl("text", { "font-size": r * 2, "data-band-id": selected.id }, del);
      text.textContent = "×";
      this._placeBandShapes(selected.id, selected.x, selected.y, selected.width, selected.height);
    }

    this._drawCropRect();
  }

  _placeBandShapes(bandId, x, y, w, h) {
    const rect = this.svg.querySelector(`rect.band-box[data-band-id="${bandId}"]`);
    if (rect) {
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", w);
      rect.setAttribute("height", h);
    }
    const handle = this.svg.querySelector(`circle.band-handle[data-band-id="${bandId}"]`);
    if (handle) {
      handle.setAttribute("cx", x + w);
      handle.setAttribute("cy", y + h);
    }
    const del = this.svg.querySelector(`g.band-delete[data-band-id="${bandId}"]`);
    if (del) del.setAttribute("transform", `translate(${x + w}, ${y})`);
  }

  _drawCropRect() {
    const existing = this.svg.querySelector("rect.crop-rect");
    if (existing) existing.remove();
    if (!this.cropRect) return;
    this._svgEl("rect", { class: "crop-rect", x: this.cropRect.x, y: this.cropRect.y, width: this.cropRect.w, height: this.cropRect.h });
  }

  _renderLaneHeader() {
    this.laneHeaderEl.innerHTML = "";
    const img = this.image;
    for (const lane of img.lanes) {
      const slot = document.createElement("div");
      slot.className = "lane-slot" + (lane.is_ladder ? " is-ladder" : "");
      slot.style.left = pct(lane.x_start, img.width);
      slot.style.width = pct(lane.x_end - lane.x_start, img.width);

      const input = document.createElement("input");
      input.className = "lane-name";
      input.placeholder = lane.is_ladder ? "Ladder" : `${lane.index + 1}`;
      input.title = `Sample name for lane ${lane.index + 1}`;
      input.setAttribute("aria-label", `Sample name for lane ${lane.index + 1}`);
      input.value = lane.label || "";
      let timer = null;
      input.addEventListener("input", () => {
        clearTimeout(timer);
        timer = setTimeout(async () => {
          try {
            this._patchLane(await api(`/api/lanes/${lane.id}`, jsonBody("PATCH", { label: input.value })));
            this._renderLanesList();
            this._renderCalibration();
          } catch (err) {
            toast(err.message);
          }
        }, 400);
      });
      slot.appendChild(input);
      this.laneHeaderEl.appendChild(slot);
    }
  }

  _laneName(lane) {
    return lane.label || (lane.is_ladder ? "Ladder" : "");
  }

  _renderLanesList() {
    this.lanesListEl.innerHTML = "";
    for (const lane of this.image.lanes) {
      const card = document.createElement("div");
      card.className = "lane-card";

      const head = document.createElement("div");
      head.className = "lane-card-head";
      const num = document.createElement("span");
      num.className = "lane-num";
      num.textContent = `Lane ${lane.index + 1}`;
      const name = document.createElement("span");
      name.className = "lane-card-name";
      name.textContent = this._laneName(lane) || "unnamed";
      name.classList.toggle("muted", !this._laneName(lane));
      const count = document.createElement("span");
      count.className = "lane-count";
      count.textContent = lane.bands.length ? `${lane.bands.length} band${lane.bands.length === 1 ? "" : "s"}` : "no bands";
      head.append(num, name, count);
      card.appendChild(head);

      const calibrated = this.image.lanes.some((l) => l.bands.some((b) => b.estimated_kda != null));
      if (lane.bands.length) {
        const list = document.createElement("ol");
        list.className = "band-rows" + (calibrated ? " with-kda" : "");
        lane.bands.forEach((band, i) => {
          const li = document.createElement("li");
          li.className = "band-row" + (band.low_confidence ? " unsure" : "");
          li.dataset.bandId = band.id;
          li.classList.toggle("selected", String(band.id) === String(this.selectedBandId));
          const share = band.percent_of_lane ?? 0;
          const kda = lane.is_ladder ? band.known_kda : band.estimated_kda;
          li.innerHTML = `
            <span class="band-idx">${i + 1}</span>
            <span class="bar"><i style="width:${Math.min(100, share)}%"></i></span>
            <span class="band-pct">${share.toFixed(1)}%</span>
            <span class="band-int" title="Background-corrected intensity">${band.intensity != null ? formatNumber(band.intensity) : ""}</span>
            ${calibrated ? `<span class="band-kda">${kda != null ? `${kda.toFixed(1)} kDa` : ""}</span>` : ""}`;
          li.addEventListener("click", () => this.select(band.id));
          li.addEventListener("mouseenter", () => this._highlight(band.id, true));
          li.addEventListener("mouseleave", () => this._highlight(band.id, false));
          list.appendChild(li);
        });
        card.appendChild(list);
      }
      this.lanesListEl.appendChild(card);
    }
  }

  _highlight(bandId, on) {
    const rect = this.svg.querySelector(`rect.band-box[data-band-id="${bandId}"]`);
    if (rect) rect.classList.toggle("hovered", on);
  }

  _renderCalibration() {
    const ladder = this.image.lanes.find((l) => l.is_ladder);
    this.ladderSelect.innerHTML = "";
    this.ladderSelect.add(new Option("None", ""));
    for (const lane of this.image.lanes) {
      const name = lane.label ? ` – ${lane.label}` : "";
      this.ladderSelect.add(new Option(`Lane ${lane.index + 1}${name}`, lane.id, false, lane.is_ladder));
    }

    this.ladderBandsEl.innerHTML = "";
    this.applyCalibrationBtn.hidden = !ladder;
    if (!ladder) return;
    if (!ladder.bands.length) {
      this.ladderBandsEl.textContent = "This lane has no bands yet. Add them on the image first.";
      return;
    }
    ladder.bands.forEach((band, i) => {
      const row = document.createElement("label");
      row.className = "ladder-band-row";
      row.textContent = `Band ${i + 1}`;
      const input = document.createElement("input");
      input.type = "number";
      input.placeholder = "kDa";
      input.dataset.bandId = band.id;
      input.value = band.known_kda ?? "";
      row.appendChild(input);
      this.ladderBandsEl.appendChild(row);
    });
  }
}

const panels = [];

function show(panel) {
  activePanel = panel;
  for (const p of panels) p.el.hidden = p !== panel;
  renderTabs();
}

function renderTabs() {
  tabsEl.hidden = panels.length < 2;
  if (tabsEl.hidden) return;
  tabsEl.innerHTML = "";
  for (const panel of panels) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "gel-tab" + (panel === activePanel ? " active" : "");
    const thumb = document.createElement("img");
    thumb.src = panel.image.url;
    thumb.alt = "";
    const label = document.createElement("span");
    label.className = "gel-tab-label";
    const name = document.createElement("span");
    name.className = "gel-tab-name";
    name.textContent = panel.image.orig_filename;
    const count = document.createElement("small");
    const n = panel.bandCount();
    count.textContent = `${n} band${n === 1 ? "" : "s"}`;
    label.append(name, count);
    tab.append(thumb, label);
    tab.addEventListener("click", () => show(panel));
    tabsEl.appendChild(tab);
  }
}

async function init() {
  try {
    const project = await api(`/api/projects/${projectId}`);
    for (const image of project.images) panels.push(new GelPanel(image));
    if (panels.length) show(panels[0]);
  } catch (err) {
    const msg = document.createElement("p");
    msg.className = "load-error";
    msg.textContent = err.message;
    workspace.appendChild(msg);
  }
}

init();
