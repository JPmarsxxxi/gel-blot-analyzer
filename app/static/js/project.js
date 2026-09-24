const SVG_NS = "http://www.w3.org/2000/svg";
const projectId = document.body.dataset.projectId;

document.getElementById("export-csv").href = `/api/projects/${projectId}/export/csv`;
document.getElementById("export-png").href = `/api/projects/${projectId}/export/png`;
document.getElementById("export-pdf").href = `/api/projects/${projectId}/export/pdf`;

const container = document.getElementById("images-container");
const panelTemplate = document.getElementById("image-panel-template");

async function api(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "Request failed.");
  return data;
}

function screenToImageCoords(svg, evt) {
  const pt = svg.createSVGPoint();
  pt.x = evt.clientX;
  pt.y = evt.clientY;
  const ctm = svg.getScreenCTM().inverse();
  const p = pt.matrixTransform(ctm);
  return { x: p.x, y: p.y };
}

function findLaneForX(image, x) {
  return image.lanes.find((l) => x >= l.x_start && x <= l.x_end);
}

class GelPanel {
  constructor(image) {
    this.image = image;
    this.mode = "select"; // "select" | "add-band" | "crop"
    this.drag = null;
    this.cropRect = null;

    const node = panelTemplate.content.cloneNode(true);
    this.el = node.querySelector(".gel-panel");
    this.imgEl = this.el.querySelector(".gel-image");
    this.svg = this.el.querySelector(".gel-overlay");
    this.laneLabelsEl = this.el.querySelector(".lane-labels");
    this.tableBody = this.el.querySelector(".bands-table tbody");
    this.ladderPanel = this.el.querySelector(".ladder-panel");
    this.ladderBandsEl = this.el.querySelector(".ladder-bands");
    this.warningEl = this.el.querySelector(".gel-warning");

    this.el.querySelector(".image-filename").textContent = image.orig_filename;

    this._wireAdjustmentControls();
    this._wireSvgEvents();

    container.appendChild(this.el);
    this.render();
  }

  // ---------------------------------------------------------- adjustments --

  _wireAdjustmentControls() {
    const rotate = this.el.querySelector(".adj-rotate");
    const brightness = this.el.querySelector(".adj-brightness");
    const contrast = this.el.querySelector(".adj-contrast");
    const sensitivity = this.el.querySelector(".adj-sensitivity");
    const cropBtn = this.el.querySelector(".crop-toggle-btn");
    const applyBtn = this.el.querySelector(".apply-adjustments-btn");
    const addBandBtn = this.el.querySelector(".add-band-toggle-btn");
    const applyCalibrationBtn = this.el.querySelector(".apply-calibration-btn");

    rotate.value = this.image.rotate_deg;
    brightness.value = this.image.brightness;
    contrast.value = this.image.contrast;
    sensitivity.value = this.image.sensitivity;

    // Live CSS preview only; real pixel adjustment happens server-side on Apply.
    const previewFilter = () => {
      this.imgEl.style.filter = `brightness(${1 + brightness.value / 150}) contrast(${contrast.value})`;
    };
    brightness.addEventListener("input", previewFilter);
    contrast.addEventListener("input", previewFilter);

    cropBtn.addEventListener("click", () => {
      this.mode = this.mode === "crop" ? "select" : "crop";
      cropBtn.classList.toggle("active", this.mode === "crop");
      addBandBtn.classList.remove("active");
      if (this.mode !== "crop") {
        this.cropRect = null;
        this._clearCropRect();
      }
    });

    addBandBtn.addEventListener("click", () => {
      this.mode = this.mode === "add-band" ? "select" : "add-band";
      addBandBtn.classList.toggle("active", this.mode === "add-band");
      cropBtn.classList.remove("active");
    });

    applyBtn.addEventListener("click", async () => {
      applyBtn.disabled = true;
      const body = {
        rotate_deg: parseFloat(rotate.value) || 0,
        brightness: parseFloat(brightness.value) || 0,
        contrast: parseFloat(contrast.value) || 1,
      };
      if (this.cropRect) {
        body.crop_x = this.cropRect.x;
        body.crop_y = this.cropRect.y;
        body.crop_w = this.cropRect.w;
        body.crop_h = this.cropRect.h;
      }
      try {
        const updated = await api(`/api/images/${this.image.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        this.image = updated;
        this.cropRect = null;
        this.mode = "select";
        cropBtn.classList.remove("active");
        this.imgEl.style.filter = "";
        // Adjustments apply incrementally server-side; sliders reset to neutral
        // for the next round rather than showing a cumulative value.
        rotate.value = 0;
        brightness.value = 0;
        contrast.value = 1;
        this.render();
      } catch (err) {
        alert(err.message);
      } finally {
        applyBtn.disabled = false;
      }
    });

    let sensitivityTimer = null;
    sensitivity.addEventListener("input", () => {
      clearTimeout(sensitivityTimer);
      sensitivityTimer = setTimeout(async () => {
        try {
          const updated = await api(`/api/images/${this.image.id}/sensitivity`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sensitivity: parseFloat(sensitivity.value) }),
          });
          this.image = updated;
          this.render();
        } catch (err) {
          alert(err.message);
        }
      }, 300);
    });

    applyCalibrationBtn.addEventListener("click", async () => {
      const ladderLane = this.image.lanes.find((l) => l.is_ladder);
      if (!ladderLane) return;
      for (const row of this.ladderBandsEl.querySelectorAll(".ladder-band-row")) {
        const bandId = row.dataset.bandId;
        const val = row.querySelector("input").value;
        await api(`/api/bands/${bandId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ known_kda: val === "" ? null : parseFloat(val) }),
        });
      }
      const updated = await api(`/api/images/${this.image.id}/calibrate`, { method: "POST" });
      this.image = updated;
      this.render();
    });
  }

  // -------------------------------------------------------------- events --

  _wireSvgEvents() {
    this.svg.addEventListener("mousedown", (e) => this._onMouseDown(e));
    document.addEventListener("mousemove", (e) => this._onMouseMove(e));
    document.addEventListener("mouseup", (e) => this._onMouseUp(e));
  }

  _onMouseDown(e) {
    const target = e.target;
    const pt = screenToImageCoords(this.svg, e);

    if (this.mode === "crop") {
      this.drag = { type: "crop-draw", start: pt };
      return;
    }

    if (this.mode === "add-band") {
      const lane = findLaneForX(this.image, pt.x);
      if (!lane) return;
      const laneWidth = lane.x_end - lane.x_start;
      const h = Math.max(10, this.image.height * 0.03);
      const w = laneWidth * 0.6;
      this._createBand(lane.id, {
        x: pt.x - w / 2,
        y: pt.y - h / 2,
        width: w,
        height: h,
      });
      return;
    }

    if (target.classList.contains("band-delete-hit")) {
      e.stopPropagation();
      this._deleteBand(target.dataset.bandId);
      return;
    }
    if (target.classList.contains("band-handle")) {
      const band = this._findBand(target.dataset.bandId);
      this.drag = { type: "resize", band, start: pt, origW: band.width, origH: band.height };
      return;
    }
    if (target.classList.contains("band-box")) {
      const band = this._findBand(target.dataset.bandId);
      this.drag = { type: "move", band, start: pt, origX: band.x, origY: band.y };
      return;
    }
    if (target.classList.contains("lane-divider")) {
      const leftId = target.dataset.leftLaneId;
      const rightId = target.dataset.rightLaneId;
      this.drag = { type: "divider", leftId, rightId, startX: pt.x };
    }
  }

  _onMouseMove(e) {
    if (!this.drag) return;
    const pt = screenToImageCoords(this.svg, e);

    if (this.drag.type === "crop-draw") {
      const x = Math.min(this.drag.start.x, pt.x);
      const y = Math.min(this.drag.start.y, pt.y);
      const w = Math.abs(pt.x - this.drag.start.x);
      const h = Math.abs(pt.y - this.drag.start.y);
      this.cropRect = { x, y, w, h };
      this._drawCropRect();
      return;
    }

    if (this.drag.type === "move") {
      const dx = pt.x - this.drag.start.x;
      const dy = pt.y - this.drag.start.y;
      const band = this.drag.band;
      const rect = this.svg.querySelector(`rect.band-box[data-band-id="${band.id}"]`);
      const newX = this.drag.origX + dx;
      const newY = this.drag.origY + dy;
      if (rect) {
        rect.setAttribute("x", newX);
        rect.setAttribute("y", newY);
      }
      this._moveHandles(band.id, newX, newY, band.width, band.height);
      this.drag.newX = newX;
      this.drag.newY = newY;
      return;
    }

    if (this.drag.type === "resize") {
      const dx = pt.x - this.drag.start.x;
      const dy = pt.y - this.drag.start.y;
      const band = this.drag.band;
      const newW = Math.max(5, this.drag.origW + dx);
      const newH = Math.max(5, this.drag.origH + dy);
      const rect = this.svg.querySelector(`rect.band-box[data-band-id="${band.id}"]`);
      if (rect) {
        rect.setAttribute("width", newW);
        rect.setAttribute("height", newH);
      }
      this._moveHandles(band.id, band.x, band.y, newW, newH);
      this.drag.newW = newW;
      this.drag.newH = newH;
      return;
    }

    if (this.drag.type === "divider") {
      const line = this.svg.querySelector(
        `line.lane-divider[data-left-lane-id="${this.drag.leftId}"]`
      );
      if (line) {
        line.setAttribute("x1", pt.x);
        line.setAttribute("x2", pt.x);
      }
      this.drag.newX = pt.x;
    }
  }

  async _onMouseUp() {
    if (!this.drag) return;
    const drag = this.drag;
    this.drag = null;

    try {
      if (drag.type === "crop-draw") {
        return; // committed on "Apply adjustments"
      }
      if (drag.type === "move" && drag.newX !== undefined) {
        const updatedLane = await api(`/api/bands/${drag.band.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x: drag.newX, y: drag.newY }),
        });
        this._patchLane(updatedLane);
        this.render();
      } else if (drag.type === "resize" && drag.newW !== undefined) {
        const updatedLane = await api(`/api/bands/${drag.band.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ width: drag.newW, height: drag.newH }),
        });
        this._patchLane(updatedLane);
        this.render();
      } else if (drag.type === "divider" && drag.newX !== undefined) {
        await api(`/api/lanes/${drag.leftId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x_end: drag.newX }),
        });
        await api(`/api/lanes/${drag.rightId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ x_start: drag.newX }),
        });
        const lane = this.image.lanes.find((l) => l.id == drag.leftId);
        const lane2 = this.image.lanes.find((l) => l.id == drag.rightId);
        if (lane) lane.x_end = drag.newX;
        if (lane2) lane2.x_start = drag.newX;
        this.render();
      }
    } catch (err) {
      alert(err.message);
      this.render();
    }
  }

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

  async _createBand(laneId, box) {
    try {
      const updatedLane = await api(`/api/lanes/${laneId}/bands`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(box),
      });
      this._patchLane(updatedLane);
      this.render();
    } catch (err) {
      alert(err.message);
    }
  }

  async _deleteBand(bandId) {
    try {
      const updatedLane = await api(`/api/bands/${bandId}`, { method: "DELETE" });
      this._patchLane(updatedLane);
      this.render();
    } catch (err) {
      alert(err.message);
    }
  }

  _moveHandles(bandId, x, y, w, h) {
    const handle = this.svg.querySelector(`circle.band-handle[data-band-id="${bandId}"]`);
    if (handle) {
      handle.setAttribute("cx", x + w);
      handle.setAttribute("cy", y + h);
    }
    const del = this.svg.querySelector(`g.band-delete[data-band-id="${bandId}"]`);
    if (del) {
      del.setAttribute("transform", `translate(${x + w}, ${y})`);
    }
  }

  _clearCropRect() {
    const existing = this.svg.querySelector("rect.crop-rect");
    if (existing) existing.remove();
  }

  _drawCropRect() {
    this._clearCropRect();
    if (!this.cropRect) return;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("class", "crop-rect");
    rect.setAttribute("x", this.cropRect.x);
    rect.setAttribute("y", this.cropRect.y);
    rect.setAttribute("width", this.cropRect.w);
    rect.setAttribute("height", this.cropRect.h);
    this.svg.appendChild(rect);
  }

  // -------------------------------------------------------------- render --

  render() {
    const img = this.image;
    this.imgEl.src = img.url;
    this.svg.setAttribute("viewBox", `0 0 ${img.width} ${img.height}`);
    this.warningEl.hidden = img.is_gel_like;

    this._renderOverlay();
    this._renderLaneLabels();
    this._renderLadderPanel();
    this._renderTable();
  }

  _renderOverlay() {
    this.svg.innerHTML = "";
    const img = this.image;

    for (let i = 0; i < img.lanes.length - 1; i++) {
      const left = img.lanes[i];
      const right = img.lanes[i + 1];
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("class", "lane-divider");
      line.setAttribute("x1", left.x_end);
      line.setAttribute("x2", left.x_end);
      line.setAttribute("y1", 0);
      line.setAttribute("y2", img.height);
      line.dataset.leftLaneId = left.id;
      line.dataset.rightLaneId = right.id;
      this.svg.appendChild(line);
    }

    for (const lane of img.lanes) {
      for (const band of lane.bands) {
        const rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("class", "band-box" + (band.low_confidence ? " low-confidence" : ""));
        rect.setAttribute("x", band.x);
        rect.setAttribute("y", band.y);
        rect.setAttribute("width", band.width);
        rect.setAttribute("height", band.height);
        rect.dataset.bandId = band.id;
        this.svg.appendChild(rect);

        const handle = document.createElementNS(SVG_NS, "circle");
        handle.setAttribute("class", "band-handle");
        handle.setAttribute("cx", band.x + band.width);
        handle.setAttribute("cy", band.y + band.height);
        handle.setAttribute("r", Math.max(3, this.image.width * 0.004));
        handle.dataset.bandId = band.id;
        this.svg.appendChild(handle);

        const del = document.createElementNS(SVG_NS, "g");
        del.setAttribute("class", "band-delete");
        del.dataset.bandId = band.id;
        del.setAttribute("transform", `translate(${band.x + band.width}, ${band.y})`);
        const r = Math.max(6, this.image.width * 0.008);
        const circle = document.createElementNS(SVG_NS, "circle");
        circle.setAttribute("r", r);
        circle.setAttribute("class", "band-delete-hit");
        circle.dataset.bandId = band.id;
        const text = document.createElementNS(SVG_NS, "text");
        text.textContent = "×";
        text.setAttribute("font-size", r * 1.6);
        text.dataset.bandId = band.id;
        del.appendChild(circle);
        del.appendChild(text);
        this.svg.appendChild(del);
      }
    }

    this._drawCropRect();
  }

  _renderLaneLabels() {
    this.laneLabelsEl.innerHTML = "";
    const img = this.image;
    const displayWidth = this.imgEl.clientWidth || img.width;
    const scale = displayWidth / img.width;

    for (const lane of img.lanes) {
      const leftPx = lane.x_start * scale;
      const widthPx = (lane.x_end - lane.x_start) * scale;

      const input = document.createElement("input");
      input.className = "lane-label-input";
      input.placeholder = `Lane ${lane.index + 1}`;
      input.value = lane.label || "";
      input.style.left = `${leftPx}px`;
      input.style.width = `${Math.max(50, widthPx)}px`;
      let labelTimer = null;
      input.addEventListener("input", () => {
        clearTimeout(labelTimer);
        labelTimer = setTimeout(async () => {
          const updated = await api(`/api/lanes/${lane.id}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: input.value }),
          });
          this._patchLane(updated);
          this._renderTable();
        }, 400);
      });
      this.laneLabelsEl.appendChild(input);

      const ladderBtn = document.createElement("button");
      ladderBtn.type = "button";
      ladderBtn.className = "lane-ladder-btn" + (lane.is_ladder ? " active" : "");
      ladderBtn.textContent = lane.is_ladder ? "Ladder ✓" : "Set as ladder";
      ladderBtn.style.left = `${leftPx}px`;
      ladderBtn.addEventListener("click", async () => {
        const updated = await api(`/api/lanes/${lane.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ is_ladder: !lane.is_ladder }),
        });
        for (const l of this.image.lanes) l.is_ladder = false;
        this._patchLane(updated);
        this.render();
      });
      this.laneLabelsEl.appendChild(ladderBtn);
    }
  }

  _renderLadderPanel() {
    const ladderLane = this.image.lanes.find((l) => l.is_ladder);
    if (!ladderLane) {
      this.ladderPanel.hidden = true;
      return;
    }
    this.ladderPanel.hidden = false;
    this.ladderBandsEl.innerHTML = "";
    ladderLane.bands.forEach((band, i) => {
      const row = document.createElement("div");
      row.className = "ladder-band-row";
      row.dataset.bandId = band.id;
      const label = document.createElement("span");
      label.textContent = `Band ${i + 1}`;
      const input = document.createElement("input");
      input.type = "number";
      input.placeholder = "kDa";
      input.value = band.known_kda ?? "";
      row.appendChild(label);
      row.appendChild(input);
      this.ladderBandsEl.appendChild(row);
    });
  }

  _renderTable() {
    this.tableBody.innerHTML = "";
    for (const lane of this.image.lanes) {
      lane.bands.forEach((band, i) => {
        const tr = document.createElement("tr");
        if (band.low_confidence) tr.classList.add("low-confidence-row");
        const kda = lane.is_ladder ? band.known_kda : band.estimated_kda;
        tr.innerHTML = `
          <td>${lane.index + 1}</td>
          <td>${lane.label || ""}</td>
          <td>${i + 1}</td>
          <td>${band.intensity != null ? band.intensity.toFixed(1) : ""}</td>
          <td>${band.percent_of_lane != null ? band.percent_of_lane.toFixed(1) : ""}</td>
          <td>${kda != null ? kda.toFixed(1) : ""}</td>
        `;
        this.tableBody.appendChild(tr);
      });
    }
  }
}

async function init() {
  const project = await api(`/api/projects/${projectId}`);
  for (const image of project.images) {
    new GelPanel(image);
  }
}

init();
