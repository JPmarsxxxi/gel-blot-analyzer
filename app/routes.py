import io
import os
import uuid
import zipfile

from flask import Blueprint, jsonify, request, send_file, send_from_directory, render_template

from app.config import ALLOWED_EXTENSIONS, UPLOADS_DIR
from app.detection import pipeline
from app.detection.bands import recompute_percent_of_lane
from app.detection.calibration import fit_calibration
from app.detection.lanes import LaneBoundary
from app.export import build_csv, build_pdf, render_annotated_image
from app.models.db import session_scope
from app.models.schema import Band, GelImage, Lane, Project
from app.serializers import serialize_image, serialize_lane, serialize_project

bp = Blueprint("api", __name__)


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _allowed(filename: str) -> bool:
    return _ext(filename) in ALLOWED_EXTENSIONS


def _validate_is_image(path: str) -> bool:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ---------------------------------------------------------------- projects --

@bp.route("/api/projects", methods=["POST"])
def create_project():
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return _error("No images provided.")

    for f in files:
        if not _allowed(f.filename):
            return _error(
                f"Unsupported file type: '{f.filename}'. Accepted: PNG, JPG/JPEG, TIFF.", 415
            )

    with session_scope() as session:
        project = Project()
        session.add(project)
        session.flush()

        project_dir = os.path.join(UPLOADS_DIR, project.id)
        os.makedirs(project_dir, exist_ok=True)

        # Pass 1: save + validate every file is a real, openable image *before*
        # any detection processing runs on any of them.
        saved: list[tuple[str, str, str]] = []  # (orig_filename, stored_name, stored_path)
        for f in files:
            ext = _ext(f.filename)
            stored_name = f"{uuid.uuid4().hex}.{ext}"
            stored_path = os.path.join(project_dir, stored_name)
            f.save(stored_path)
            saved.append((f.filename, stored_name, stored_path))

            if not _validate_is_image(stored_path):
                for _, _, p in saved:
                    if os.path.exists(p):
                        os.remove(p)
                session.delete(project)
                return _error(f"'{f.filename}' is not a valid image file.", 415)

            # Browsers can't display TIFF, and many are 16-bit; store an 8-bit PNG instead.
            if ext in ("tif", "tiff"):
                from app.detection.imaging import load_rgb, save_rgb

                png_name = f"{os.path.splitext(stored_name)[0]}.png"
                png_path = os.path.join(project_dir, png_name)
                save_rgb(load_rgb(stored_path), png_path)
                os.remove(stored_path)
                saved[-1] = (f.filename, png_name, png_path)

        # Pass 2: all files validated -- now run detection for each.
        for order_index, (orig_filename, stored_name, stored_path) in enumerate(saved):
            from PIL import Image as PILImage

            with PILImage.open(stored_path) as im:
                width, height = im.size

            gel_image = GelImage(
                project_id=project.id,
                order_index=order_index,
                orig_filename=orig_filename,
                stored_filename=stored_name,
                source_filename=stored_name,
                width=width,
                height=height,
            )
            session.add(gel_image)
            session.flush()

            result = pipeline.run_full_detection(stored_path, sensitivity=gel_image.sensitivity)
            gel_image.is_gel_like = result.is_gel_like

            for lane_idx, (lane_boundary, bands) in enumerate(zip(result.lanes, result.bands_by_lane)):
                lane = Lane(
                    index=lane_idx,
                    x_start=lane_boundary.x_start,
                    x_end=lane_boundary.x_end,
                )
                gel_image.lanes.append(lane)
                session.flush()
                for band in bands:
                    lane.bands.append(
                        Band(
                            x=band.x,
                            y=band.y,
                            width=band.width,
                            height=band.height,
                            intensity=band.intensity,
                            percent_of_lane=band.percent_of_lane,
                            confidence=band.confidence,
                            low_confidence=band.low_confidence,
                        )
                    )

        project_id = project.id

    return jsonify({"project_id": project_id}), 201


@bp.route("/api/projects/<project_id>", methods=["GET"])
def get_project(project_id):
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return _error("Project not found.", 404)
        return jsonify(serialize_project(project))


# ------------------------------------------------------------------ images --

def _image_path(image: GelImage) -> str:
    return os.path.join(UPLOADS_DIR, image.project_id, image.stored_filename)


def _source_path(image: GelImage) -> str:
    return os.path.join(UPLOADS_DIR, image.project_id, image.source_filename)


@bp.route("/api/images/<int:image_id>", methods=["PATCH"])
def update_image_adjustments(image_id):
    body = request.get_json(force=True, silent=True) or {}
    with session_scope() as session:
        image = session.get(GelImage, image_id)
        if image is None:
            return _error("Image not found.", 404)

        # Adjustments apply incrementally to the currently-displayed image (the
        # crop rectangle, in particular, is drawn by the user against that
        # current image's pixel space, not the pristine original's).
        crop_x = body.get("crop_x", 0.0)
        crop_y = body.get("crop_y", 0.0)
        crop_w = body.get("crop_w")
        crop_h = body.get("crop_h")
        rotate_deg = body.get("rotate_deg", 0.0)
        brightness = body.get("brightness", 0.0)
        contrast = body.get("contrast", 1.0)

        old_stored_path = _image_path(image)
        project_dir = os.path.join(UPLOADS_DIR, image.project_id)
        new_stored_name = f"{uuid.uuid4().hex}.png"
        new_stored_path = os.path.join(project_dir, new_stored_name)

        width, height = pipeline.apply_image_adjustments(
            old_stored_path,
            new_stored_path,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            rotate_deg=rotate_deg,
            brightness=brightness,
            contrast=contrast,
        )

        # Stored fields record the *last-applied* increment (for display/debug),
        # not a cumulative transform to replay -- sliders reset to neutral below.
        image.crop_x, image.crop_y, image.crop_w, image.crop_h = crop_x, crop_y, crop_w, crop_h
        image.rotate_deg, image.brightness, image.contrast = 0.0, 0.0, 1.0

        image.stored_filename = new_stored_name
        image.width = width
        image.height = height

        result = pipeline.run_full_detection(new_stored_path, sensitivity=image.sensitivity)
        image.is_gel_like = result.is_gel_like

        image.lanes.clear()
        session.flush()

        for lane_idx, (lane_boundary, bands) in enumerate(zip(result.lanes, result.bands_by_lane)):
            lane = Lane(
                index=lane_idx,
                x_start=lane_boundary.x_start,
                x_end=lane_boundary.x_end,
            )
            image.lanes.append(lane)
            session.flush()
            for band in bands:
                lane.bands.append(
                    Band(
                        x=band.x,
                        y=band.y,
                        width=band.width,
                        height=band.height,
                        intensity=band.intensity,
                        percent_of_lane=band.percent_of_lane,
                        confidence=band.confidence,
                        low_confidence=band.low_confidence,
                    )
                )

        session.flush()
        response = serialize_image(image)

    if old_stored_path != _source_path(image) and os.path.exists(old_stored_path):
        os.remove(old_stored_path)

    return jsonify(response)


@bp.route("/api/images/<int:image_id>/sensitivity", methods=["PATCH"])
def update_sensitivity(image_id):
    body = request.get_json(force=True, silent=True) or {}
    if "sensitivity" not in body:
        return _error("Missing 'sensitivity'.")

    with session_scope() as session:
        image = session.get(GelImage, image_id)
        if image is None:
            return _error("Image not found.", 404)

        image.sensitivity = float(body["sensitivity"])
        lanes = list(image.lanes)
        lane_boundaries = [LaneBoundary(x_start=l.x_start, x_end=l.x_end) for l in lanes]

        bands_by_lane = pipeline.run_band_redetection(
            _image_path(image), lane_boundaries, image.sensitivity
        )

        for lane, bands in zip(lanes, bands_by_lane):
            lane.bands.clear()
            session.flush()
            for band in bands:
                lane.bands.append(
                    Band(
                        x=band.x,
                        y=band.y,
                        width=band.width,
                        height=band.height,
                        intensity=band.intensity,
                        percent_of_lane=band.percent_of_lane,
                        confidence=band.confidence,
                        low_confidence=band.low_confidence,
                    )
                )

        session.flush()
        return jsonify(serialize_image(image))


# ------------------------------------------------------------------- lanes --

@bp.route("/api/lanes/<int:lane_id>", methods=["PATCH"])
def update_lane(lane_id):
    body = request.get_json(force=True, silent=True) or {}
    with session_scope() as session:
        lane = session.get(Lane, lane_id)
        if lane is None:
            return _error("Lane not found.", 404)

        if "x_start" in body:
            lane.x_start = float(body["x_start"])
        if "x_end" in body:
            lane.x_end = float(body["x_end"])
        if "label" in body:
            lane.label = str(body["label"])
        if "is_ladder" in body:
            is_ladder = bool(body["is_ladder"])
            if is_ladder:
                for other in lane.image.lanes:
                    if other.id != lane.id:
                        other.is_ladder = False
            lane.is_ladder = is_ladder

        session.flush()
        return jsonify(serialize_lane(lane))


@bp.route("/api/lanes/<int:lane_id>/bands", methods=["POST"])
def add_band(lane_id):
    body = request.get_json(force=True, silent=True) or {}
    for field in ("x", "y", "width", "height"):
        if field not in body:
            return _error(f"Missing '{field}'.")

    with session_scope() as session:
        lane = session.get(Lane, lane_id)
        if lane is None:
            return _error("Lane not found.", 404)

        image = lane.image
        signal = pipeline.get_signal(_image_path(image))
        x, y, w, h = int(body["x"]), int(body["y"]), int(body["width"]), int(body["height"])
        box_signal = signal[max(0, y) : y + h, max(0, x) : x + w]
        intensity = float(box_signal.sum()) if box_signal.size else 0.0

        band = Band(
            x=float(x),
            y=float(y),
            width=float(w),
            height=float(h),
            intensity=intensity,
            confidence=1.0,
            low_confidence=False,
            manually_edited=True,
        )
        lane.bands.append(band)
        session.flush()

        _refresh_lane_percentages(lane)
        session.flush()
        return jsonify(serialize_lane(lane)), 201


def _refresh_lane_percentages(lane: Lane) -> None:
    intensities = [b.intensity or 0.0 for b in lane.bands]
    percents = recompute_percent_of_lane(intensities)
    for band, pct in zip(lane.bands, percents):
        band.percent_of_lane = pct


# ------------------------------------------------------------------- bands --

@bp.route("/api/bands/<int:band_id>", methods=["PATCH"])
def update_band(band_id):
    body = request.get_json(force=True, silent=True) or {}
    with session_scope() as session:
        band = session.get(Band, band_id)
        if band is None:
            return _error("Band not found.", 404)

        lane = band.lane
        geometry_changed = False
        for field in ("x", "y", "width", "height"):
            if field in body:
                setattr(band, field, float(body[field]))
                geometry_changed = True

        if "known_kda" in body:
            band.known_kda = float(body["known_kda"]) if body["known_kda"] is not None else None

        if geometry_changed:
            band.manually_edited = True
            image = lane.image
            signal = pipeline.get_signal(_image_path(image))
            x0, y0 = int(band.x), int(band.y)
            x1, y1 = int(band.x + band.width), int(band.y + band.height)
            box_signal = signal[max(0, y0) : y1, max(0, x0) : x1]
            band.intensity = float(box_signal.sum()) if box_signal.size else 0.0
            _refresh_lane_percentages(lane)

        session.flush()
        return jsonify(serialize_lane(lane))


@bp.route("/api/bands/<int:band_id>", methods=["DELETE"])
def delete_band(band_id):
    with session_scope() as session:
        band = session.get(Band, band_id)
        if band is None:
            return _error("Band not found.", 404)
        lane = band.lane
        session.delete(band)
        session.flush()
        _refresh_lane_percentages(lane)
        session.flush()
        return jsonify(serialize_lane(lane))


# ------------------------------------------------------------- calibration --

@bp.route("/api/images/<int:image_id>/calibrate", methods=["POST"])
def calibrate_ladder(image_id):
    with session_scope() as session:
        image = session.get(GelImage, image_id)
        if image is None:
            return _error("Image not found.", 404)

        ladder_lane = next((l for l in image.lanes if l.is_ladder), None)
        if ladder_lane is None:
            return _error("No lane marked as ladder.", 400)

        points = [(b.y + b.height / 2, b.known_kda) for b in ladder_lane.bands]
        curve = fit_calibration(points)

        for lane in image.lanes:
            if lane.id == ladder_lane.id:
                continue
            for band in lane.bands:
                band.estimated_kda = curve.estimate_kda(band.y + band.height / 2) if curve else None

        session.flush()
        return jsonify(serialize_image(image))


# ------------------------------------------------------------------ export --

@bp.route("/api/projects/<project_id>/export/csv", methods=["GET"])
def export_csv(project_id):
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return _error("Project not found.", 404)
        data = build_csv(project)

    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"gel_analysis_{project_id[:8]}.csv",
    )


@bp.route("/api/projects/<project_id>/export/png", methods=["GET"])
def export_png(project_id):
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return _error("Project not found.", 404)

        images = list(project.images)
        if len(images) == 1:
            annotated = render_annotated_image(images[0])
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            return send_file(
                buf, mimetype="image/png", as_attachment=True,
                download_name=f"{os.path.splitext(images[0].orig_filename)[0]}_annotated.png",
            )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for image in images:
                annotated = render_annotated_image(image)
                img_buf = io.BytesIO()
                annotated.save(img_buf, format="PNG")
                name = f"{os.path.splitext(image.orig_filename)[0]}_annotated.png"
                zf.writestr(name, img_buf.getvalue())
        zip_buf.seek(0)
        return send_file(
            zip_buf, mimetype="application/zip", as_attachment=True,
            download_name=f"gel_analysis_{project_id[:8]}_images.zip",
        )


@bp.route("/api/projects/<project_id>/export/pdf", methods=["GET"])
def export_pdf(project_id):
    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return _error("Project not found.", 404)
        data = build_pdf(project)

    return send_file(
        io.BytesIO(data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"gel_analysis_{project_id[:8]}.pdf",
    )


# -------------------------------------------------------------------- misc --

@bp.route("/storage/uploads/<project_id>/<path:filename>")
def serve_upload(project_id, filename):
    return send_from_directory(os.path.join(UPLOADS_DIR, project_id), filename)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/p/<project_id>")
def project_view(project_id):
    return render_template("project.html", project_id=project_id)
