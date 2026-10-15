import glob
import io
import os

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import inspect, text

from app.config import BASE_DIR, UPLOADS_DIR
from app.detection.membrane import find_membrane
from app.models.db import _engine, init_db
from tests.conftest import generate_sample, synth_png_bytes

MEMBRANE = (900, 600, 220, 400)  # x, y, w, h inside a 2125 x 1700 frame, like a ChemiDoc export


def chemidoc_frame() -> np.ndarray:
    frame = np.full((1700, 2125), 124, np.uint8)
    x, y, w, h = MEMBRANE
    frame[y : y + h, x : x + w] = 200
    for lane_x in (x + 40, x + 130):
        frame[y + 150 : y + 165, lane_x : lane_x + 60] = 60
    return frame


def png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def upload(client, data: bytes, name="export.png"):
    resp = client.post("/api/projects", data={"images": (io.BytesIO(data), name)}, content_type="multipart/form-data")
    assert resp.status_code == 201, resp.get_json()
    return client.get(f"/api/projects/{resp.get_json()['project_id']}").get_json()


def test_finds_membrane_in_a_uniform_frame():
    x, y, w, h = find_membrane(chemidoc_frame())
    mx, my, mw, mh = MEMBRANE
    assert x <= mx and y <= my and x + w >= mx + mw and y + h >= my + mh
    assert w < mw * 1.1 and h < mh * 1.1


def test_leaves_gels_blots_and_tiny_images_alone():
    for seed in range(1, 6):
        assert find_membrane(generate_sample(seed=seed).image) is None
    assert find_membrane(np.full((50, 50), 128, np.uint8)) is None
    page = np.full((300, 800), 250, np.uint8)
    page[140:160, 100:200] = 30  # one dark band on a white strip
    assert find_membrane(page) is None


def test_real_chemidoc_export_is_cropped_if_available():
    paths = glob.glob(os.path.join("C:/", "Users", "User", "western_blots", "zenodo_21330132", "data", "*", "*", "*(merge).tif"))
    if not paths:
        pytest.skip("raw ChemiDoc exports not downloaded")
    from app.detection.imaging import load_rgb, to_grayscale

    assert find_membrane(to_grayscale(load_rgb(sorted(paths)[0]))) is not None


def test_upload_auto_crops_keeps_original_and_undo_restores(client):
    project = upload(client, png_bytes(chemidoc_frame()))
    image = project["images"][0]
    assert image["auto_cropped"] is True
    assert image["width"] < 300 and image["height"] < 450
    assert sum(len(l["bands"]) for l in image["lanes"]) >= 1
    folder = os.path.join(UPLOADS_DIR, project["id"])
    assert len(os.listdir(folder)) == 2

    resp = client.post(f"/api/images/{image['id']}/undo-crop")
    assert resp.status_code == 200
    undone = resp.get_json()
    assert undone["auto_cropped"] is False
    assert (undone["width"], undone["height"]) == (2125, 1700)
    assert undone["crop_w"] is None
    assert len(os.listdir(folder)) == 1
    assert client.post(f"/api/images/{image['id']}/undo-crop").status_code == 400


def test_ordinary_gel_is_not_cropped(client):
    data = synth_png_bytes(3)
    image = upload(client, data, "gel.png")["images"][0]
    assert image["auto_cropped"] is False
    assert (image["width"], image["height"]) == Image.open(io.BytesIO(data)).size
    assert client.post(f"/api/images/{image['id']}/undo-crop").status_code == 400


def test_manual_adjustment_withdraws_undo(client):
    image = upload(client, png_bytes(chemidoc_frame()))["images"][0]
    resp = client.patch(f"/api/images/{image['id']}", json={"brightness": 5})
    assert resp.status_code == 200 and resp.get_json()["auto_cropped"] is False


def test_init_db_adds_missing_column_once(client):
    upload(client, synth_png_bytes(2), "gel.png")
    with _engine.begin() as conn:
        conn.execute(text("ALTER TABLE gel_images DROP COLUMN auto_cropped"))
    init_db()
    init_db()
    assert "auto_cropped" in {c["name"] for c in inspect(_engine).get_columns("gel_images")}
    with _engine.connect() as conn:
        assert {v for (v,) in conn.execute(text("select auto_cropped from gel_images"))} == {0}


def test_project_page_has_undo_crop_notice():
    page = open(os.path.join(BASE_DIR, "app", "templates", "project.html"), encoding="utf-8").read()
    script = open(os.path.join(BASE_DIR, "app", "static", "js", "project.js"), encoding="utf-8").read()
    assert 'class="crop-notice" hidden' in page and "undo-crop-btn" in page
    assert "this.cropNoticeEl.hidden = !img.auto_cropped" in script
    assert "/undo-crop`" in script
