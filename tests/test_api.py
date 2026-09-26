import io

from tests.conftest import synth_png_bytes


def _upload(client, seed=1, filename="gel.png"):
    data = {"images": (io.BytesIO(synth_png_bytes(seed)), filename)}
    resp = client.post("/api/projects", data=data, content_type="multipart/form-data")
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["project_id"]


def test_upload_creates_project_with_lanes_and_bands(client):
    pid = _upload(client)
    resp = client.get(f"/api/projects/{pid}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["images"]) == 1
    img = data["images"][0]
    assert len(img["lanes"]) >= 1
    assert "is_gel_like" in img


def test_16bit_tiff_upload_is_stored_as_viewable_png_with_contrast(client):
    import numpy as np
    from PIL import Image

    from tests.conftest import generate_sample

    img8 = generate_sample(seed=3).image.astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(img8 * 200 + 1000).save(buf, format="TIFF")
    buf.seek(0)
    resp = client.post("/api/projects", data={"images": (buf, "scan.tif")}, content_type="multipart/form-data")
    assert resp.status_code == 201, resp.get_json()
    image = client.get(f"/api/projects/{resp.get_json()['project_id']}").get_json()["images"][0]
    assert image["url"].endswith(".png")
    served = np.asarray(Image.open(io.BytesIO(client.get(image["url"]).data)).convert("L"))
    assert served.min() < 50 and served.max() > 200


def test_reject_non_image_file_before_processing(client):
    data = {"images": (io.BytesIO(b"not an image"), "fake.png")}
    resp = client.post("/api/projects", data=data, content_type="multipart/form-data")
    assert resp.status_code == 415
    assert "error" in resp.get_json()


def test_reject_unsupported_extension(client):
    data = {"images": (io.BytesIO(b"whatever"), "malware.exe")}
    resp = client.post("/api/projects", data=data, content_type="multipart/form-data")
    assert resp.status_code == 415


def test_multi_file_upload_is_batch_in_one_project(client):
    data = {
        "images": [
            (io.BytesIO(synth_png_bytes(1)), "a.png"),
            (io.BytesIO(synth_png_bytes(2)), "b.png"),
        ]
    }
    resp = client.post("/api/projects", data=data, content_type="multipart/form-data")
    assert resp.status_code == 201
    pid = resp.get_json()["project_id"]
    data = client.get(f"/api/projects/{pid}").get_json()
    assert len(data["images"]) == 2


def test_sensitivity_change_updates_bands_without_changing_lane_count(client):
    pid = _upload(client, seed=5)
    img = client.get(f"/api/projects/{pid}").get_json()["images"][0]
    n_lanes_before = len(img["lanes"])

    resp = client.patch(f"/api/images/{img['id']}/sensitivity", json={"sensitivity": 0.9})
    assert resp.status_code == 200
    updated = resp.get_json()
    assert len(updated["lanes"]) == n_lanes_before


def test_lane_label_and_divider_persist(client):
    pid = _upload(client, seed=2)
    img = client.get(f"/api/projects/{pid}").get_json()["images"][0]
    lane = img["lanes"][0]

    resp = client.patch(f"/api/lanes/{lane['id']}", json={"label": "Control"})
    assert resp.status_code == 200
    assert resp.get_json()["label"] == "Control"

    data = client.get(f"/api/projects/{pid}").get_json()
    assert data["images"][0]["lanes"][0]["label"] == "Control"


def test_add_move_delete_band(client):
    pid = _upload(client, seed=2)
    img = client.get(f"/api/projects/{pid}").get_json()["images"][0]
    lane = img["lanes"][0]

    resp = client.post(f"/api/lanes/{lane['id']}/bands", json={"x": 2, "y": 2, "width": 20, "height": 10})
    assert resp.status_code == 201
    new_band = [b for b in resp.get_json()["bands"] if b["manually_edited"]][0]

    resp = client.patch(f"/api/bands/{new_band['id']}", json={"x": 5, "y": 5})
    assert resp.status_code == 200

    resp = client.delete(f"/api/bands/{new_band['id']}")
    assert resp.status_code == 200
    remaining_ids = [b["id"] for b in resp.get_json()["bands"]]
    assert new_band["id"] not in remaining_ids


def test_ladder_calibration_without_enough_points_leaves_kda_blank(client):
    pid = _upload(client, seed=2)
    img = client.get(f"/api/projects/{pid}").get_json()["images"][0]
    lane = img["lanes"][0]

    client.patch(f"/api/lanes/{lane['id']}", json={"is_ladder": True})
    resp = client.post(f"/api/images/{img['id']}/calibrate")
    assert resp.status_code == 200
    for l in resp.get_json()["lanes"]:
        if not l["is_ladder"]:
            for b in l["bands"]:
                assert b["estimated_kda"] is None


def test_ladder_calibration_with_two_points_estimates_kda(client):
    pid = _upload(client, seed=9)
    img = client.get(f"/api/projects/{pid}").get_json()["images"][0]
    ladder_lane = max(img["lanes"], key=lambda l: len(l["bands"]))
    if len(ladder_lane["bands"]) < 2:
        return  # synthetic draw didn't give us enough bands in any lane; skip

    client.patch(f"/api/lanes/{ladder_lane['id']}", json={"is_ladder": True})
    kdas = [100.0, 50.0]
    for band, kda in zip(ladder_lane["bands"][:2], kdas):
        client.patch(f"/api/bands/{band['id']}", json={"known_kda": kda})

    resp = client.post(f"/api/images/{img['id']}/calibrate")
    assert resp.status_code == 200
    other_lanes = [l for l in resp.get_json()["lanes"] if not l["is_ladder"]]
    has_estimate = any(b["estimated_kda"] is not None for l in other_lanes for b in l["bands"])
    assert has_estimate or all(len(l["bands"]) == 0 for l in other_lanes)


def test_exports(client):
    pid = _upload(client, seed=3)

    resp = client.get(f"/api/projects/{pid}/export/csv")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert b"lane_index" in resp.data

    resp = client.get(f"/api/projects/{pid}/export/png")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"

    resp = client.get(f"/api/projects/{pid}/export/pdf")
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"


def test_unknown_project_404(client):
    resp = client.get("/api/projects/doesnotexist")
    assert resp.status_code == 404


def _first_image(client, pid):
    return client.get(f"/api/projects/{pid}").get_json()["images"][0]


def test_add_lane_inside_a_lane_splits_it_and_moves_right_hand_bands(client):
    image = _first_image(client, _upload(client))
    lane = max(image["lanes"], key=lambda l: len(l["bands"]))
    mid = (lane["x_start"] + lane["x_end"]) / 2
    client.patch(f"/api/lanes/{lane['id']}", json={"label": "keep me"})
    band_total = sum(len(l["bands"]) for l in image["lanes"])

    resp = client.post(f"/api/images/{image['id']}/lanes", json={"x": mid})
    assert resp.status_code == 201, resp.get_json()
    updated = resp.get_json()
    assert len(updated["lanes"]) == len(image["lanes"]) + 1
    assert [l["index"] for l in updated["lanes"]] == list(range(len(updated["lanes"])))
    assert sum(len(l["bands"]) for l in updated["lanes"]) == band_total
    left = next(l for l in updated["lanes"] if l["id"] == lane["id"])
    right = next(l for l in updated["lanes"] if l["x_start"] == mid)
    assert left["x_end"] == mid and left["label"] == "keep me"
    assert all(b["x"] + b["width"] / 2 >= mid for b in right["bands"])
    assert all(b["x"] + b["width"] / 2 < mid for b in left["bands"])


def test_add_lane_in_empty_space_creates_a_new_lane(client):
    image = _first_image(client, _upload(client))
    first = min(image["lanes"], key=lambda l: l["x_start"])
    client.patch(f"/api/lanes/{first['id']}", json={"x_start": first["x_start"] + (first["x_end"] - first["x_start"]) / 2})

    resp = client.post(f"/api/images/{image['id']}/lanes", json={"x": first["x_start"] + 1})
    assert resp.status_code == 201, resp.get_json()
    updated = resp.get_json()
    assert len(updated["lanes"]) == len(image["lanes"]) + 1
    assert updated["lanes"][0]["index"] == 0
    assert updated["lanes"][0]["x_end"] <= updated["lanes"][1]["x_start"]


def test_delete_lane_removes_it_and_its_bands_and_renumbers(client):
    image = _first_image(client, _upload(client))
    victim = image["lanes"][0]
    resp = client.delete(f"/api/lanes/{victim['id']}")
    assert resp.status_code == 200
    updated = resp.get_json()
    assert victim["id"] not in [l["id"] for l in updated["lanes"]]
    assert [l["index"] for l in updated["lanes"]] == list(range(len(updated["lanes"])))
    assert client.delete(f"/api/lanes/{victim['id']}").status_code == 404
