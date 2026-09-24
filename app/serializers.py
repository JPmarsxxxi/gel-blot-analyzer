from app.models.schema import Band, GelImage, Lane, Project


def serialize_band(band: Band) -> dict:
    return {
        "id": band.id,
        "lane_id": band.lane_id,
        "x": band.x,
        "y": band.y,
        "width": band.width,
        "height": band.height,
        "intensity": band.intensity,
        "percent_of_lane": band.percent_of_lane,
        "confidence": band.confidence,
        "low_confidence": band.low_confidence,
        "known_kda": band.known_kda,
        "estimated_kda": band.estimated_kda,
        "manually_edited": band.manually_edited,
    }


def serialize_lane(lane: Lane) -> dict:
    return {
        "id": lane.id,
        "index": lane.index,
        "x_start": lane.x_start,
        "x_end": lane.x_end,
        "label": lane.label,
        "is_ladder": lane.is_ladder,
        "bands": [serialize_band(b) for b in lane.bands],
    }


def serialize_image(image: GelImage) -> dict:
    return {
        "id": image.id,
        "order_index": image.order_index,
        "orig_filename": image.orig_filename,
        "url": f"/storage/uploads/{image.project_id}/{image.stored_filename}",
        "width": image.width,
        "height": image.height,
        "crop_x": image.crop_x,
        "crop_y": image.crop_y,
        "crop_w": image.crop_w,
        "crop_h": image.crop_h,
        "rotate_deg": image.rotate_deg,
        "brightness": image.brightness,
        "contrast": image.contrast,
        "sensitivity": image.sensitivity,
        "is_gel_like": image.is_gel_like,
        "lanes": [serialize_lane(l) for l in image.lanes],
    }


def serialize_project(project: Project) -> dict:
    return {
        "id": project.id,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
        "images": [serialize_image(i) for i in project.images],
    }
