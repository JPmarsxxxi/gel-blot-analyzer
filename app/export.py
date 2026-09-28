"""CSV / annotated-PNG / PDF report export."""
import csv
import io
import os

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from app.config import UPLOADS_DIR
from app.models.schema import GelImage, Project

HIGH_CONF_COLOR = (0, 180, 90)
LOW_CONF_COLOR = (220, 40, 40)
LANE_LABEL_COLOR = (30, 60, 200)


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box, color, width=2, dash=6, gap=4):
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (sx, sy), (ex, ey) in edges:
        length = max(abs(ex - sx), abs(ey - sy))
        if length == 0:
            continue
        steps = int(length // (dash + gap)) + 1
        for i in range(steps):
            t0 = (i * (dash + gap)) / length
            t1 = min(1.0, (i * (dash + gap) + dash) / length)
            px0 = sx + (ex - sx) * t0
            py0 = sy + (ey - sy) * t0
            px1 = sx + (ex - sx) * t1
            py1 = sy + (ey - sy) * t1
            draw.line([(px0, py0), (px1, py1)], fill=color, width=width)


def render_annotated_image(gel_image: GelImage) -> Image.Image:
    path = os.path.join(UPLOADS_DIR, str(gel_image.project_id), gel_image.stored_filename)
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", max(12, img.width // 60))
    except OSError:
        font = ImageFont.load_default()

    for lane in gel_image.lanes:
        for band in lane.bands:
            box = (band.x, band.y, band.x + band.width, band.y + band.height)
            if band.low_confidence:
                _dashed_rectangle(draw, box, LOW_CONF_COLOR, width=2)
            else:
                draw.rectangle(box, outline=HIGH_CONF_COLOR, width=2)

        if lane.label:
            label_x = lane.x_start + 2
            label_y = max(0, 2)
            draw.text((label_x, label_y), lane.label, fill=LANE_LABEL_COLOR, font=font)
        draw.line([(lane.x_start, 0), (lane.x_start, img.height)], fill=(150, 150, 150), width=1)

    return img


def build_csv(project: Project) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["image", "lane_index", "sample_label", "band_index", "intensity", "percent_of_lane", "kda"]
    )
    for image in project.images:
        for lane in image.lanes:
            for i, band in enumerate(lane.bands, start=1):
                kda = band.estimated_kda if not lane.is_ladder else band.known_kda
                writer.writerow(
                    [
                        image.orig_filename,
                        lane.index,
                        lane.label,
                        i,
                        f"{band.intensity:.2f}" if band.intensity is not None else "",
                        f"{band.percent_of_lane:.2f}" if band.percent_of_lane is not None else "",
                        f"{kda:.2f}" if kda is not None else "",
                    ]
                )
    return buf.getvalue().encode("utf-8")


def build_pdf(project: Project) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Gel/Blot Analysis Report", styles["Title"]))
    story.append(Paragraph(f"Project: {project.id}", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    for image in project.images:
        annotated = render_annotated_image(image)
        img_buf = io.BytesIO()
        annotated.save(img_buf, format="PNG")
        img_buf.seek(0)

        max_w = 6.5 * inch
        scale = min(1.0, max_w / annotated.width)
        story.append(Paragraph(image.orig_filename, styles["Heading2"]))
        story.append(RLImage(img_buf, width=annotated.width * scale, height=annotated.height * scale))
        story.append(Spacer(1, 0.15 * inch))

        table_data = [["Lane", "Sample", "Band #", "Intensity", "% of lane", "kDa"]]
        for lane in image.lanes:
            for i, band in enumerate(lane.bands, start=1):
                kda = band.estimated_kda if not lane.is_ladder else band.known_kda
                table_data.append(
                    [
                        str(lane.index),
                        lane.label or "",
                        str(i),
                        f"{band.intensity:.1f}" if band.intensity is not None else "",
                        f"{band.percent_of_lane:.1f}" if band.percent_of_lane is not None else "",
                        f"{kda:.1f}" if kda is not None else "",
                    ]
                )

        table = Table(table_data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ]
            )
        )
        story.append(table)
        story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()
