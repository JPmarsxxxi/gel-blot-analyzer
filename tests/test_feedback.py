import os

from app.config import BASE_DIR

MAILTO = "mailto:hello@fumiscience.com"


def test_both_pages_link_to_feedback(client):
    assert MAILTO in client.get("/").get_data(as_text=True)
    page = open(os.path.join(BASE_DIR, "app", "templates", "project.html"), encoding="utf-8").read()
    assert MAILTO in page and ">Feedback</a>" in page
