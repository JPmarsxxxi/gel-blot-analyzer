"""Production entry point: gunicorn app.wsgi:app"""
from app.maintenance import purge_stale_projects, start_purge_thread
from app.server import create_app

app = create_app()
purge_stale_projects()
start_purge_thread()
