"""
Local server entrypoint for the FastAPI app.

uvicorn loads the FastAPI app from this file:

    uvicorn app:app --reload --host 127.0.0.1 --port 8000

This file does not define API routes.
The actual API routes are defined in src.api.
"""

from src.api import app