"""Run the site and the API together on your laptop:  python dev.py  ->  http://localhost:8000

On Vercel, files in public/ are served as the site and api/index.py serves /api/*.
"""

import os

import uvicorn
from fastapi.staticfiles import StaticFiles

from api.index import app

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "public"), html=True), name="site")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
