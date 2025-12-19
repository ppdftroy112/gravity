"""
FastAPI Main Application
Golf Game with Reinforcement Learning
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api_routes import router as api_router


# Create FastAPI app
app = FastAPI(
    title="AI Golf Game",
    description="Golf game with Stable-Baselines3 reinforcement learning",
    version="1.0.0"
)

# CORS middleware (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def root():
    """Serve the main frontend page."""
    return FileResponse(frontend_path / "index.html")


@app.get("/style.css")
async def get_css():
    """Serve CSS file."""
    return FileResponse(frontend_path / "style.css", media_type="text/css")


@app.get("/game.js")
async def get_game_js():
    """Serve game JavaScript."""
    return FileResponse(frontend_path / "game.js", media_type="application/javascript")


@app.get("/api.js")
async def get_api_js():
    """Serve API JavaScript."""
    return FileResponse(frontend_path / "api.js", media_type="application/javascript")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
