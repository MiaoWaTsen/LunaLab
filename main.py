"""
LunaLab — Application Entry Point

Usage:
    python main.py

Starts the FastAPI server with uvicorn on http://localhost:8000
"""

import uvicorn


def main():
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
