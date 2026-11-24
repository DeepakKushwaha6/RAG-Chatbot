"""Start the DocuMind RAG chatbot server.

    python run.py
"""

from __future__ import annotations

import uvicorn

from app.config import settings

if __name__ == "__main__":
    print(f"DocuMind RAG chatbot -> http://{settings.host}:{settings.port}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
