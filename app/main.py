from dotenv import load_dotenv
import os

# Load environment variables from .env file if it exists
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from app.utils import Output, QdrantService

# API Key from environment
API_KEY = os.environ.get("API_KEY", "default-api-key-change-in-production")
security = HTTPBearer()

# Global service instance (initialized at startup)
qdrant_service: Optional[QdrantService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup: Initialize QdrantService and load documents
    global qdrant_service
    qdrant_service = QdrantService(k=2)
    qdrant_service.connect()
    qdrant_service.load()
    print("Index initialized and documents loaded")
    yield
    # Shutdown: Cleanup (if needed in the future)


app = FastAPI(
    title="RAG Pipeline API",
    description="Query Game of Thrones laws using RAG",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The query string to search for")
    k: Optional[int] = Field(None, ge=1, le=20, description="Number of similar vectors to return (optional, default: 2)")


def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key from header"""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.post("/query", response_model=Output)
async def query_laws(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Query the laws database using RAG.
    
    Requires X-API-Key header for authentication.
    Optionally accepts k parameter to override the number of similar vectors returned.
    """
    global qdrant_service
    
    if qdrant_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    # Validate query length
    if len(request.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long (max 1000 characters)")
    
    # Override k if provided
    original_k = qdrant_service.k
    if request.k is not None:
        qdrant_service.k = request.k
    
    try:
        result = qdrant_service.query(request.query)
    except Exception as e:
        # Restore original k on error
        qdrant_service.k = original_k
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    finally:
        # Restore original k
        qdrant_service.k = original_k
    
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service_initialized": qdrant_service is not None}