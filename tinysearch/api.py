"""
FastAPI interface for TinySearch
"""
from typing import Dict, Any, List, Optional, Union, cast
from pathlib import Path
import os
import json
import shutil
import secrets
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Body, Depends, File, UploadFile, BackgroundTasks, Header, Security, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import tempfile

from .config import Config
from .cli import load_embedder, load_indexer, load_query_engine, load_adapter, load_splitter
from .flow.controller import FlowController


# Models for API requests and responses
class QueryRequest(BaseModel):
    """
    Query request model
    """
    query: str = Field(..., description="Query string")
    top_k: int = Field(5, description="Number of results to return")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional query parameters")


class QueryResponse(BaseModel):
    """
    Query response model
    """
    matches: List[Dict[str, Any]] = Field(..., description="List of matching chunks")


class HealthResponse(BaseModel):
    """
    Health check response model
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="TinySearch version")


class IndexBuildRequest(BaseModel):
    """
    Index build request model
    """
    data_path: str = Field(..., description="Path to data file or directory")
    force_rebuild: bool = Field(False, description="Force rebuild even if already processed")
    extensions: Optional[List[str]] = Field(None, description="File extensions to process")
    recursive: bool = Field(True, description="Whether to recursively process subdirectories")


class IndexBuildResponse(BaseModel):
    """
    Index build response model
    """
    status: str = Field(..., description="Build status")
    message: str = Field(..., description="Build message")
    processed_files: List[str] = Field(..., description="List of processed files")


class IndexStatsResponse(BaseModel):
    """
    Index stats response model
    """
    stats: Dict[str, Any] = Field(..., description="Index statistics")


class StatusResponse(BaseModel):
    """
    Status response model
    """
    status: str = Field(..., description="Status")
    message: str = Field(..., description="Message")


class ApiKeyResponse(BaseModel):
    """
    API key response model
    """
    api_key: str = Field(..., description="Generated API key")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")


# API Key security scheme
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Global variables
app = FastAPI(
    title="TinySearch API",
    description="API for TinySearch vector retrieval system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files for the web UI
package_dir = Path(__file__).resolve().parent
static_dir = package_dir / "api" / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# Global components
config = None
embedder = None
indexer = None
query_engine = None
flow_controller = None

# Authentication and rate limiting storage
api_keys = {}  # {api_key: {"expires_at": datetime, "created_at": datetime}}
rate_limit_storage = {}  # {api_key: {"requests": [(timestamp, path), ...], "limit": int, "window": int}}


def get_version() -> str:
    """
    Get the TinySearch version
    
    Returns:
        Version string
    """
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"


def initialize_components():
    """
    Initialize the TinySearch components
    """
    global config, embedder, indexer, query_engine, flow_controller
    
    config_path = os.environ.get("TINYSEARCH_CONFIG", "config.yaml")
    
    try:
        # Load configuration
        config = Config(config_path)
        config_dict = config.config
        
        # Load components
        data_adapter = load_adapter(config)
        text_splitter = load_splitter(config)
        embedder = load_embedder(config)
        indexer = load_indexer(config)
        query_engine = load_query_engine(config, embedder, indexer)
        
        # Initialize flow controller
        flow_controller = FlowController(
            data_adapter=data_adapter,
            text_splitter=text_splitter,
            embedder=embedder,
            indexer=indexer,
            query_engine=query_engine,
            config=config_dict
        )
        
        # Load index
        index_path = config.get("indexer.index_path", "index.faiss")
        if Path(index_path).exists():
            indexer.load(index_path)
            print(f"TinySearch initialized with index: {index_path}")
        else:
            print(f"Warning: Index not found: {index_path}")
            
        # Set up default API key if enabled
        if config.get("api.auth_enabled", False) and config.get("api.default_key"):
            add_api_key(config.get("api.default_key"), None)
    except Exception as e:
        print(f"Error initializing TinySearch: {e}")
        raise e


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Verify the provided API key
    
    Args:
        api_key: API key from header
    
    Returns:
        Validated API key or None if authentication is disabled
    
    Raises:
        HTTPException: If API key is invalid or expired
    """
    global config
    
    # Check if auth is enabled
    if not config or not config.get("api.auth_enabled", False):
        return None
        
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key missing",
            headers={"WWW-Authenticate": "API key required in X-API-Key header"},
        )
        
    if api_key not in api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Invalid API key"},
        )
    
    # Check expiration
    key_data = api_keys[api_key]
    if key_data.get("expires_at") and datetime.now() > key_data["expires_at"]:
        # Remove expired key
        del api_keys[api_key]
        raise HTTPException(
            status_code=401,
            detail="API Key expired",
            headers={"WWW-Authenticate": "API key expired"},
        )
    
    return api_key


def check_rate_limit(api_key: Optional[str], request: Request):
    """
    Check rate limiting for the current request
    
    Args:
        api_key: API key
        request: FastAPI request object
    
    Raises:
        HTTPException: If rate limit is exceeded
    """
    global config
    
    # Check if rate limiting is enabled
    if not config or not config.get("api.rate_limit_enabled", False):
        return
    
    # Use client IP if no API key (for public endpoints)
    client_id = api_key if api_key else request.client.host if request.client else "unknown"
    now = time.time()
    
    # Get rate limit settings from config
    default_limit = config.get("api.rate_limit", 60)  # requests
    default_window = config.get("api.rate_limit_window", 60)  # seconds
    
    # Initialize rate limit data if needed
    if client_id not in rate_limit_storage:
        rate_limit_storage[client_id] = {
            "requests": [],
            "limit": default_limit,
            "window": default_window
        }
    
    client_data = rate_limit_storage[client_id]
    
    # Clean up old requests outside the window
    window_start = now - client_data["window"]
    client_data["requests"] = [r for r in client_data["requests"] if r[0] >= window_start]
    
    # Check if limit exceeded
    if len(client_data["requests"]) >= client_data["limit"]:
        # Find time until oldest request expires
        oldest_time = client_data["requests"][0][0]
        reset_seconds = oldest_time + client_data["window"] - now
        retry_after = max(1, int(reset_seconds))
        
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )
    
    # Add current request
    client_data["requests"].append((now, request.url.path))


def add_api_key(key: Optional[str] = None, expires_in_days: Optional[int] = 30) -> str:
    """
    Add a new API key
    
    Args:
        key: Custom API key (will be generated if None)
        expires_in_days: Number of days until expiration (None for no expiration)
        
    Returns:
        Generated API key
    """
    generated_key = key if key is not None else secrets.token_urlsafe(32)
    
    expires_at = datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None
    
    api_keys[generated_key] = {
        "created_at": datetime.now(),
        "expires_at": expires_at
    }
    
    return generated_key


@app.on_event("startup")
def startup_event():
    """
    Startup event handler
    """
    try:
        initialize_components()
    except Exception as e:
        print(f"Failed to initialize components: {e}")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    return {
        "status": "ok" if query_engine is not None else "not_initialized",
        "version": get_version()
    }


@app.post("/api-key", response_model=ApiKeyResponse)
def generate_api_key(
    expires_in_days: Optional[int] = Query(30, description="Days until key expires (None for no expiration)"),
    master_key: Optional[str] = Header(None)
):
    """
    Generate a new API key
    
    Args:
        expires_in_days: Days until key expires (None for no expiration)
        master_key: Master API key for authentication
    
    Returns:
        Generated API key information
    """
    global config
    
    # Check if auth is enabled
    if not config or not config.get("api.auth_enabled", False):
        raise HTTPException(
            status_code=400,
            detail="API authentication is not enabled"
        )
    
    # Check master key
    config_master_key = config.get("api.master_key")
    if not config_master_key or master_key != config_master_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid master key"
        )
    
    # Generate key
    api_key = add_api_key(None, expires_in_days)
    
    expires_at = None
    if expires_in_days:
        expires_at = datetime.now() + timedelta(days=expires_in_days)
    
    return {
        "api_key": api_key,
        "expires_at": expires_at
    }


@app.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Query endpoint
    
    Args:
        request: FastAPI request
        query_request: Query request
        api_key: API key
        
    Returns:
        Query results
    """
    # Check rate limit
    check_rate_limit(api_key, request)
    
    if query_engine is None:
        raise HTTPException(status_code=500, detail="TinySearch not initialized")
    
    try:
        # Execute query
        results = query_engine.retrieve(query_request.query, top_k=query_request.top_k)
        
        # Format results
        matches = []
        for result in results:
            match = {
                "text": result["text"],
                "score": result["score"]
            }
            
            # Include metadata if available
            if "metadata" in result and result["metadata"]:
                match["metadata"] = result["metadata"]
            
            matches.append(match)
        
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/build", response_model=IndexBuildResponse)
async def build_index(
    request: Request,
    build_request: IndexBuildRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Build index endpoint
    
    Args:
        request: FastAPI request
        build_request: Build request
        api_key: API key
        
    Returns:
        Build status
    """
    # Check rate limit
    check_rate_limit(api_key, request)
    
    if flow_controller is None:
        raise HTTPException(status_code=500, detail="TinySearch not initialized")
    
    try:
        data_path = Path(build_request.data_path)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {build_request.data_path}")
        
        # Build index
        flow_controller.build_index(
            data_path=data_path,
            force_reprocess=build_request.force_rebuild,
            extensions=build_request.extensions,
            recursive=build_request.recursive
        )
        
        # Save index
        flow_controller.save_index()
        
        # Get stats
        stats = flow_controller.get_stats()
        
        return {
            "status": "success",
            "message": f"Index built with {len(stats['processed_files'])} files",
            "processed_files": stats["processed_files"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/upload", response_model=StatusResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Upload a document and add it to the index
    
    Args:
        request: FastAPI request
        background_tasks: Background tasks
        file: Uploaded file
        api_key: API key
        
    Returns:
        Upload status
    """
    # Check rate limit
    check_rate_limit(api_key, request)
    
    if flow_controller is None:
        raise HTTPException(status_code=500, detail="TinySearch not initialized")
    
    try:
        # Save uploaded file to a temporary location
        temp_dir = Path(tempfile.mkdtemp())
        # Ensure filename is not None before using with Path
        safe_filename = file.filename if file.filename is not None else "uploaded_file"
        temp_file = temp_dir / safe_filename
        
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the file
        flow_controller.process_file(temp_file)
        
        # Save index
        flow_controller.save_index()
        
        # Clean up in the background
        background_tasks.add_task(shutil.rmtree, temp_dir)
        
        return {
            "status": "success",
            "message": f"Document {file.filename} uploaded and indexed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Get index statistics
    
    Args:
        request: FastAPI request
        api_key: API key
        
    Returns:
        Index statistics
    """
    # Check rate limit
    check_rate_limit(api_key, request)
    
    if flow_controller is None:
        raise HTTPException(status_code=500, detail="TinySearch not initialized")
    
    try:
        stats = flow_controller.get_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/clear", response_model=StatusResponse)
async def clear_index(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Clear the index
    
    Args:
        request: FastAPI request
        api_key: API key
        
    Returns:
        Clear status
    """
    # Check rate limit
    check_rate_limit(api_key, request)
    
    if flow_controller is None:
        raise HTTPException(status_code=500, detail="TinySearch not initialized")
    
    try:
        # Clear cache
        flow_controller.clear_cache()
        
        # Reset indexer
        global indexer, config
        if config is not None:
            indexer = load_indexer(config)
            
            # Update flow controller's indexer
            if flow_controller is not None:
                flow_controller.indexer = indexer
            
            # Save empty index
            flow_controller.save_index()
        
        return {
            "status": "success",
            "message": "Index cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server
    
    Args:
        host: Host address
        port: Port number
        reload: Whether to enable auto-reload
    """
    import uvicorn
    
    # Start the server
    uvicorn.run("tinysearch.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_api() 