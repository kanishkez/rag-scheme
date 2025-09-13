"""
FastAPI server for government scheme RAG system.
Provides endpoints for CSV ingestion and semantic search.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import logging
from dotenv import load_dotenv

from utils import csv_processor
from rag import RAGSystem

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Government Scheme RAG API",
    description="RAG system for government scheme search using OpenAI embeddings and FAISS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem()

# Pydantic models for request/response
class IngestRequest(BaseModel):
    csv_path: str = "updated_data.csv"
    force_rebuild: bool = False

class IngestResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    index_stats: Dict[str, Any]

class QueryRequest(BaseModel):
    question: str
    k: int = 5

class QueryResponse(BaseModel):
    success: bool
    query: str
    matches: List[Dict[str, Any]]
    total_matches: int

class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_stats: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        # Try to load existing index
        if rag_system.load_index():
            logger.info("Loaded existing FAISS index")
        else:
            logger.info("No existing index found. Ready for CSV ingestion.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Government Scheme RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "ingest": "POST /ingest",
        "query": "POST /query"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        index_loaded = rag_system.index is not None
        index_stats = rag_system.get_index_stats() if index_loaded else None
        
        return HealthResponse(
            status="healthy" if index_loaded else "no_index",
            index_loaded=index_loaded,
            index_stats=index_stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            index_loaded=False,
            index_stats={"error": str(e)}
        )

@app.post("/ingest", response_model=IngestResponse)
async def ingest_csv(request: IngestRequest):
    """
    Ingest CSV file and build/update FAISS index.
    
    Args:
        request: IngestRequest with CSV path and options
        
    Returns:
        IngestResponse with processing results
    """
    try:
        # Check if CSV file exists
        if not os.path.exists(request.csv_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CSV file not found: {request.csv_path}"
            )
        
        # Check if index already exists and force_rebuild is False
        if not request.force_rebuild and rag_system.index is not None:
            return IngestResponse(
                success=True,
                message="Index already exists. Use force_rebuild=true to rebuild.",
                documents_processed=rag_system.index.ntotal,
                index_stats=rag_system.get_index_stats()
            )
        
        logger.info(f"Processing CSV: {request.csv_path}")
        
        # Process CSV and create documents
        processor = csv_processor.SchemeDocumentProcessor(request.csv_path)
        documents, metadata = processor.process_schemes()
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid documents found in CSV"
            )
        
        # Build RAG index
        success = rag_system.build_index(documents, metadata)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to build index"
            )
        
        # Get index statistics
        index_stats = rag_system.get_index_stats()
        
        return IngestResponse(
            success=True,
            message=f"Successfully processed {len(documents)} documents from {request.csv_path}",
            documents_processed=len(documents),
            index_stats=index_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_schemes(request: QueryRequest):
    """
    Query the scheme database using natural language.
    
    Args:
        request: QueryRequest with question and number of results
        
    Returns:
        QueryResponse with matching schemes
    """
    try:
        # Check if index is loaded
        if rag_system.index is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No index loaded. Please ingest data first using /ingest endpoint."
            )
        
        # Validate query
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Search for similar schemes
        results = rag_system.search(request.question, request.k)
        
        return QueryResponse(
            success=True,
            query=request.question,
            matches=results,
            total_matches=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get statistics about the current index."""
    try:
        if rag_system.index is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No index loaded"
            )
        
        return rag_system.get_index_stats()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv("HF_TOKEN"):
        logger.error("HF_TOKEN not found in environment variables")
        logger.info("Please create a .env file with your Hugging Face token")
        exit(1)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )