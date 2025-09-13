from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path

from utils.embeddings import EmbeddingService
from utils.faiss_handler import FAISSHandler
from utils.csv_processor import CSVProcessor

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
faiss_handler = FAISSHandler()
csv_processor = CSVProcessor()

class IngestRequest(BaseModel):
    csv_path: str
    text_column: str = "text"  # For scheme data, this will be ignored as we use multiple columns

class IngestResponse(BaseModel):
    message: str
    chunks_processed: int
    index_size: int

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    results: List[dict]
    query: str
    k: int

@router.post("/ingest", response_model=IngestResponse)
async def ingest_csv(request: IngestRequest):
    """Ingest CSV file and create/update FAISS index."""
    try:
        # Validate CSV file
        if not os.path.exists(request.csv_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CSV file not found: {request.csv_path}"
            )
        
        if not csv_processor.validate_csv(request.csv_path, request.text_column):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV file validation failed. For scheme data, ensure columns 'scheme_name' and 'details' exist. For other data, ensure column '{request.text_column}' exists."
            )
        
        # Process CSV
        chunks, sources = csv_processor.process_csv(request.csv_path, request.text_column)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text data found in CSV"
            )
        
        # Generate embeddings
        embeddings = embedding_service.embed_texts(chunks)
        
        # Create or update FAISS index
        if faiss_handler.index is None:
            faiss_handler.create_index(len(embeddings[0]))
        
        # Add embeddings to index
        faiss_handler.add_embeddings(embeddings, chunks)
        
        # Save index
        faiss_handler.save_index()
        
        return IngestResponse(
            message=f"Successfully ingested {len(chunks)} chunks from {request.csv_path}",
            chunks_processed=len(chunks),
            index_size=faiss_handler.get_index_size()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during ingestion: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the FAISS index for similar documents."""
    try:
        # Check if index exists
        if faiss_handler.index is None:
            # Try to load existing index
            if not faiss_handler.load_index():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No FAISS index found. Please ingest data first using /ingest endpoint."
                )
        
        # Generate query embedding
        query_embedding = embedding_service.embed_query(request.query)
        
        # Search for similar documents
        results = faiss_handler.search(query_embedding, request.k)
        
        # Format results
        formatted_results = []
        for i, (score, text) in enumerate(results):
            # Extract scheme name if available
            scheme_name = "Unknown Scheme"
            if "Scheme:" in text:
                try:
                    scheme_part = text.split("Scheme:")[1].split("|")[0].strip()
                    scheme_name = scheme_part[:100] + "..." if len(scheme_part) > 100 else scheme_part
                except:
                    pass
            
            formatted_results.append({
                "rank": i + 1,
                "score": score,
                "similarity": f"{score:.4f}",
                "scheme_name": scheme_name,
                "content": text,
                "preview": text[:200] + "..." if len(text) > 200 else text
            })
        
        return QueryResponse(
            results=formatted_results,
            query=request.query,
            k=request.k
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during query: {str(e)}"
        )

@router.get("/status")
async def get_status():
    """Get the current status of the system."""
    index_size = faiss_handler.get_index_size() if faiss_handler.index else 0
    index_loaded = faiss_handler.index is not None
    
    return {
        "index_loaded": index_loaded,
        "index_size": index_size,
        "embedding_service_ready": True
    }
