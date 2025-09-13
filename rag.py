"""
RAG (Retrieval-Augmented Generation) system using Hugging Face embeddings and FAISS.
Handles vector storage, similarity search, and document retrieval.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    RAG system for government scheme retrieval using Hugging Face embeddings and FAISS.
    """
    
    def __init__(self, 
                 index_path: str = "faiss_index",
                 embedding_model: str = "intfloat/multilingual-e5-base",
                 dimension: int = 768):
        """
        Initialize the RAG system.
        
        Args:
            index_path: Path to store FAISS index
            embedding_model: Hugging Face embedding model name
            dimension: Embedding dimension (768 for multilingual-e5-base)
        """
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.documents = []
        
        # Initialize Hugging Face client
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        self.client = InferenceClient(model=embedding_model, token=hf_token)
        logger.info(f"Initialized RAG system with model: {embedding_model}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using Hugging Face API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding creation fails
        """
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Process in batches to avoid rate limits
            batch_size = 50  # Smaller batch size for HF API
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Get embeddings from Hugging Face
                batch_embeddings = self.client.feature_extraction(batch)
                
                # Convert to list of lists if needed
                if isinstance(batch_embeddings, np.ndarray):
                    if batch_embeddings.ndim == 2:
                        batch_embeddings = batch_embeddings.tolist()
                    elif batch_embeddings.ndim == 3:
                        # Handle case where we get 3D array
                        batch_embeddings = batch_embeddings.tolist()
                
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            logger.info("Embeddings created successfully")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.client.feature_extraction(text)
            
            # Convert to list if needed
            if isinstance(embedding, np.ndarray):
                if embedding.ndim == 2:
                    embedding = embedding[0].tolist()
                else:
                    embedding = embedding.tolist()
            
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise Exception(f"Failed to create embedding: {str(e)}")
    
    def create_faiss_index(self, embeddings: List[List[float]]) -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            FAISS index
        """
        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Create FAISS index
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            index.add(embeddings_array)
            
            logger.info(f"Created FAISS index with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise Exception(f"Failed to create FAISS index: {str(e)}")
    
    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]]) -> bool:
        """
        Build the complete FAISS index from documents and metadata.
        
        Args:
            documents: List of document texts
            metadata: List of metadata dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents or not metadata:
                raise ValueError("Documents and metadata cannot be empty")
            
            if len(documents) != len(metadata):
                raise ValueError("Documents and metadata must have the same length")
            
            logger.info(f"Building index for {len(documents)} documents")
            
            # Create embeddings
            embeddings = self.create_embeddings(documents)
            
            # Create FAISS index
            self.index = self.create_faiss_index(embeddings)
            
            # Store documents and metadata
            self.documents = documents
            self.metadata = metadata
            
            # Save index
            self.save_index()
            
            logger.info("Index built and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            return False
    
    def save_index(self) -> bool:
        """
        Save the FAISS index and metadata to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.index is None:
                raise ValueError("No index to save")
            
            # Create directory if it doesn't exist
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            
            # Save metadata
            with open(self.index_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f)
            
            # Save documents
            with open(self.index_path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)
            
            # Save configuration
            config = {
                "embedding_model": self.embedding_model,
                "dimension": self.dimension
            }
            with open(self.index_path / "config.pkl", "wb") as f:
                pickle.dump(config, f)
            
            logger.info(f"Index saved to {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self) -> bool:
        """
        Load the FAISS index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if files exist
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            documents_file = self.index_path / "documents.pkl"
            config_file = self.index_path / "config.pkl"
            
            if not all([index_file.exists(), metadata_file.exists(), 
                       documents_file.exists(), config_file.exists()]):
                logger.info("Index files not found")
                return False
            
            # Load configuration
            with open(config_file, "rb") as f:
                config = pickle.load(f)
            
            # Update configuration
            self.embedding_model = config.get("embedding_model", self.embedding_model)
            self.dimension = config.get("dimension", self.dimension)
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata and documents
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            
            with open(documents_file, "rb") as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with metadata and scores
        """
        try:
            if self.index is None:
                raise ValueError("Index not loaded. Please load or build an index first.")
            
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Convert to numpy array and normalize
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Search
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata) and idx < len(self.documents):
                    result = {
                        "scheme": self.metadata[idx].get("scheme_name", "Unknown"),
                        "sector": self.metadata[idx].get("sector", "Unknown"),
                        "state": self.metadata[idx].get("state", "Unknown"),
                        "eligibility": self.metadata[idx].get("eligibility", "Not specified"),
                        "benefits": self.metadata[idx].get("benefits", "Not specified"),
                        "official_url": self.metadata[idx].get("official_url", "Not available"),
                        "level": self.metadata[idx].get("level", "Unknown"),
                        "tags": self.metadata[idx].get("tags", ""),
                        "score": float(score),
                        "document": self.documents[idx]
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise Exception(f"Search failed: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"error": "No index loaded"}
        
        return {
            "total_documents": self.index.ntotal,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP",
            "metadata_count": len(self.metadata),
            "documents_count": len(self.documents)
        }
