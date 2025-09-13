import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional
from pathlib import Path

class FAISSHandler:
    def __init__(self, index_path: str = "faiss_index"):
        self.index_path = Path(index_path)
        self.index = None
        self.metadata = []
        self.dimension = None
        
    def create_index(self, dimension: int):
        """Create a new FAISS index with the specified dimension."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata = []
    
    def add_embeddings(self, embeddings: List[List[float]], texts: List[str]):
        """Add embeddings and their corresponding texts to the index."""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(texts)
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[float, str]]:
        """Search for the top-k most similar embeddings."""
        if self.index is None:
            raise ValueError("Index not initialized.")
        
        # Convert query to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.index.search(query_array, k)
        
        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))
        
        return results
    
    def save_index(self):
        """Save the FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save.")
        
        # Create directory if it doesn't exist
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        
        # Save metadata
        with open(self.index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save dimension info
        with open(self.index_path / "dimension.txt", "w") as f:
            f.write(str(self.dimension))
    
    def load_index(self) -> bool:
        """Load the FAISS index and metadata from disk."""
        try:
            # Check if files exist
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            dimension_file = self.index_path / "dimension.txt"
            
            if not all([index_file.exists(), metadata_file.exists(), dimension_file.exists()]):
                return False
            
            # Load dimension
            with open(dimension_file, "r") as f:
                self.dimension = int(f.read().strip())
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_index_size(self) -> int:
        """Get the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
