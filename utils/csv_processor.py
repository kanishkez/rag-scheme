import pandas as pd
from typing import List, Tuple
import re

class CSVProcessor:
    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (4 characters per token average)."""
        return len(text) // 4
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token limit."""
        if self.estimate_tokens(text) <= self.max_tokens:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed the limit
            if self.estimate_tokens(current_chunk + " " + sentence) > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # If single sentence is too long, split by words
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        if self.estimate_tokens(word_chunk + " " + word) > self.max_tokens:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                # If single word is too long, truncate
                                chunks.append(word[:self.max_tokens * 4])
                        else:
                            word_chunk += " " + word if word_chunk else word
                    current_chunk = word_chunk
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_csv(self, csv_path: str, text_column: str = "text") -> Tuple[List[str], List[str]]:
        """Process CSV file and return chunked texts and their sources."""
        try:
            df = pd.read_csv(csv_path)
            
            # For scheme data, combine multiple relevant columns for better search
            if "scheme_name" in df.columns and "details" in df.columns:
                # This is scheme data - combine multiple columns
                all_chunks = []
                all_sources = []
                
                for idx, row in df.iterrows():
                    # Combine relevant columns for comprehensive search
                    combined_text = self._combine_scheme_columns(row)
                    
                    if not combined_text.strip():
                        continue
                    
                    chunks = self.chunk_text(combined_text)
                    all_chunks.extend(chunks)
                    
                    # Create source identifier with scheme name
                    scheme_name = str(row.get('scheme_name', f'row_{idx}')).replace('"', '').strip()
                    source = f"{scheme_name}_row_{idx}"
                    all_sources.extend([source] * len(chunks))
                
                return all_chunks, all_sources
            
            else:
                # Standard processing for other CSV files
                if text_column not in df.columns:
                    raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
                
                all_chunks = []
                all_sources = []
                
                for idx, row in df.iterrows():
                    text = str(row[text_column])
                    if pd.isna(text) or text.strip() == "":
                        continue
                    
                    chunks = self.chunk_text(text)
                    all_chunks.extend(chunks)
                    
                    # Create source identifier
                    source = f"row_{idx}"
                    all_sources.extend([source] * len(chunks))
                
                return all_chunks, all_sources
            
        except Exception as e:
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _combine_scheme_columns(self, row) -> str:
        """Combine multiple columns from scheme data for comprehensive search."""
        combined_parts = []
        
        # Add scheme name
        if 'scheme_name' in row and pd.notna(row['scheme_name']):
            scheme_name = str(row['scheme_name']).replace('"', '').strip()
            combined_parts.append(f"Scheme: {scheme_name}")
        
        # Add details (main content)
        if 'details' in row and pd.notna(row['details']):
            details = str(row['details']).replace('"', '').strip()
            combined_parts.append(f"Details: {details}")
        
        # Add benefits
        if 'benefits' in row and pd.notna(row['benefits']):
            benefits = str(row['benefits']).replace('"', '').strip()
            combined_parts.append(f"Benefits: {benefits}")
        
        # Add eligibility
        if 'eligibility' in row and pd.notna(row['eligibility']):
            eligibility = str(row['eligibility']).replace('"', '').strip()
            combined_parts.append(f"Eligibility: {eligibility}")
        
        # Add application process
        if 'application' in row and pd.notna(row['application']):
            application = str(row['application']).replace('"', '').strip()
            combined_parts.append(f"Application Process: {application}")
        
        # Add required documents
        if 'documents' in row and pd.notna(row['documents']):
            documents = str(row['documents']).replace('"', '').strip()
            combined_parts.append(f"Required Documents: {documents}")
        
        # Add scheme category
        if 'schemeCategory' in row and pd.notna(row['schemeCategory']):
            category = str(row['schemeCategory']).replace('"', '').strip()
            combined_parts.append(f"Category: {category}")
        
        # Add level (State/Central)
        if 'level' in row and pd.notna(row['level']):
            level = str(row['level']).replace('"', '').strip()
            combined_parts.append(f"Level: {level}")
        
        # Add tags
        if 'tags' in row and pd.notna(row['tags']):
            tags = str(row['tags']).replace('"', '').strip()
            combined_parts.append(f"Tags: {tags}")
        
        return " | ".join(combined_parts)
    
    def validate_csv(self, csv_path: str, text_column: str = "text") -> bool:
        """Validate that CSV file exists and has the required column."""
        try:
            df = pd.read_csv(csv_path)
            
            # For scheme data, check for scheme_name and details columns
            if "scheme_name" in df.columns and "details" in df.columns:
                return True
            
            # For other CSV files, check for the specified text column
            return text_column in df.columns
        except Exception:
            return False
