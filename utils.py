"""
CSV parsing and document creation utilities for government scheme data.
Handles parsing of updated_data.csv and creates structured documents for RAG.
"""

import pandas as pd
import os
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemeDocumentProcessor:
    """
    Processes government scheme CSV data and creates structured documents for RAG.
    """
    
    def __init__(self, csv_path: str = "updated_data.csv"):
        self.csv_path = csv_path
        self.schemes_data = []
        self.documents = []
        
    def load_csv(self) -> pd.DataFrame:
        """
        Load and validate the CSV file.
        
        Returns:
            pd.DataFrame: Loaded CSV data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Map column names to expected format (case-insensitive)
            column_mapping = self._map_columns(df.columns)
            df = df.rename(columns=column_mapping)
            
            # Validate required columns
            required_columns = ['scheme_name', 'details', 'benefits', 'eligibility']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                logger.info("Available columns will be used for document creation")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """
        Map CSV columns to standardized names.
        
        Args:
            columns: List of column names from CSV
            
        Returns:
            Dict mapping original names to standardized names
        """
        column_mapping = {}
        
        # Define mapping patterns (case-insensitive)
        mappings = {
            'scheme_name': ['scheme_name', 'schemename', 'scheme name', 'name'],
            'details': ['details', 'description', 'desc', 'about'],
            'benefits': ['benefits', 'benefit', 'financial_assistance', 'assistance'],
            'eligibility': ['eligibility', 'eligibilitycriteria', 'criteria', 'who_can_apply'],
            'application': ['application', 'how_to_apply', 'process', 'steps'],
            'documents': ['documents', 'required_documents', 'docs'],
            'level': ['level', 'government_level', 'state_central'],
            'scheme_category': ['schemeCategory', 'category', 'sector', 'domain'],
            'tags': ['tags', 'tag', 'keywords'],
            'state': ['state', 'region', 'location'],
            'official_url': ['officialurl', 'url', 'link', 'website']
        }
        
        for standard_name, patterns in mappings.items():
            for col in columns:
                if any(pattern.lower() in col.lower() for pattern in patterns):
                    column_mapping[col] = standard_name
                    break
        
        return column_mapping
    
    def create_documents(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Create text documents and metadata from CSV data.
        
        Args:
            df: DataFrame with scheme data
            
        Returns:
            Tuple of (documents, metadata_list)
        """
        documents = []
        metadata_list = []
        
        for idx, row in df.iterrows():
            try:
                # Create document text by combining key fields
                doc_parts = []
                
                # Add scheme name
                if 'scheme_name' in row and pd.notna(row['scheme_name']):
                    scheme_name = str(row['scheme_name']).strip().replace('"', '')
                    doc_parts.append(f"Scheme: {scheme_name}")
                
                # Add details/description
                if 'details' in row and pd.notna(row['details']):
                    details = str(row['details']).strip().replace('"', '')
                    doc_parts.append(f"Description: {details}")
                
                # Add benefits
                if 'benefits' in row and pd.notna(row['benefits']):
                    benefits = str(row['benefits']).strip().replace('"', '')
                    doc_parts.append(f"Benefits: {benefits}")
                
                # Add eligibility criteria
                if 'eligibility' in row and pd.notna(row['eligibility']):
                    eligibility = str(row['eligibility']).strip().replace('"', '')
                    doc_parts.append(f"Eligibility: {eligibility}")
                
                # Add application process
                if 'application' in row and pd.notna(row['application']):
                    application = str(row['application']).strip().replace('"', '')
                    doc_parts.append(f"Application Process: {application}")
                
                # Add required documents
                if 'documents' in row and pd.notna(row['documents']):
                    documents_req = str(row['documents']).strip().replace('"', '')
                    doc_parts.append(f"Required Documents: {documents_req}")
                
                # Add category/sector
                if 'scheme_category' in row and pd.notna(row['scheme_category']):
                    category = str(row['scheme_category']).strip().replace('"', '')
                    doc_parts.append(f"Category: {category}")
                
                # Add level (State/Central)
                if 'level' in row and pd.notna(row['level']):
                    level = str(row['level']).strip().replace('"', '')
                    doc_parts.append(f"Level: {level}")
                
                # Add tags
                if 'tags' in row and pd.notna(row['tags']):
                    tags = str(row['tags']).strip().replace('"', '')
                    doc_parts.append(f"Tags: {tags}")
                
                # Create the document text
                document_text = " | ".join(doc_parts)
                
                if document_text.strip():
                    documents.append(document_text)
                    
                    # Create metadata
                    metadata = {
                        'scheme_name': str(row.get('scheme_name', 'Unknown')).replace('"', '').strip(),
                        'sector': str(row.get('scheme_category', 'Unknown')).replace('"', '').strip(),
                        'state': str(row.get('state', 'All India')).replace('"', '').strip(),
                        'eligibility': str(row.get('eligibility', 'Not specified')).replace('"', '').strip(),
                        'benefits': str(row.get('benefits', 'Not specified')).replace('"', '').strip(),
                        'official_url': str(row.get('official_url', 'Not available')).replace('"', '').strip(),
                        'level': str(row.get('level', 'Unknown')).replace('"', '').strip(),
                        'tags': str(row.get('tags', '')).replace('"', '').strip(),
                        'row_index': idx
                    }
                    metadata_list.append(metadata)
                    
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                continue
        
        logger.info(f"Created {len(documents)} documents from {len(df)} rows")
        return documents, metadata_list
    
    def process_schemes(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Complete processing pipeline: load CSV and create documents.
        
        Returns:
            Tuple of (documents, metadata_list)
        """
        df = self.load_csv()
        return self.create_documents(df)
    
    def get_scheme_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the processed schemes.
        
        Returns:
            Dict with summary statistics
        """
        if not self.schemes_data:
            return {"error": "No data processed yet"}
        
        df = pd.DataFrame(self.schemes_data)
        
        summary = {
            "total_schemes": len(df),
            "states": df['state'].value_counts().to_dict() if 'state' in df.columns else {},
            "sectors": df['sector'].value_counts().to_dict() if 'sector' in df.columns else {},
            "levels": df['level'].value_counts().to_dict() if 'level' in df.columns else {}
        }
        
        return summary
