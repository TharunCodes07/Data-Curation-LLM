"""
RAG (Retrieval-Augmented Generation) data management system using ChromaDB.
This module handles document storage, embedding generation, and similarity search.
"""
import hashlib
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import config
from src.logger import app_logger
from src.scrapers.web_scraper import ScrapedContent
from src.correction.gemini_corrector import CorrectionResult


class RAGDataManager:
    """Manages document storage and retrieval for RAG system using ChromaDB."""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """Initialize the RAG data manager."""
        self.config = config.get_rag_config()
        self.db_config = config.get_database_config()
        
        # Database settings
        self.db_path = db_path or self.db_config.get('chroma_db_path', './data/chroma_db')
        self.collection_name = collection_name or self.db_config.get('collection_name', 'curated_documents')
        
        # RAG settings
        self.embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.top_k = self.config.get('top_k_results', 5)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        # Initialize components
        self._initialize_database()
        self._initialize_embedding_model()
        self._initialize_text_splitter()
        
        app_logger.info(f"RAGDataManager initialized with collection '{self.collection_name}'")
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection."""
        # Create database directory
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            app_logger.info(f"Loaded existing collection '{self.collection_name}'")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Curated documents for RAG system"}
            )
            app_logger.info(f"Created new collection '{self.collection_name}'")
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            app_logger.info(f"Loaded embedding model '{self.embedding_model_name}'")
        except Exception as e:
            app_logger.error(f"Error loading embedding model: {e}")
            # Fallback to a smaller model
            self.embedding_model_name = 'all-MiniLM-L6-v2'
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            app_logger.info(f"Loaded fallback embedding model '{self.embedding_model_name}'")
    
    def _initialize_text_splitter(self):
        """Initialize text splitter for document chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        app_logger.info(f"Initialized text splitter with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def add_scraped_content(self, scraped_content: List[ScrapedContent]) -> Dict[str, Any]:
        """Add scraped content to the RAG database."""
        app_logger.info(f"Adding {len(scraped_content)} scraped documents to RAG database")
        
        documents = []
        metadatas = []
        ids = []
        
        for content in scraped_content:
            # Create document chunks
            chunks = self.text_splitter.split_text(content.content)
            
            for i, chunk in enumerate(chunks):
                # Create unique ID for each chunk
                chunk_id = f"{hashlib.md5(content.url.encode()).hexdigest()}_{i}"
                
                documents.append(chunk)
                metadatas.append({
                    'source_url': content.url,
                    'title': content.title,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split()),
                    'source_type': 'scraped',
                    'timestamp': content.timestamp,
                    'content_hash': content.content_hash,
                    'domain': content.metadata.get('domain', ''),
                    'original_word_count': content.word_count
                })
                ids.append(chunk_id)
        
        # Generate embeddings
        app_logger.info(f"Generating embeddings for {len(documents)} chunks")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        result = {
            'total_documents': len(scraped_content),
            'total_chunks': len(documents),
            'avg_chunks_per_doc': len(documents) / len(scraped_content) if scraped_content else 0,
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        app_logger.info(f"Successfully added {len(documents)} chunks from {len(scraped_content)} documents")
        return result
    
    def add_corrected_content(self, corrections: List[CorrectionResult], 
                            source_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add corrected content to the RAG database."""
        app_logger.info(f"Adding {len(corrections)} corrected texts to RAG database")
        
        documents = []
        metadatas = []
        ids = []
        
        for i, correction in enumerate(corrections):
            if correction.corrected_text != correction.original_text:
                # Only add if there were actual corrections
                chunks = self.text_splitter.split_text(correction.corrected_text)
                
                for j, chunk in enumerate(chunks):
                    # Create unique ID for corrected chunk
                    chunk_id = f"corrected_{hashlib.md5(correction.original_text.encode()).hexdigest()}_{i}_{j}"
                    
                    documents.append(chunk)
                    metadatas.append({
                        'source_type': 'corrected',
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'word_count': len(chunk.split()),
                        'correction_confidence': correction.confidence,
                        'changes_made': len(correction.changes_made),
                        'errors_addressed': ', '.join(correction.errors_addressed),
                        'processing_time': correction.processing_time,
                        'timestamp': time.time(),
                        'original_length': len(correction.original_text),
                        'corrected_length': len(correction.corrected_text),
                        **(source_metadata or {})
                    })
                    ids.append(chunk_id)
        
        if documents:
            # Generate embeddings
            app_logger.info(f"Generating embeddings for {len(documents)} corrected chunks")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        
        result = {
            'total_corrections': len(corrections),
            'total_chunks_added': len(documents),
            'corrections_with_changes': len([c for c in corrections if c.corrected_text != c.original_text]),
            'avg_confidence': sum(c.confidence for c in corrections) / len(corrections) if corrections else 0
        }
        
        app_logger.info(f"Successfully added {len(documents)} chunks from {len(corrections)} corrections")
        return result
    
    def query_similar(self, query: str, n_results: int = None, 
                     filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query for similar documents."""
        n_results = n_results or self.top_k
        
        app_logger.info(f"Querying for {n_results} similar documents")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Build where filter
        where_filter = filter_metadata or {}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                'distance': results['distances'][0][i]
            })
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in formatted_results 
            if r['similarity'] >= self.similarity_threshold
        ]
        
        app_logger.info(f"Found {len(filtered_results)} results above similarity threshold {self.similarity_threshold}")
        return filtered_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(limit=min(100, count))
            
            stats = {
                'total_documents': count,
                'embedding_model': self.embedding_model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'collection_name': self.collection_name,
                'db_path': self.db_path
            }
            
            if count > 0:
                # Analyze source types
                source_types = {}
                domains = {}
                total_word_count = 0
                
                for metadata in sample_results['metadatas']:
                    source_type = metadata.get('source_type', 'unknown')
                    source_types[source_type] = source_types.get(source_type, 0) + 1
                    
                    domain = metadata.get('domain', 'unknown')
                    domains[domain] = domains.get(domain, 0) + 1
                    
                    total_word_count += metadata.get('word_count', 0)
                
                stats.update({
                    'source_types': source_types,
                    'top_domains': dict(sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5]),
                    'avg_word_count_per_chunk': total_word_count / len(sample_results['metadatas']) if sample_results['metadatas'] else 0,
                    'sample_size': len(sample_results['metadatas'])
                })
            
            app_logger.info(f"Collection stats: {count} documents")
            return stats
            
        except Exception as e:
            app_logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by metadata filters."""
        app_logger.info(f"Searching by metadata: {metadata_filter}")
        
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            for i in range(len(results['documents'])):
                formatted_results.append({
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            app_logger.info(f"Found {len(formatted_results)} documents matching metadata filter")
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"Error searching by metadata: {e}")
            return []
    
    def delete_documents(self, ids: List[str] = None, where_filter: Dict[str, Any] = None) -> int:
        """Delete documents by IDs or metadata filter."""
        try:
            if ids:
                self.collection.delete(ids=ids)
                deleted_count = len(ids)
                app_logger.info(f"Deleted {deleted_count} documents by IDs")
            elif where_filter:
                # Get IDs first
                results = self.collection.get(where=where_filter, include=['metadatas'])
                if results['ids']:
                    self.collection.delete(where=where_filter)
                    deleted_count = len(results['ids'])
                    app_logger.info(f"Deleted {deleted_count} documents by metadata filter")
                else:
                    deleted_count = 0
                    app_logger.info("No documents found matching filter")
            else:
                raise ValueError("Must provide either ids or where_filter")
            
            return deleted_count
            
        except Exception as e:
            app_logger.error(f"Error deleting documents: {e}")
            return 0
    
    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Curated documents for RAG system"}
            )
            app_logger.info(f"Reset collection '{self.collection_name}'")
        except Exception as e:
            app_logger.error(f"Error resetting collection: {e}")
    
    def export_collection(self, output_file: str) -> str:
        """Export collection data to JSON file."""
        try:
            results = self.collection.get(include=['documents', 'metadatas'])
            
            export_data = {
                'collection_name': self.collection_name,
                'export_timestamp': time.time(),
                'total_documents': len(results['documents']),
                'embedding_model': self.embedding_model_name,
                'documents': []
            }
            
            for i in range(len(results['documents'])):
                export_data['documents'].append({
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
            
            # Create output directory
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            app_logger.info(f"Exported {len(results['documents'])} documents to {output_file}")
            return output_file
            
        except Exception as e:
            app_logger.error(f"Error exporting collection: {e}")
            return ""


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize RAG manager
        rag_manager = RAGDataManager()
        
        # Get collection stats
        stats = rag_manager.get_collection_stats()
        print("Collection Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test query if collection has data
        if stats.get('total_documents', 0) > 0:
            print("\nTesting similarity search...")
            results = rag_manager.query_similar("natural language processing", n_results=3)
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i} (similarity: {result['similarity']:.3f}):")
                print(f"  Document: {result['document'][:100]}...")
                print(f"  Source: {result['metadata'].get('source_url', 'Unknown')}")
        else:
            print("\nNo documents in collection. Add some data first using the scraper and correction modules.")
        
    except Exception as e:
        print(f"Error: {e}")