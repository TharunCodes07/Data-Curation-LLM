"""
Smart text corrector that only sends error-containing chunks to LLM for correction.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from ..config import config
from ..grammar.cfg_checker import CFGGrammarChecker, GrammarError
from ..utils.text_chunker import TextChunker

app_logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Result of text correction process"""
    original_text: str
    corrected_text: str
    had_errors: bool
    error_count: int
    errors_fixed: List[str]
    correction_confidence: float

class SmartTextCorrector:
    """Smart text corrector that only corrects chunks with CFG errors"""
    
    def __init__(self):
        """Initialize the smart corrector"""
        self.llm = ChatGoogleGenerativeAI(
            model=config.get('llm.gemini.model', 'gemini-2.5-flash'),
            temperature=config.get('llm.gemini.temperature', 0.3),
            max_output_tokens=config.get('llm.gemini.max_tokens', 1000)
        )
        
        self.cfg_checker = CFGGrammarChecker()
        self.text_chunker = TextChunker(
            chunk_size=config.get('grammar.chunk_size', 500),
            overlap=config.get('grammar.chunk_overlap', 50)
        )
        
        app_logger.info("Smart text corrector initialized")
    
    def correct_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correct a single document by checking chunks for errors and only correcting those with issues.
        
        Args:
            document: Document dictionary with 'content' and other metadata
            
        Returns:
            Corrected document with correction statistics
        """
        content = document.get('content', '')
        if not content.strip():
            return document
        
        app_logger.info(f"Processing document: {document.get('title', 'Untitled')}")
        
        # Split content into chunks
        chunks = self.text_chunker.chunk_text(content)
        app_logger.info(f"Split into {len(chunks)} chunks")
        
        corrected_chunks = []
        total_errors = 0
        corrected_chunks_count = 0
        all_corrections = []
        
        for i, chunk in enumerate(chunks):
            chunk_result = self._process_chunk(chunk, i)
            corrected_chunks.append(chunk_result.corrected_text)
            
            if chunk_result.had_errors:
                corrected_chunks_count += 1
                total_errors += chunk_result.error_count
                all_corrections.extend(chunk_result.errors_fixed)
                app_logger.info(f"Chunk {i+1}: Fixed {chunk_result.error_count} errors")
            else:
                app_logger.debug(f"Chunk {i+1}: No errors found, skipped LLM correction")
        
        # Combine corrected chunks
        corrected_content = ' '.join(corrected_chunks)
        
        # Update document
        corrected_document = document.copy()
        corrected_document['content'] = corrected_content
        corrected_document['correction_stats'] = {
            'total_chunks': len(chunks),
            'chunks_corrected': corrected_chunks_count,
            'total_errors_found': total_errors,
            'errors_fixed': all_corrections,
            'efficiency_ratio': f"{corrected_chunks_count}/{len(chunks)}"
        }
        
        app_logger.info(f"Document correction complete: {corrected_chunks_count}/{len(chunks)} chunks needed correction")
        
        return corrected_document
    
    def _process_chunk(self, chunk: str, chunk_index: int) -> CorrectionResult:
        """
        Process a single chunk - check for errors and correct if needed.
        
        Args:
            chunk: Text chunk to process
            chunk_index: Index of the chunk for logging
            
        Returns:
            CorrectionResult with correction details
        """
        # First, check for CFG errors
        errors = self.cfg_checker.check_text(chunk)
        
        if not errors:
            # No errors found, return original chunk
            return CorrectionResult(
                original_text=chunk,
                corrected_text=chunk,
                had_errors=False,
                error_count=0,
                errors_fixed=[],
                correction_confidence=1.0
            )
        
        # Errors found, send to LLM for correction
        app_logger.info(f"Chunk {chunk_index + 1}: Found {len(errors)} errors, sending to LLM for correction")
        
        corrected_text = self._correct_chunk_with_llm(chunk, errors)
        
        # Extract error types for reporting
        error_types = [f"{error.rule_name}: {error.description}" for error in errors[:3]]  # Top 3 errors
        
        return CorrectionResult(
            original_text=chunk,
            corrected_text=corrected_text,
            had_errors=True,
            error_count=len(errors),
            errors_fixed=error_types,
            correction_confidence=0.8  # Default confidence
        )
    
    def _correct_chunk_with_llm(self, chunk: str, errors: List[GrammarError]) -> str:
        """
        Correct a chunk using LLM with context about specific errors found.
        
        Args:
            chunk: Text chunk with errors
            errors: List of grammar errors detected
            
        Returns:
            Corrected text
        """
        # Create error context for the LLM
        error_context = []
        for error in errors[:5]:  # Limit to top 5 errors
            error_context.append(f"- {error.rule_name}: {error.description} (found: '{error.text_snippet}')")
        
        error_summary = "\n".join(error_context)
        
        prompt = f"""
        Please correct the following text. Focus on fixing these specific grammar errors that were detected:

        DETECTED ERRORS:
        {error_summary}

        TEXT TO CORRECT:
        {chunk}

        Please return ONLY the corrected text, maintaining the original meaning and style. 
        Fix the grammar errors while preserving the content structure.
        """
        
        try:
            response = self.llm.invoke(prompt)
            corrected_text = response.content.strip()
            
            # Basic validation - ensure we got a reasonable response
            if len(corrected_text) < len(chunk) // 2:
                app_logger.warning("LLM response too short, using original text")
                return chunk
            
            return corrected_text
            
        except Exception as e:
            app_logger.error(f"Error correcting chunk with LLM: {e}")
            return chunk
    
    def correct_scraped_data(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Correct all documents in scraped data.
        
        Args:
            scraped_data: List of scraped documents
            
        Returns:
            List of corrected documents with correction statistics
        """
        app_logger.info(f"Starting smart correction of {len(scraped_data)} documents")
        
        corrected_documents = []
        total_chunks_processed = 0
        total_chunks_corrected = 0
        
        for i, document in enumerate(scraped_data):
            try:
                corrected_doc = self.correct_document(document)
                corrected_documents.append(corrected_doc)
                
                # Update statistics
                stats = corrected_doc.get('correction_stats', {})
                total_chunks_processed += stats.get('total_chunks', 0)
                total_chunks_corrected += stats.get('chunks_corrected', 0)
                
                app_logger.info(f"Processed document {i+1}/{len(scraped_data)}")
                
            except Exception as e:
                app_logger.error(f"Error correcting document {i+1}: {e}")
                # Add original document if correction fails
                corrected_documents.append(document)
        
        # Log final statistics
        efficiency = (total_chunks_corrected / total_chunks_processed * 100) if total_chunks_processed > 0 else 0
        app_logger.info(f"Smart correction complete: {total_chunks_corrected}/{total_chunks_processed} chunks corrected ({efficiency:.1f}% efficiency)")
        
        return corrected_documents

    def get_correction_summary(self, corrected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of correction statistics.
        
        Args:
            corrected_data: List of corrected documents
            
        Returns:
            Summary statistics
        """
        total_docs = len(corrected_data)
        total_chunks = 0
        total_corrected = 0
        total_errors = 0
        
        for doc in corrected_data:
            stats = doc.get('correction_stats', {})
            total_chunks += stats.get('total_chunks', 0)
            total_corrected += stats.get('chunks_corrected', 0)
            total_errors += stats.get('total_errors_found', 0)
        
        efficiency = (total_corrected / total_chunks * 100) if total_chunks > 0 else 0
        
        return {
            'total_documents': total_docs,
            'total_chunks': total_chunks,
            'chunks_corrected': total_corrected,
            'chunks_skipped': total_chunks - total_corrected,
            'total_errors_found': total_errors,
            'efficiency_percentage': round(efficiency, 1),
            'llm_calls_saved': total_chunks - total_corrected
        }