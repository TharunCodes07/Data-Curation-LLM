"""
Text chunking utilities for processing large texts efficiently.
"""
from typing import List, Optional
import re


class TextChunker:
    """Utility class for intelligent text chunking"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks, preserving sentence boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns"""
        # Basic sentence splitting pattern
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Re-add sentence endings (except for last sentence)
        result = []
        original_parts = re.findall(r'[^.!?]*[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if i < len(original_parts):
                # Try to match with original parts that include punctuation
                for part in original_parts:
                    if sentence in part:
                        result.append(part.strip())
                        break
                else:
                    result.append(sentence)
            else:
                result.append(sentence)
        
        return result
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last part of text for overlap with next chunk"""
        if len(text) <= self.overlap:
            return text
        
        # Try to find a good break point (end of sentence)
        overlap_start = len(text) - self.overlap
        
        # Look for sentence ending within overlap region
        overlap_text = text[overlap_start:]
        sentence_start = re.search(r'[.!?]\s+', overlap_text)
        
        if sentence_start:
            # Start from the sentence beginning
            return overlap_text[sentence_start.end():]
        else:
            # Use word boundary
            words = overlap_text.split()
            if len(words) > 1:
                return " ".join(words[1:])  # Skip first partial word
            else:
                return overlap_text