"""
Context-aware text correction module using Gemini API.
This module takes grammar errors and provides corrected text while maintaining context.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import config
from src.logger import app_logger
from src.grammar.cfg_checker import GrammarError


@dataclass
class CorrectionResult:
    """Represents a text correction result."""
    original_text: str
    corrected_text: str
    changes_made: List[str]
    confidence: float
    reasoning: str
    processing_time: float
    errors_addressed: List[str]


class GeminiTextCorrector:
    """Text corrector using Gemini API for context-aware grammar correction."""
    
    def __init__(self, api_key: str = None):
        """Initialize the text corrector."""
        self.config = config.get_llm_config()
        self.grammar_config = config.get_grammar_config()
        
        # Initialize LLM
        if not api_key:
            api_key = config.google_api_key
        
        if not api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.config['gemini']['model'],
            temperature=self.config['gemini']['temperature'],
            google_api_key=api_key
        )
        
        # Initialize output parser
        self.output_parser = StrOutputParser()
        
        # Create correction chain
        self.correction_chain = self._create_correction_chain()
        
        app_logger.info("GeminiTextCorrector initialized")
    
    def _create_correction_chain(self):
        """Create the correction chain with prompt template."""
        prompt_template = ChatPromptTemplate.from_template("""
You are an expert grammar and writing assistant. Your task is to correct grammatical errors in the given text while maintaining the original meaning and context.

INSTRUCTIONS:
1. Fix all grammatical errors including:
   - Subject-verb disagreements
   - Incorrect tense usage
   - Missing or incorrect articles (a, an, the)
   - Preposition errors
   - Sentence fragments
   - Run-on sentences
   - Comma splices

2. Maintain the original:
   - Meaning and intent
   - Style and tone
   - Technical terminology
   - Overall structure

3. Make minimal changes - only fix clear grammatical errors
4. Do not change the content or add new information
5. Preserve formatting and paragraph structure

DETECTED ERRORS:
{errors_summary}

ORIGINAL TEXT:
{original_text}

CONTEXT (surrounding text):
{context}

Please provide ONLY the corrected text without any explanations or commentary. The corrected text should be ready to use directly.
""")
        
        return prompt_template | self.llm | self.output_parser
    
    def _create_explanation_chain(self):
        """Create a chain to explain the corrections made."""
        prompt_template = ChatPromptTemplate.from_template("""
You are an expert grammar teacher. Analyze the corrections made to the text and provide a clear explanation.

ORIGINAL TEXT:
{original_text}

CORRECTED TEXT:
{corrected_text}

DETECTED ERRORS:
{errors_summary}

Please provide:
1. A list of specific changes made (be precise about what was changed)
2. The reasoning behind each change
3. Your confidence in the corrections (0.0 to 1.0)

Format your response as:
CHANGES:
- [Specific change 1]: [Reason]
- [Specific change 2]: [Reason]

REASONING:
[Overall explanation of the correction approach]

CONFIDENCE: [0.0-1.0]
""")
        
        return prompt_template | self.llm | self.output_parser
    
    def correct_text_with_errors(self, text: str, errors: List[GrammarError], context: str = "") -> CorrectionResult:
        """Correct text using detected grammar errors for guidance."""
        start_time = time.time()
        
        app_logger.info(f"Correcting text with {len(errors)} detected errors")
        
        # Prepare errors summary
        errors_summary = self._format_errors_for_prompt(errors)
        
        # Prepare context
        if not context:
            context = text  # Use the text itself as context if none provided
        
        try:
            # Get corrected text
            corrected_text = self.correction_chain.invoke({
                "original_text": text,
                "errors_summary": errors_summary,
                "context": context
            })
            
            # Clean up the corrected text
            corrected_text = corrected_text.strip()
            
            # Get explanation
            explanation_chain = self._create_explanation_chain()
            explanation = explanation_chain.invoke({
                "original_text": text,
                "corrected_text": corrected_text,
                "errors_summary": errors_summary
            })
            
            # Parse explanation
            changes_made, reasoning, confidence = self._parse_explanation(explanation)
            
            processing_time = time.time() - start_time
            
            result = CorrectionResult(
                original_text=text,
                corrected_text=corrected_text,
                changes_made=changes_made,
                confidence=confidence,
                reasoning=reasoning,
                processing_time=processing_time,
                errors_addressed=[error.rule_name for error in errors]
            )
            
            app_logger.info(f"Text correction completed in {processing_time:.2f}s with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            app_logger.error(f"Error correcting text: {e}")
            return CorrectionResult(
                original_text=text,
                corrected_text=text,  # Return original if correction fails
                changes_made=[],
                confidence=0.0,
                reasoning=f"Error during correction: {e}",
                processing_time=time.time() - start_time,
                errors_addressed=[]
            )
    
    def correct_text_simple(self, text: str, context: str = "") -> CorrectionResult:
        """Correct text without specific error guidance."""
        start_time = time.time()
        
        app_logger.info("Correcting text without specific error guidance")
        
        # Create a simple correction prompt
        simple_prompt = ChatPromptTemplate.from_template("""
You are an expert grammar assistant. Correct any grammatical errors in the following text while maintaining the original meaning and style.

ORIGINAL TEXT:
{text}

CONTEXT:
{context}

Provide ONLY the corrected text without explanations.
""")
        
        simple_chain = simple_prompt | self.llm | self.output_parser
        
        try:
            corrected_text = simple_chain.invoke({
                "text": text,
                "context": context if context else text
            })
            
            corrected_text = corrected_text.strip()
            
            # Simple change detection
            changes_made = self._detect_simple_changes(text, corrected_text)
            
            processing_time = time.time() - start_time
            
            result = CorrectionResult(
                original_text=text,
                corrected_text=corrected_text,
                changes_made=changes_made,
                confidence=0.8,  # Default confidence for simple corrections
                reasoning="General grammar correction without specific error analysis",
                processing_time=processing_time,
                errors_addressed=["General grammar check"]
            )
            
            app_logger.info(f"Simple text correction completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            app_logger.error(f"Error in simple text correction: {e}")
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                changes_made=[],
                confidence=0.0,
                reasoning=f"Error during correction: {e}",
                processing_time=time.time() - start_time,
                errors_addressed=[]
            )
    
    def _format_errors_for_prompt(self, errors: List[GrammarError]) -> str:
        """Format errors for the correction prompt."""
        if not errors:
            return "No specific errors detected."
        
        error_descriptions = []
        for i, error in enumerate(errors, 1):
            error_descriptions.append(
                f"{i}. {error.rule_name} ({error.severity}): "
                f"'{error.text_snippet}' - {error.description}"
            )
        
        return "\n".join(error_descriptions)
    
    def _parse_explanation(self, explanation: str) -> Tuple[List[str], str, float]:
        """Parse the explanation to extract changes, reasoning, and confidence."""
        changes_made = []
        reasoning = ""
        confidence = 0.7  # Default confidence
        
        try:
            # Extract changes
            changes_match = re.search(r'CHANGES:\s*(.*?)\s*REASONING:', explanation, re.DOTALL)
            if changes_match:
                changes_text = changes_match.group(1)
                changes_made = [
                    line.strip().lstrip('- ')
                    for line in changes_text.split('\n')
                    if line.strip() and line.strip().startswith('-')
                ]
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.*?)\s*CONFIDENCE:', explanation, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', explanation)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0.0, 1.0]
            
        except Exception as e:
            app_logger.warning(f"Error parsing explanation: {e}")
            reasoning = explanation  # Use full explanation as reasoning if parsing fails
        
        return changes_made, reasoning, confidence
    
    def _detect_simple_changes(self, original: str, corrected: str) -> List[str]:
        """Detect simple changes between original and corrected text."""
        if original == corrected:
            return ["No changes made"]
        
        changes = []
        
        # Simple word-level comparison
        original_words = original.split()
        corrected_words = corrected.split()
        
        if len(original_words) != len(corrected_words):
            changes.append(f"Length changed from {len(original_words)} to {len(corrected_words)} words")
        
        # Find differing words
        for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
            if orig != corr:
                changes.append(f"Changed '{orig}' to '{corr}' at position {i+1}")
        
        if not changes:
            changes.append("Minor formatting or punctuation changes")
        
        return changes[:5]  # Limit to first 5 changes
    
    def correct_chunks_with_context(self, chunks: List[Tuple[str, List[GrammarError]]], 
                                   full_text: str = "") -> List[CorrectionResult]:
        """Correct multiple text chunks while maintaining context awareness."""
        app_logger.info(f"Correcting {len(chunks)} text chunks with context awareness")
        
        results = []
        
        for i, (chunk_text, chunk_errors) in enumerate(chunks):
            app_logger.info(f"Correcting chunk {i+1}/{len(chunks)}")
            
            # Prepare context from surrounding chunks
            context = self._get_chunk_context(chunks, i, full_text)
            
            # Correct the chunk
            if chunk_errors:
                result = self.correct_text_with_errors(chunk_text, chunk_errors, context)
            else:
                result = self.correct_text_simple(chunk_text, context)
            
            results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        app_logger.info(f"Completed correction of {len(chunks)} chunks")
        return results
    
    def _get_chunk_context(self, chunks: List[Tuple[str, List[GrammarError]]], 
                          current_index: int, full_text: str = "") -> str:
        """Get context for a specific chunk from surrounding chunks."""
        context_parts = []
        
        # Add previous chunk
        if current_index > 0:
            prev_chunk = chunks[current_index - 1][0]
            context_parts.append(f"Previous: {prev_chunk[-200:]}")  # Last 200 chars
        
        # Add next chunk
        if current_index < len(chunks) - 1:
            next_chunk = chunks[current_index + 1][0]
            context_parts.append(f"Next: {next_chunk[:200]}")  # First 200 chars
        
        # If no surrounding chunks and full_text available, use that
        if not context_parts and full_text:
            context_parts.append(f"Full document context: {full_text[:500]}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def format_correction_report(self, results: List[CorrectionResult]) -> str:
        """Format correction results into a readable report."""
        if not results:
            return "No corrections performed."
        
        report = f"Text Correction Report\n{'=' * 50}\n\n"
        
        total_changes = sum(len(r.changes_made) for r in results if r.changes_made != ["No changes made"])
        avg_confidence = sum(r.confidence for r in results) / len(results)
        total_time = sum(r.processing_time for r in results)
        
        report += f"Summary:\n"
        report += f"- Total chunks processed: {len(results)}\n"
        report += f"- Total changes made: {total_changes}\n"
        report += f"- Average confidence: {avg_confidence:.2f}\n"
        report += f"- Total processing time: {total_time:.2f}s\n\n"
        
        for i, result in enumerate(results, 1):
            if result.changes_made and result.changes_made != ["No changes made"]:
                report += f"Chunk {i}:\n"
                report += f"Original: {result.original_text[:100]}...\n"
                report += f"Corrected: {result.corrected_text[:100]}...\n"
                report += f"Changes: {', '.join(result.changes_made[:3])}\n"
                report += f"Confidence: {result.confidence:.2f}\n"
                report += f"Errors addressed: {', '.join(result.errors_addressed)}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Natural language processing is a field of computer science and artificial intelligence. 
    It focus on the interaction between computers and human language. NLP helps machine to 
    understand, interpret, and generate human language in a valuable way. The applications 
    of natural language processing includes machine translation, sentiment analysis, and 
    text summarization. However, there is many challenges in this field such as ambiguity, 
    context understanding, and cultural nuances.
    """
    
    try:
        # Initialize corrector
        corrector = GeminiTextCorrector()
        
        # Test simple correction
        print("Testing simple correction...")
        result = corrector.correct_text_simple(sample_text)
        
        print(f"Original: {result.original_text[:100]}...")
        print(f"Corrected: {result.corrected_text[:100]}...")
        print(f"Changes: {result.changes_made}")
        print(f"Confidence: {result.confidence}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Test with grammar errors (would need to run grammar checker first)
        print("\nNote: For full testing with detected errors, run the grammar checker first")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure GOOGLE_API_KEY is set correctly")