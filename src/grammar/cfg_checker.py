"""
Grammar checker that uses CFG rules to detect errors in text.
"""
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

from .cfg_generator import AdvancedCFGSystem, GrammarRule
from src.config import config
from src.logger import app_logger

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    app_logger.info("Downloading required NLTK data for grammar checking")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')


@dataclass
class GrammarError:
    """Represents a detected grammar error."""
    rule_id: str
    rule_name: str
    error_type: str
    severity: str
    description: str
    text_snippet: str
    start_pos: int
    end_pos: int
    suggestion: str
    confidence: float
    context: str


class CFGGrammarChecker:
    """Grammar checker using context-free grammar rules."""
    
    def __init__(self, rules_file: str = None, use_api: bool = False):
        """Initialize the grammar checker."""
        self.chunk_size = config.get('grammar.chunk_size', 500)
        self.context_window = config.get('grammar.context_window', 100)
        
        # Initialize advanced CFG system
        self.cfg_system = None
        self.rules = []
        
        if use_api:
            try:
                self.cfg_system = AdvancedCFGSystem(rules_file)
                self.rules = self.cfg_system.get_all_rules()
                app_logger.info("Advanced CFG system initialized with API support")
            except Exception as e:
                app_logger.warning(f"Could not initialize AdvancedCFGSystem: {e}")
                app_logger.info("Will use basic built-in rules only")
        elif rules_file:
            try:
                self.cfg_system = AdvancedCFGSystem(rules_file)
                self.rules = self.cfg_system.get_all_rules()
            except Exception as e:
                app_logger.warning(f"Could not load rules from {rules_file}: {e}")
        
        # Add basic rules if no CFG rules loaded
        if not self.rules:
            self.rules = self._get_basic_rules()
        
        app_logger.info(f"CFGGrammarChecker initialized with {len(self.rules)} CFG rules")
    
    def _get_basic_rules(self) -> List[GrammarRule]:
        """Get basic fallback rules if no CFG rules are loaded."""
        from .cfg_generator import GrammarRule
        
        basic_rules = [
            GrammarRule(
                rule_id="BASIC_001",
                name="subject_verb_disagreement",
                pattern=r"\b(he|she|it)\s+are\b",
                description="Singular subjects should use 'is' not 'are'",
                severity="high",
                confidence=0.9,
                category="agreement",
                examples=["he are running", "she are happy"]
            ),
            GrammarRule(
                rule_id="BASIC_002",
                name="double_negative",
                pattern=r"\b(don't|doesn't|didn't|won't|can't|couldn't|shouldn't|wouldn't)\s+\w*n't\b",
                description="Avoid double negatives",
                severity="medium",
                confidence=0.8,
                category="syntax",
                examples=["don't can't", "won't shouldn't"]
            ),
            GrammarRule(
                rule_id="BASIC_003",
                name="incomplete_sentence",
                pattern=r"^\s*[A-Z][^.!?]*[a-z]\s*$",
                description="Sentence appears incomplete (no ending punctuation)",
                severity="medium",
                confidence=0.7,
                category="punctuation",
                examples=["This is a sentence", "The quick brown fox"]
            )
        ]
        
        return basic_rules
    
    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get context around an error position."""
        context_start = max(0, start - self.context_window)
        context_end = min(len(text), end + self.context_window)
        
        context = text[context_start:context_end]
        
        # Mark the error position in context
        error_start = start - context_start
        error_end = end - context_start
        
        if error_start >= 0 and error_end <= len(context):
            context = (
                context[:error_start] + 
                "**" + context[error_start:error_end] + "**" + 
                context[error_end:]
            )
        
        return context.strip()
    
    def _apply_basic_rules(self, text: str) -> List[GrammarError]:
        """Apply basic hardcoded grammar rules."""
        errors = []
        
        # Rule 1: Subject-verb disagreement patterns
        patterns = [
            (r'\b(he|she|it)\s+are\b', "Subject-verb disagreement: use 'is' instead of 'are'"),
            (r'\b(they|we|you)\s+is\b', "Subject-verb disagreement: use 'are' instead of 'is'"),
            (r'\b(I)\s+are\b', "Subject-verb disagreement: use 'am' instead of 'are'"),
            (r'\b(was)\s+(they|we|you)\b', "Subject-verb disagreement: use 'were' instead of 'was'"),
        ]
        
        for pattern, suggestion in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                errors.append(GrammarError(
                    rule_id="BASIC_SUBJ_VERB",
                    rule_name="Basic Subject-Verb Agreement",
                    error_type="grammar",
                    severity="high",
                    description=suggestion,
                    text_snippet=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggestion=suggestion,
                    confidence=0.8,
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        # Rule 2: Missing articles
        article_patterns = [
            (r'\b(is|was)\s+([A-Z][a-z]+)\b', "Consider adding an article before the noun"),
            (r'\bwent\s+to\s+([a-z]+)\b', "Consider adding 'the' before the location"),
        ]
        
        for pattern, suggestion in article_patterns:
            for match in re.finditer(pattern, text):
                errors.append(GrammarError(
                    rule_id="BASIC_ARTICLES",
                    rule_name="Basic Article Usage",
                    error_type="grammar",
                    severity="medium",
                    description=suggestion,
                    text_snippet=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggestion=suggestion,
                    confidence=0.6,
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        # Rule 3: Common spelling/grammar mistakes
        common_mistakes = [
            (r'\bthere\s+is\s+many\b', "Use 'there are many' instead of 'there is many'"),
            (r'\bincludes\b(?=\s+[a-z])', "Check if subject is plural - should be 'include'"),
            (r'\bfocus\s+on\b', "Should be 'focuses on' for singular subject"),
            (r'\bmachine\s+to\s+understand\b', "Should be 'machines to understand' (plural)"),
        ]
        
        for pattern, suggestion in common_mistakes:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                errors.append(GrammarError(
                    rule_id="BASIC_COMMON",
                    rule_name="Common Grammar Mistakes",
                    error_type="grammar",
                    severity="medium",
                    description=suggestion,
                    text_snippet=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggestion=suggestion,
                    confidence=0.7,
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return errors
    
    def _apply_cfg_rules(self, text: str) -> List[GrammarError]:
        """Apply CFG rules to detect grammar errors."""
        errors = []
        
        for rule in self.rules:
            try:
                if not rule.pattern:
                    continue
                
                # Apply the rule pattern
                for match in re.finditer(rule.pattern, text, re.IGNORECASE):
                    error = GrammarError(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        error_type=rule.category,  # Use category instead of error_type
                        severity=rule.severity,
                        description=rule.description,
                        text_snippet=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        suggestion=rule.description,
                        confidence=rule.confidence,
                        context=self._get_context(text, match.start(), match.end())
                    )
                    errors.append(error)
                    
            except re.error as e:
                app_logger.warning(f"Invalid regex pattern in rule {rule.rule_id}: {e}")
                continue
        
        return errors
    
    def _apply_pos_based_rules(self, text: str) -> List[GrammarError]:
        """Apply POS-tag based grammar rules."""
        errors = []
        
        sentences = sent_tokenize(text)
        current_pos = 0
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Rule: Detect potential subject-verb disagreement using POS tags
            for i in range(len(pos_tags) - 1):
                word, pos = pos_tags[i]
                next_word, next_pos = pos_tags[i + 1]
                
                # Singular noun followed by plural verb
                if pos in ['NN', 'NNP'] and next_pos == 'VBP' and next_word.lower() == 'are':
                    word_start = current_pos + sentence.find(word + ' ' + next_word)
                    word_end = word_start + len(word + ' ' + next_word)
                    
                    errors.append(GrammarError(
                        rule_id="POS_SUBJ_VERB",
                        rule_name="POS Subject-Verb Agreement",
                        error_type="grammar",
                        severity="high",
                        description=f"Singular noun '{word}' should not be followed by plural verb 'are'",
                        text_snippet=f"{word} {next_word}",
                        start_pos=word_start,
                        end_pos=word_end,
                        suggestion=f"Use '{word} is' instead of '{word} are'",
                        confidence=0.75,
                        context=self._get_context(text, word_start, word_end)
                    ))
            
            current_pos += len(sentence) + 1  # +1 for sentence separator
        
        return errors
    
    def check_text(self, text: str) -> List[GrammarError]:
        """Check text for grammar errors using all available rules."""
        app_logger.info(f"Checking text of {len(text)} characters for grammar errors")
        
        all_errors = []
        
        # Apply basic hardcoded rules
        basic_errors = self._apply_basic_rules(text)
        all_errors.extend(basic_errors)
        app_logger.info(f"Found {len(basic_errors)} errors using basic rules")
        
        # Apply CFG rules
        cfg_errors = self._apply_cfg_rules(text)
        all_errors.extend(cfg_errors)
        app_logger.info(f"Found {len(cfg_errors)} errors using CFG rules")
        
        # Apply POS-based rules
        pos_errors = self._apply_pos_based_rules(text)
        all_errors.extend(pos_errors)
        app_logger.info(f"Found {len(pos_errors)} errors using POS rules")
        
        # Apply CYK parsing for structural analysis if CFG system is available
        if self.cfg_system:
            cyk_errors = self._apply_cyk_analysis(text)
            all_errors.extend(cyk_errors)
            app_logger.info(f"Found {len(cyk_errors)} errors using CYK parsing")
        
        # Remove duplicates and sort by position
        unique_errors = self._deduplicate_errors(all_errors)
        unique_errors.sort(key=lambda x: x.start_pos)
        
        app_logger.info(f"Total unique errors found: {len(unique_errors)}")
        return unique_errors
    
    def _deduplicate_errors(self, errors: List[GrammarError]) -> List[GrammarError]:
        """Remove duplicate errors that overlap or are too similar."""
        if not errors:
            return []
        
        # Sort by position
        errors.sort(key=lambda x: x.start_pos)
        
        unique_errors = []
        for error in errors:
            # Check if this error overlaps with any existing error
            is_duplicate = False
            for existing in unique_errors:
                # Check position overlap
                if (error.start_pos < existing.end_pos and 
                    error.end_pos > existing.start_pos):
                    # Keep the error with higher confidence
                    if error.confidence > existing.confidence:
                        unique_errors.remove(existing)
                        unique_errors.append(error)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_errors.append(error)
        
        return unique_errors
    
    def _apply_cyk_analysis(self, text: str) -> List[GrammarError]:
        """Apply CYK parsing to detect structural grammar errors."""
        errors = []
        
        # Split text into sentences
        sentences = self.cfg_system.split_into_sentences(text)
        
        for sentence in sentences:
            # Parse sentence using CYK algorithm
            is_valid, reason = self.cfg_system.parse_sentence_cyk(sentence)
            
            if not is_valid and reason != "Empty sentence":
                # Find the sentence position in the original text
                sentence_pos = text.find(sentence)
                if sentence_pos == -1:
                    sentence_pos = 0
                
                errors.append(GrammarError(
                    rule_id="CYK_PARSE",
                    rule_name="CYK Structural Analysis",
                    error_type="syntax",
                    severity="medium",
                    description=f"Sentence structure issue: {reason}",
                    text_snippet=sentence,
                    start_pos=sentence_pos,
                    end_pos=sentence_pos + len(sentence),
                    suggestion=f"Check sentence structure - {reason}",
                    confidence=0.7,
                    context=self._get_context(text, sentence_pos, sentence_pos + len(sentence))
                ))
        
        return errors
    
    def check_chunks(self, text: str) -> List[Tuple[str, List[GrammarError]]]:
        """Check text in chunks and return errors for each chunk."""
        chunks = self._split_into_chunks(text)
        results = []
        
        for i, chunk in enumerate(chunks):
            app_logger.info(f"Checking chunk {i+1}/{len(chunks)}")
            errors = self.check_text(chunk)
            results.append((chunk, errors))
        
        return results
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_error_summary(self, errors: List[GrammarError]) -> Dict[str, Any]:
        """Get summary statistics of detected errors."""
        if not errors:
            return {
                'total_errors': 0,
                'by_severity': {},
                'by_type': {},
                'by_rule': {}
            }
        
        summary = {
            'total_errors': len(errors),
            'by_severity': {},
            'by_type': {},
            'by_rule': {},
            'avg_confidence': sum(e.confidence for e in errors) / len(errors)
        }
        
        for error in errors:
            # By severity
            summary['by_severity'][error.severity] = summary['by_severity'].get(error.severity, 0) + 1
            
            # By type
            summary['by_type'][error.error_type] = summary['by_type'].get(error.error_type, 0) + 1
            
            # By rule
            summary['by_rule'][error.rule_name] = summary['by_rule'].get(error.rule_name, 0) + 1
        
        return summary
    
    def format_errors_report(self, errors: List[GrammarError]) -> str:
        """Format errors into a readable report."""
        if not errors:
            return "No grammar errors found."
        
        report = f"Grammar Check Report\n{'=' * 40}\n\n"
        report += f"Total errors found: {len(errors)}\n\n"
        
        # Group by severity
        high_errors = [e for e in errors if e.severity == 'high']
        medium_errors = [e for e in errors if e.severity == 'medium']
        low_errors = [e for e in errors if e.severity == 'low']
        
        for severity, error_list in [('HIGH', high_errors), ('MEDIUM', medium_errors), ('LOW', low_errors)]:
            if error_list:
                report += f"{severity} SEVERITY ERRORS ({len(error_list)}):\n"
                report += "-" * 30 + "\n"
                
                for i, error in enumerate(error_list, 1):
                    report += f"{i}. {error.rule_name}\n"
                    report += f"   Text: '{error.text_snippet}'\n"
                    report += f"   Issue: {error.description}\n"
                    report += f"   Suggestion: {error.suggestion}\n"
                    report += f"   Context: ...{error.context}...\n"
                    report += f"   Confidence: {error.confidence:.2f}\n\n"
        
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
    
    # Test with basic rules first
    checker = CFGGrammarChecker(use_api=False)
    errors = checker.check_text(sample_text)
    
    print(f"Found {len(errors)} grammar errors:")
    report = checker.format_errors_report(errors)
    print(report)
    
    summary = checker.get_error_summary(errors)
    print("Error Summary:")
    print(f"- Total: {summary['total_errors']}")
    print(f"- By severity: {summary['by_severity']}")
    print(f"- Average confidence: {summary['avg_confidence']:.2f}")