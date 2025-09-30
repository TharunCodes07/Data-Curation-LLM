"""
Advanced Context-Free Grammar system with CYK parsing and persistent rule management.
Integrates sophisticated CFG generation, lexicon management, and sentence parsing.
"""
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

import google.generativeai as genai

from src.config import config
from src.logger import app_logger

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    app_logger.info("Downloading required NLTK data")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')


@dataclass
class GrammarRule:
    """Represents a CFG grammar rule with enhanced metadata"""
    rule_id: str
    name: str
    pattern: str
    description: str
    severity: str  # 'high', 'medium', 'low'
    confidence: float
    category: str  # 'syntax', 'agreement', 'punctuation', etc.
    examples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class DynamicGrammarRule:
    """Represents a dynamic CFG production rule (e.g., NP VP -> S)"""
    left_hand_side: Tuple[str, ...]  # e.g., ('NP', 'VP')
    right_hand_side: str  # e.g., 'S'
    confidence: float
    frequency: int  # How often this rule was seen
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'left_hand_side': list(self.left_hand_side),
            'right_hand_side': self.right_hand_side,
            'confidence': self.confidence,
            'frequency': self.frequency
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicGrammarRule':
        """Create from dictionary"""
        return cls(
            left_hand_side=tuple(data['left_hand_side']),
            right_hand_side=data['right_hand_side'],
            confidence=data['confidence'],
            frequency=data['frequency']
        )


class AdvancedCFGSystem:
    """Advanced CFG system with CYK parsing, persistent lexicon, and rule generation"""
    
    def __init__(self, rules_file: Optional[str] = None):
        """Initialize advanced CFG system with centralized rule bank"""
        # Always use centralized rule bank
        self.rules_file = './data/centralized_rule_bank.json'
        self.rules: List[GrammarRule] = []
        
        # Persistent data files
        data_dir = Path('./data')
        data_dir.mkdir(parents=True, exist_ok=True)
        self.lexicon_file = data_dir / "persistent_lexicon.json"
        self.grammar_file = data_dir / "persistent_grammar.json"
        
        # Centralized dynamic banks
        self.centralized_dynamic_grammar_file = data_dir / "centralized_dynamic_grammar.json"
        self.centralized_dynamic_lexicon_file = data_dir / "centralized_dynamic_lexicon.json"
        
        # Initialize Google AI client
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.llm = genai.GenerativeModel(
            model_name=config.get('llm.gemini.model', 'gemini-2.5-flash')
        )
        
        # Initialize components with persistent data
        self.lexicon: Dict[str, Set[str]] = {}
        self.grammar: Dict[Tuple[str, ...], List[str]] = {}
        
        # Create data directory if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data or initialize base components
        self.load_persistent_data()
        
        app_logger.info(f"Advanced CFG System initialized with {len(self.lexicon)} words, {len(self.grammar)} rules")
    
    def load_persistent_data(self):
        """Load existing lexicon and grammar or initialize base components"""
        lexicon_loaded = self.load_lexicon()
        grammar_loaded = self.load_grammar()
        
        if not lexicon_loaded:
            app_logger.info("Initializing base lexicon...")
            self.initialize_base_lexicon()
        else:
            app_logger.info(f"Loaded existing lexicon: {len(self.lexicon)} words")
        
        if not grammar_loaded:
            app_logger.info("Initializing base grammar...")
            self.initialize_base_grammar()
        else:
            app_logger.info(f"Loaded existing grammar: {len(self.grammar)} rules")
        
        # Load existing CFG rules
        self.load_rules()
        
        # Load centralized dynamic banks
        self.load_centralized_dynamic_banks()
    
    def initialize_base_lexicon(self):
        """Initialize comprehensive base lexicon"""
        base_lexicon = {
            # Determiners
            'the': {'DET'}, 'a': {'DET'}, 'an': {'DET'}, 'this': {'DET'}, 'that': {'DET'}, 
            'my': {'DET'}, 'your': {'DET'}, 'his': {'DET'}, 'her': {'DET'}, 'its': {'DET'},
            'our': {'DET'}, 'their': {'DET'}, 'some': {'DET'}, 'many': {'DET'}, 'few': {'DET'},
            'all': {'DET'}, 'every': {'DET'}, 'each': {'DET'}, 'any': {'DET'}, 'no': {'DET'},
            
            # Nouns
            'dog': {'N'}, 'cat': {'N'}, 'man': {'N'}, 'woman': {'N'}, 'child': {'N'},
            'book': {'N'}, 'table': {'N'}, 'chair': {'N'}, 'house': {'N'}, 'car': {'N'},
            'computer': {'N'}, 'phone': {'N'}, 'school': {'N'}, 'work': {'N'}, 'home': {'N'},
            'time': {'N'}, 'day': {'N'}, 'year': {'N'}, 'water': {'N'}, 'food': {'N'},
            'apple': {'N'}, 'city': {'N'}, 'teacher': {'N'}, 'student': {'N'}, 'friend': {'N'},
            
            # Pronouns
            'i': {'PRP'}, 'you': {'PRP'}, 'he': {'PRP'}, 'she': {'PRP'}, 'it': {'PRP'},
            'we': {'PRP'}, 'they': {'PRP'}, 'me': {'PRP'}, 'him': {'PRP'}, 'her': {'PRP'},
            'us': {'PRP'}, 'them': {'PRP'}, 'who': {'PRP'}, 'what': {'PRP'}, 'which': {'PRP'},
            
            # Verbs
            'be': {'V', 'AUX'}, 'is': {'AUX'}, 'are': {'AUX'}, 'am': {'AUX'}, 'was': {'AUX'}, 'were': {'AUX'},
            'have': {'V', 'AUX'}, 'has': {'AUX'}, 'had': {'AUX'}, 'do': {'V', 'AUX'}, 'does': {'AUX'}, 'did': {'AUX'},
            'will': {'AUX'}, 'would': {'AUX'}, 'can': {'AUX'}, 'could': {'AUX'}, 'should': {'AUX'},
            'may': {'AUX'}, 'might': {'AUX'}, 'must': {'AUX'},
            'go': {'V'}, 'come': {'V'}, 'see': {'V'}, 'know': {'V'}, 'think': {'V'},
            'say': {'V'}, 'get': {'V'}, 'make': {'V'}, 'take': {'V'}, 'give': {'V'},
            'run': {'V'}, 'walk': {'V'}, 'eat': {'V'}, 'drink': {'V'}, 'sleep': {'V'},
            'runs': {'V'}, 'goes': {'V'}, 'sees': {'V'}, 'eats': {'V'}, 'going': {'V'}, 'eating': {'V'},
            
            # Adjectives
            'good': {'ADJ'}, 'bad': {'ADJ'}, 'big': {'ADJ'}, 'small': {'ADJ'}, 'new': {'ADJ'},
            'old': {'ADJ'}, 'young': {'ADJ'}, 'happy': {'ADJ'}, 'sad': {'ADJ'}, 'beautiful': {'ADJ'},
            'red': {'ADJ'}, 'blue': {'ADJ'}, 'green': {'ADJ'}, 'black': {'ADJ'}, 'white': {'ADJ'},
            
            # Adverbs
            'very': {'ADV'}, 'really': {'ADV'}, 'quite': {'ADV'}, 'too': {'ADV'}, 'so': {'ADV'},
            'quickly': {'ADV'}, 'slowly': {'ADV'}, 'carefully': {'ADV'}, 'well': {'ADV'}, 'badly': {'ADV'},
            'always': {'ADV'}, 'never': {'ADV'}, 'sometimes': {'ADV'}, 'often': {'ADV'}, 'usually': {'ADV'},
            'today': {'ADV'}, 'yesterday': {'ADV'}, 'tomorrow': {'ADV'}, 'now': {'ADV'}, 'then': {'ADV'},
            'please': {'ADV'},
            
            # Prepositions
            'in': {'P'}, 'on': {'P'}, 'at': {'P'}, 'to': {'P'}, 'from': {'P'}, 'with': {'P'},
            'by': {'P'}, 'for': {'P'}, 'of': {'P'}, 'about': {'P'}, 'under': {'P'}, 'over': {'P'},
            'through': {'P'}, 'during': {'P'}, 'before': {'P'}, 'after': {'P'}, 'between': {'P'},
            
            # Conjunctions
            'and': {'CC'}, 'or': {'CC'}, 'but': {'CC'}, 'so': {'CC'}, 'because': {'CC'},
            'if': {'CC'}, 'when': {'CC'}, 'while': {'CC'}, 'although': {'CC'}, 'since': {'CC'},
            
            # Additional words
            'thank': {'V'}, 'thanks': {'N'}
        }
        self.lexicon.update(base_lexicon)
    
    def initialize_base_grammar(self):
        """Initialize comprehensive base grammar rules in CNF"""
        base_grammar = {
            # Core sentence structures
            ('NP', 'VP'): ['S'],
            
            # Noun phrase rules
            ('DET', 'N'): ['NP'],
            ('ADJ', 'N'): ['NP'],
            ('DET', 'ADJ'): ['ADJP'],
            ('ADJP', 'N'): ['NP'],
            ('N',): ['NP'],
            ('PRP',): ['NP'],
            
            # Verb phrase rules
            ('V',): ['VP'],
            ('V', 'NP'): ['VP'],
            ('V', 'ADV'): ['VP'],
            ('ADV', 'V'): ['VP'],
            ('AUX', 'V'): ['VP'],
            ('AUX', 'ADJ'): ['VP'],
            ('AUX', 'NP'): ['VP'],
            ('AUX', 'VP'): ['VP'],
            ('VP', 'ADV'): ['VP'],
            ('ADV', 'VP'): ['VP'],
            ('AUX', 'PP'): ['VP'],
            ('V', 'PP'): ['VP'],
            ('VP', 'PP'): ['VP'],
            
            # Prepositional phrases
            ('P', 'NP'): ['PP'],
            ('P', 'N'): ['PP'],
            
            # Adjective phrases
            ('ADV', 'ADJ'): ['ADJP'],
            ('ADJ',): ['ADJP'],
            
            # Complex structures
            ('NP', 'PP'): ['NP'],
            ('VP', 'PP'): ['VP'],
            ('NP', 'ADJP'): ['NP'],
            
            # Coordination
            ('N', 'CC'): ['N_COORD'],
            ('N_COORD', 'N'): ['NP'],
            ('NP', 'CC'): ['NP_COORD'],
            ('NP_COORD', 'NP'): ['NP'],
            ('VP', 'CC'): ['VP_COORD'],
            ('VP_COORD', 'VP'): ['VP'],
            ('CC', 'NP'): ['NP'],
            ('NP', 'CC'): ['NP'],
            
            # Terminal productions (unary rules)
            ('DET',): ['DET'], ('N',): ['N'], ('V',): ['V'], ('ADJ',): ['ADJ'],
            ('ADV',): ['ADV'], ('P',): ['P'], ('CC',): ['CC'], ('PRP',): ['PRP'],
            ('AUX',): ['AUX'], ('PP',): ['VP']
        }
        self.grammar.update(base_grammar)
    
    def load_lexicon(self) -> bool:
        """Load persistent lexicon from file"""
        try:
            with open(self.lexicon_file, 'r', encoding='utf-8') as f:
                lexicon_data = json.load(f)
            
            # Convert lists back to sets
            for word, pos_list in lexicon_data.items():
                self.lexicon[word] = set(pos_list)
            
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            app_logger.error(f"Error loading lexicon: {e}")
            return False
    
    def load_grammar(self) -> bool:
        """Load persistent grammar from file"""
        try:
            with open(self.grammar_file, 'r', encoding='utf-8') as f:
                grammar_data = json.load(f)
            
            # Convert string keys back to tuples
            for key, productions in grammar_data.items():
                if ' -> ' in key:
                    rule_parts = key.split(' -> ')
                    if len(rule_parts) == 2:
                        rule_tuple = tuple(rule_parts[0].split())
                    else:
                        continue
                else:
                    rule_tuple = (key,)
                
                self.grammar[rule_tuple] = productions
            
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            app_logger.error(f"Error loading grammar: {e}")
            return False
    
    def save_persistent_data(self):
        """Save current lexicon and grammar to persistent files"""
        try:
            # Save lexicon
            lexicon_data = {word: list(pos_set) for word, pos_set in self.lexicon.items()}
            with open(self.lexicon_file, 'w', encoding='utf-8') as f:
                json.dump(lexicon_data, f, indent=2, ensure_ascii=False)
            
            # Save grammar
            grammar_data = {}
            for rule_tuple, productions in self.grammar.items():
                if len(rule_tuple) == 1:
                    key = rule_tuple[0]
                else:
                    key = ' '.join(rule_tuple)
                grammar_data[key] = productions
            
            with open(self.grammar_file, 'w', encoding='utf-8') as f:
                json.dump(grammar_data, f, indent=2, ensure_ascii=False)
            
            app_logger.info(f"Saved persistent data: {len(self.lexicon)} words, {len(self.grammar)} rules")
            
        except Exception as e:
            app_logger.error(f"Error saving persistent data: {e}")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences"""
        # Clean up the text
        text = text.strip()
        
        # Use NLTK sentence tokenizer
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to regex
            sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) > 1:  # Must have at least 2 words
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def update_lexicon_from_text(self, text: str) -> Dict[str, Set[str]]:
        """Update lexicon with new words from text using Gemini"""
        # Find words that are not in our current lexicon
        words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
        new_words = words_in_text - set(self.lexicon.keys())
        
        if not new_words:
            app_logger.info("Lexicon up-to-date (no new words)")
            return self.lexicon
        
        app_logger.info(f"Updating lexicon with {len(new_words)} new words...")
        
        try:
            # Only analyze new words to save API calls (limit to 50 words per call)
            new_words_list = list(new_words)[:50]
            new_words_text = ' '.join(new_words_list)
            
            prompt = f"""
            Analyze these English words and provide word-to-POS mappings:
            
            Words: "{new_words_text}"
            
            Use these POS tags only:
            - N: noun (dog, book, happiness, John, London)
            - V: verb (run, eat, think, is, have)  
            - ADJ: adjective (big, happy, red, beautiful)
            - ADV: adverb (quickly, very, well, today)
            - DET: determiner (the, a, this, my, every)
            - PRP: pronoun (he, she, it, they, mine, who)
            - AUX: auxiliary verb (is, has, will, can, must)
            - P: preposition (in, on, with, to, about)
            - CC: conjunction (and, but, or, because, if)
            
            Return ONLY a JSON object:
            {{"word1": ["POS1"], "word2": ["POS1", "POS2"], ...}}
            """
            
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Add new words to existing lexicon
                added_count = 0
                for word, pos_list in result.items():
                    word_clean = word.lower().strip('.,!?";')
                    if word_clean and word_clean not in self.lexicon:
                        self.lexicon[word_clean] = set(pos_list)
                        added_count += 1
                
                app_logger.info(f"Added {added_count} words to lexicon")
                return self.lexicon
            else:
                app_logger.warning("Failed to parse lexicon response, using fallback")
                return self._fallback_lexicon_generation(new_words)
                
        except Exception as e:
            app_logger.error(f"Error updating lexicon: {e}")
            return self._fallback_lexicon_generation(new_words)
    
    def _fallback_lexicon_generation(self, new_words: Set[str]) -> Dict[str, Set[str]]:
        """Fallback lexicon generation using heuristics"""
        for word in new_words:
            if word not in self.lexicon:
                # Simple heuristic assignment
                if word.endswith('ly'):
                    self.lexicon[word] = {'ADV'}
                elif word.endswith('ing') or word.endswith('ed') or word.endswith('s'):
                    self.lexicon[word] = {'V'}
                elif word.endswith('er') or word.endswith('est'):
                    self.lexicon[word] = {'ADJ'}
                else:
                    self.lexicon[word] = {'N'}  # Default to noun
        
        return self.lexicon
    
    def parse_sentence_cyk(self, sentence: str) -> Tuple[bool, str]:
        """Parse a single sentence using CYK algorithm"""
        words = sentence.split()
        n = len(words)
        
        if n == 0:
            return False, "Empty sentence"
        
        # Initialize CYK table
        table = [[set() for _ in range(n)] for _ in range(n)]
        
        # Fill diagonal (terminal productions)
        for i, word in enumerate(words):
            word_clean = word.lower().strip('.,!?";')
            
            if word_clean in self.lexicon:
                pos_tags = self.lexicon[word_clean]
                table[i][i].update(pos_tags)
                
                # Apply unary rules
                changed = True
                while changed:
                    changed = False
                    current_tags = table[i][i].copy()
                    for tag in current_tags:
                        if (tag,) in self.grammar:
                            for production in self.grammar[(tag,)]:
                                if production not in table[i][i]:
                                    table[i][i].add(production)
                                    changed = True
            else:
                # Unknown word - try to guess POS
                if word_clean.endswith('ly'):
                    table[i][i].add('ADV')
                elif word_clean.endswith('ing') or word_clean.endswith('ed'):
                    table[i][i].add('V')
                elif word[0].isupper():  # Proper noun
                    table[i][i].add('N')
                else:
                    table[i][i].add('N')  # Default to noun
                
                # Apply unary rules for guessed POS
                current_tags = table[i][i].copy()
                for tag in current_tags:
                    if (tag,) in self.grammar:
                        for production in self.grammar[(tag,)]:
                            table[i][i].add(production)
        
        # Fill upper triangle (binary productions)
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for k in range(i, j):
                    left_cell = table[i][k]
                    right_cell = table[k + 1][j]
                    
                    for left_tag in left_cell:
                        for right_tag in right_cell:
                            rule = (left_tag, right_tag)
                            if rule in self.grammar:
                                for production in self.grammar[rule]:
                                    table[i][j].add(production)
        
        # Check if sentence is valid
        is_valid = 'S' in table[0][n - 1] if n > 0 else False
        
        # Provide reason if invalid
        if not is_valid:
            unknown_words = [word for word in words if word.lower().strip('.,!?";') not in self.lexicon]
            if unknown_words:
                reason = f"Unknown words: {', '.join(unknown_words)}"
            else:
                reason = "Invalid sentence structure"
        else:
            reason = "Grammatically correct"
        
        return is_valid, reason
    
    def generate_rules_from_text(self, text: str, max_rules: int = 20) -> List[GrammarRule]:
        """Generate CFG rules by analyzing text patterns and add to centralized rule bank"""
        app_logger.info(f"Generating CFG rules from {len(text)} characters of text")
        
        # First update lexicon from text
        self.update_lexicon_from_text(text)
        
        # Generate dynamic grammar and lexicon rules
        app_logger.info("Generating dynamic CFG production rules...")
        self.update_dynamic_banks_from_text(text)
        
        # Create prompt for rule generation
        prompt = f"""
        Analyze the following text and generate Context-Free Grammar (CFG) rules for detecting grammatical errors.

        Text to analyze:
        {text[:5000]}

        Generate up to {max_rules} grammar rules in JSON format. Each rule should detect a specific type of grammatical error.

        Required JSON format:
        [
          {{
            "rule_id": "rule_001",
            "name": "subject_verb_agreement",
            "pattern": "grammatical_pattern_to_check",
            "description": "what this rule checks for",
            "severity": "high",
            "confidence": 0.85,
            "category": "agreement",
            "examples": ["example of error this catches", "another example"]
          }},
          ...
        ]

        Focus on common English grammar errors like:
        - Subject-verb agreement
        - Tense consistency
        - Article usage (a/an/the)
        - Pronoun agreement
        - Word order
        - Missing or incorrect punctuation
        - Incomplete sentences

        Return only the JSON array, no additional text.
        """
        
        try:
            response = self.llm.generate_content(prompt)
            app_logger.info(f"LLM response received: {len(response.text)} characters")
            app_logger.debug(f"Raw LLM response: {response.text[:500]}...")
            
            # Parse response and create GrammarRule objects
            rules = self._parse_rule_response(response.text)
            
            # Add to centralized rule bank (avoid duplicates)
            new_rules = []
            existing_names = {rule.name for rule in self.rules}
            existing_patterns = {rule.pattern for rule in self.rules}
            
            for rule in rules:
                # Check for duplicates by both name and pattern
                if rule.name not in existing_names and rule.pattern not in existing_patterns:
                    self.rules.append(rule)
                    new_rules.append(rule)
                    app_logger.info(f"Added new rule to bank: {rule.name}")
            
            app_logger.info(f"Added {len(new_rules)} new rules to centralized rule bank (total: {len(self.rules)})")
            
            # Save persistent data and centralized rule bank
            self.save_persistent_data()
            self.save_centralized_rule_bank()
            
            return new_rules
            
        except Exception as e:
            app_logger.error(f"Error generating CFG rules: {e}")
            return []
    
    def _parse_rule_response(self, response: str) -> List[GrammarRule]:
        """Parse LLM response and create GrammarRule objects"""
        try:
            app_logger.debug(f"Parsing response of length: {len(response)}")
            app_logger.debug(f"First 200 chars: {response[:200]}")
            
            # Clean response and extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            response = response.strip()
            app_logger.debug(f"Cleaned response length: {len(response)}")
            app_logger.debug(f"Cleaned response: {response[:300]}...")
            
            # Parse JSON
            rules_data = json.loads(response)
            
            # Create GrammarRule objects
            rules = []
            for i, rule_data in enumerate(rules_data):
                rule = GrammarRule(
                    rule_id=rule_data.get('rule_id', f'rule_{i+1:03d}'),
                    name=rule_data.get('name', ''),
                    pattern=rule_data.get('pattern', ''),
                    description=rule_data.get('description', ''),
                    severity=rule_data.get('severity', 'medium'),
                    confidence=rule_data.get('confidence', 0.5),
                    category=rule_data.get('category', 'syntax'),
                    examples=rule_data.get('examples', [])
                )
                
                if rule.name and rule.pattern:  # Only add valid rules
                    rules.append(rule)
            
            return rules
            
        except json.JSONDecodeError as e:
            app_logger.error(f"Failed to parse CFG rules JSON: {e}")
            return []
        except Exception as e:
            app_logger.error(f"Error parsing CFG rule response: {e}")
            return []
    
    def load_rules(self) -> bool:
        """Load existing rules from centralized rule bank"""
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            
            self.rules = []
            for rule_data in rules_data:
                rule = GrammarRule(**rule_data)
                self.rules.append(rule)
            
            app_logger.info(f"Loaded {len(self.rules)} CFG rules from centralized rule bank: {self.rules_file}")
            return True
            
        except FileNotFoundError:
            app_logger.info(f"No existing centralized rule bank found at {self.rules_file}")
            return False
        except Exception as e:
            app_logger.error(f"Error loading CFG rules from centralized rule bank: {e}")
            return False
    
    def save_centralized_rule_bank(self) -> str:
        """Save all rules to centralized rule bank"""
        try:
            # Convert rules to dictionaries
            rules_data = [rule.to_dict() for rule in self.rules]
            
            # Create directory if it doesn't exist
            Path(self.rules_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to centralized rule bank
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            app_logger.info(f"Saved {len(self.rules)} CFG rules to centralized rule bank: {self.rules_file}")
            return self.rules_file
            
        except Exception as e:
            app_logger.error(f"Error saving centralized rule bank: {e}")
            raise
    
    def save_rules(self, output_file: Optional[str] = None) -> str:
        """Save rules to JSON file (backward compatibility)"""
        if output_file:
            # If specific output file requested, save there too
            try:
                rules_data = [rule.to_dict() for rule in self.rules]
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(rules_data, f, indent=2, ensure_ascii=False)
                
                app_logger.info(f"Saved {len(self.rules)} CFG rules to {output_file}")
                return output_file
            except Exception as e:
                app_logger.error(f"Error saving CFG rules to {output_file}: {e}")
                raise
        
        # Always save to centralized rule bank
        return self.save_centralized_rule_bank()
    
    def get_all_rules(self) -> List[GrammarRule]:
        """Get all loaded rules"""
        return self.rules.copy()
    
    def get_rules_by_category(self, category: str) -> List[GrammarRule]:
        """Get rules filtered by category"""
        return [rule for rule in self.rules if rule.category == category]
    
    def get_rules_by_severity(self, severity: str) -> List[GrammarRule]:
        """Get rules filtered by severity"""
        return [rule for rule in self.rules if rule.severity == severity]
    
    def generate_dynamic_grammar_from_text(self, text: str) -> Dict[Tuple[str, ...], List[str]]:
        """Generate dynamic CFG production rules from text using POS tagging"""
        app_logger.info(f"Generating dynamic grammar rules from {len(text)} characters of text")
        
        # Tokenize and POS tag sentences
        sentences = sent_tokenize(text)
        new_grammar_rules = {}
        
        for sentence in sentences[:50]:  # Limit to first 50 sentences for efficiency
            try:
                words = word_tokenize(sentence.lower())
                pos_tags = pos_tag(words)
                
                # Convert NLTK POS tags to simplified tags
                simplified_tags = []
                for word, pos in pos_tags:
                    simplified_pos = self._simplify_pos_tag(pos)
                    simplified_tags.append(simplified_pos)
                    
                    # Update dynamic lexicon
                    if word not in self.lexicon:
                        self.lexicon[word] = set()
                    self.lexicon[word].add(simplified_pos)
                
                # Generate production rules from POS sequence
                if len(simplified_tags) >= 2:
                    # Create rules for different phrase structures
                    self._extract_grammar_patterns(simplified_tags, new_grammar_rules)
                    
            except Exception as e:
                app_logger.debug(f"Error processing sentence: {e}")
                continue
        
        # Merge with existing grammar
        for lhs, rhs_list in new_grammar_rules.items():
            if lhs in self.grammar:
                # Merge and deduplicate
                existing = set(self.grammar[lhs])
                for rhs in rhs_list:
                    if rhs not in existing:
                        self.grammar[lhs].append(rhs)
            else:
                self.grammar[lhs] = rhs_list
        
        app_logger.info(f"Generated {len(new_grammar_rules)} new dynamic grammar patterns")
        return new_grammar_rules
    
    def _simplify_pos_tag(self, nltk_pos: str) -> str:
        """Convert NLTK POS tags to simplified CFG categories"""
        pos_mapping = {
            # Nouns
            'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N',
            # Verbs
            'VB': 'V', 'VBD': 'V', 'VBG': 'V', 'VBN': 'V', 'VBP': 'V', 'VBZ': 'V',
            # Adjectives
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            # Adverbs
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV',
            # Pronouns
            'PRP': 'PRP', 'PRP$': 'PRP', 'WP': 'PRP', 'WP$': 'PRP',
            # Determiners
            'DT': 'DET', 'PDT': 'DET', 'WDT': 'DET',
            # Prepositions
            'IN': 'P', 'TO': 'P',
            # Conjunctions
            'CC': 'CC', 'IN': 'CC',
            # Auxiliaries/Modals
            'MD': 'AUX',
            # Particles
            'RP': 'PRT',
            # Interjections
            'UH': 'INTJ'
        }
        return pos_mapping.get(nltk_pos, 'OTHER')
    
    def _extract_grammar_patterns(self, pos_sequence: List[str], grammar_rules: Dict[Tuple[str, ...], List[str]]):
        """Extract CFG production rules from POS sequence"""
        if len(pos_sequence) < 2:
            return
        
        # Common CFG patterns to extract
        patterns_to_extract = [
            # Sentence patterns
            (['N', 'V'], 'S'),
            (['PRP', 'V'], 'S'),
            (['DET', 'N', 'V'], 'S'),
            (['N', 'AUX', 'ADJ'], 'S'),
            
            # Noun phrase patterns
            (['DET', 'N'], 'NP'),
            (['DET', 'ADJ', 'N'], 'NP'),
            (['ADJ', 'N'], 'NP'),
            (['PRP'], 'NP'),
            
            # Verb phrase patterns
            (['V'], 'VP'),
            (['V', 'N'], 'VP'),
            (['V', 'NP'], 'VP'),
            (['AUX', 'V'], 'VP'),
            (['AUX', 'ADJ'], 'VP'),
            
            # Prepositional phrase patterns
            (['P', 'N'], 'PP'),
            (['P', 'NP'], 'PP'),
            
            # Adjective phrase patterns
            (['ADV', 'ADJ'], 'ADJP'),
            (['ADJ'], 'ADJP')
        ]
        
        # Check for patterns in the sequence
        for i in range(len(pos_sequence)):
            for pattern, result in patterns_to_extract:
                if i + len(pattern) <= len(pos_sequence):
                    subsequence = pos_sequence[i:i+len(pattern)]
                    if subsequence == pattern:
                        lhs = tuple(pattern)
                        if lhs not in grammar_rules:
                            grammar_rules[lhs] = []
                        if result not in grammar_rules[lhs]:
                            grammar_rules[lhs].append(result)
    
    def save_centralized_dynamic_banks(self):
        """Save centralized dynamic grammar and lexicon banks"""
        try:
            # Save dynamic grammar
            grammar_data = {}
            for lhs, rhs_list in self.grammar.items():
                key = ' '.join(lhs) if isinstance(lhs, tuple) else str(lhs)
                grammar_data[key] = rhs_list
            
            with open(self.centralized_dynamic_grammar_file, 'w', encoding='utf-8') as f:
                json.dump(grammar_data, f, indent=2, ensure_ascii=False)
            
            # Save dynamic lexicon
            lexicon_data = {}
            for word, pos_set in self.lexicon.items():
                lexicon_data[word] = list(pos_set)
            
            with open(self.centralized_dynamic_lexicon_file, 'w', encoding='utf-8') as f:
                json.dump(lexicon_data, f, indent=2, ensure_ascii=False)
            
            app_logger.info(f"Saved centralized dynamic banks: {len(self.grammar)} grammar rules, {len(self.lexicon)} lexicon entries")
            
        except Exception as e:
            app_logger.error(f"Error saving centralized dynamic banks: {e}")
    
    def load_centralized_dynamic_banks(self):
        """Load centralized dynamic grammar and lexicon banks"""
        try:
            # Load dynamic grammar
            if self.centralized_dynamic_grammar_file.exists():
                with open(self.centralized_dynamic_grammar_file, 'r', encoding='utf-8') as f:
                    grammar_data = json.load(f)
                
                self.grammar = {}
                for key, rhs_list in grammar_data.items():
                    lhs = tuple(key.split())
                    self.grammar[lhs] = rhs_list
                
                app_logger.info(f"Loaded centralized dynamic grammar: {len(self.grammar)} rules")
            
            # Load dynamic lexicon
            if self.centralized_dynamic_lexicon_file.exists():
                with open(self.centralized_dynamic_lexicon_file, 'r', encoding='utf-8') as f:
                    lexicon_data = json.load(f)
                
                self.lexicon = {}
                for word, pos_list in lexicon_data.items():
                    self.lexicon[word] = set(pos_list)
                
                app_logger.info(f"Loaded centralized dynamic lexicon: {len(self.lexicon)} words")
            
        except Exception as e:
            app_logger.error(f"Error loading centralized dynamic banks: {e}")
    
    def update_dynamic_banks_from_text(self, text: str):
        """Update both dynamic grammar and lexicon from text"""
        app_logger.info("Updating centralized dynamic banks from text")
        
        # Load existing centralized banks
        self.load_centralized_dynamic_banks()
        
        # Generate new dynamic rules from text
        new_grammar = self.generate_dynamic_grammar_from_text(text)
        
        # Save updated centralized banks
        self.save_centralized_dynamic_banks()
        
        app_logger.info(f"Updated centralized dynamic banks with {len(new_grammar)} new patterns")
        return new_grammar
    
    def get_centralized_banks_summary(self) -> Dict[str, Any]:
        """Get summary of all centralized banks"""
        return {
            "pattern_based_rules": len(self.rules),
            "dynamic_grammar_rules": len(self.grammar),
            "dynamic_lexicon_entries": len(self.lexicon),
            "total_words_in_lexicon": sum(len(pos_set) for pos_set in self.lexicon.values()),
            "sample_grammar_rules": dict(list(self.grammar.items())[:5]),
            "sample_lexicon": dict(list(self.lexicon.items())[:10])
        }


# Maintain backward compatibility
class CFGRuleGenerator(AdvancedCFGSystem):
    """Backward compatibility alias for CFG rule generation"""
    pass