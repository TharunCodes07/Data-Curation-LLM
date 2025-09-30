"""
Configuration loader for the data curation pipeline.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        # Load environment variables
        load_dotenv()
        
        # Load YAML configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Override with environment variables where applicable
        self._load_env_overrides()
    
    def _load_env_overrides(self):
        """Load environment variable overrides."""
        # API Keys
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Override config with env vars if they exist
        if os.getenv('CHROMA_DB_PATH'):
            self.config['database']['chroma_db_path'] = os.getenv('CHROMA_DB_PATH')
        
        if os.getenv('RAG_COLLECTION_NAME'):
            self.config['database']['collection_name'] = os.getenv('RAG_COLLECTION_NAME')
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'scraping.max_pages_per_site')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_scraping_config(self) -> Dict[str, Any]:
        """Get scraping configuration."""
        return self.config.get('scraping', {})
    
    def get_grammar_config(self) -> Dict[str, Any]:
        """Get grammar configuration."""
        return self.config.get('grammar', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get('llm', {})
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self.config.get('rag', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})


# Global configuration instance
config = Config()