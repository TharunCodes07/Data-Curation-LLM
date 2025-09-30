"""
Web scraper module for extracting clean text content from websites.
"""
import asyncio
import hashlib
import time
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup, Comment
import re
from dataclasses import dataclass
from pathlib import Path
import json

from src.config import config
from src.logger import app_logger


@dataclass
class ScrapedContent:
    """Data class for scraped content."""
    url: str
    title: str
    content: str
    word_count: int
    timestamp: float
    content_hash: str
    metadata: Dict


class WebScraper:
    """Web scraper for extracting clean text content from websites."""
    
    def __init__(self, delay: float = None, timeout: int = None):
        """Initialize the web scraper."""
        self.config = config.get_scraping_config()
        self.delay = delay or self.config.get('request_delay', 1)
        self.timeout = timeout or self.config.get('timeout', 30)
        self.max_pages = self.config.get('max_pages_per_site', 100)
        
        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': self.config.get('user_agent', 
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
            **self.config.get('headers', {})
        })
        
        # Content deduplication
        self.seen_hashes: Set[str] = set()
        
        app_logger.info(f"WebScraper initialized with delay={self.delay}s, timeout={self.timeout}s")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove common noise patterns
        noise_patterns = [
            r'Cookie\s+Policy.*?(?=\n|$)',
            r'Privacy\s+Policy.*?(?=\n|$)',
            r'Terms\s+of\s+Service.*?(?=\n|$)',
            r'Subscribe\s+to.*?newsletter.*?(?=\n|$)',
            r'Follow\s+us\s+on.*?(?=\n|$)',
            r'Share\s+this.*?(?=\n|$)',
        ]
        
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Clean up multiple punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract content from BeautifulSoup object."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip() if soup.title.string else ""
        
        # Try different content extraction strategies
        content = ""
        
        # Strategy 1: Look for main content areas
        main_selectors = [
            'main', 'article', '[role="main"]', '.main-content', 
            '.content', '.post-content', '.entry-content', '.article-content'
        ]
        
        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                content = main_element.get_text(separator=' ', strip=True)
                if len(content) > 200:  # Minimum content length
                    break
        
        # Strategy 2: If no main content found, extract from body
        if not content or len(content) < 200:
            # Remove navigation, sidebar, and other non-content elements
            for element in soup.find_all(['nav', 'sidebar', 'aside', 'footer', 'header']):
                element.decompose()
            
            body = soup.find('body')
            if body:
                content = body.get_text(separator=' ', strip=True)
        
        # Strategy 3: Fallback to all text
        if not content:
            content = soup.get_text(separator=' ', strip=True)
        
        # Clean the content
        content = self._clean_text(content)
        
        # Calculate content hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        return {
            'title': self._clean_text(title),
            'content': content,
            'word_count': len(content.split()) if content else 0,
            'content_hash': content_hash,
            'metadata': {
                'domain': urlparse(url).netloc,
                'path': urlparse(url).path,
                'content_length': len(content)
            }
        }
    
    def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape a single URL and return content."""
        try:
            app_logger.info(f"Scraping URL: {url}")
            
            # Add delay between requests
            time.sleep(self.delay)
            
            # Make request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            extracted = self._extract_content(soup, url)
            
            # Check for duplicates
            if extracted['content_hash'] in self.seen_hashes:
                app_logger.info(f"Duplicate content detected for {url}, skipping")
                return None
            
            # Check minimum content length
            if extracted['word_count'] < 50:
                app_logger.warning(f"Content too short for {url} (only {extracted['word_count']} words)")
                return None
            
            self.seen_hashes.add(extracted['content_hash'])
            
            return ScrapedContent(
                url=url,
                title=extracted['title'],
                content=extracted['content'],
                word_count=extracted['word_count'],
                timestamp=time.time(),
                content_hash=extracted['content_hash'],
                metadata=extracted['metadata']
            )
            
        except requests.exceptions.RequestException as e:
            app_logger.error(f"Network error scraping {url}: {e}")
            return None
        except Exception as e:
            app_logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_urls(self, urls: List[str]) -> List[ScrapedContent]:
        """Scrape multiple URLs and return all content."""
        results = []
        
        app_logger.info(f"Starting to scrape {len(urls)} URLs")
        
        for i, url in enumerate(urls, 1):
            if len(results) >= self.max_pages:
                app_logger.info(f"Reached maximum pages limit ({self.max_pages})")
                break
                
            app_logger.info(f"Processing {i}/{len(urls)}: {url}")
            
            content = self.scrape_url(url)
            if content:
                results.append(content)
                app_logger.info(f"Successfully scraped {url} ({content.word_count} words)")
            else:
                app_logger.warning(f"Failed to scrape or skipped {url}")
        
        app_logger.info(f"Scraping completed. Successfully scraped {len(results)} pages")
        return results
    
    def save_results(self, results: List[ScrapedContent], output_file: str = None):
        """Save scraping results to JSON file."""
        if not output_file:
            output_file = f"data/scraped_content_{int(time.time())}.json"
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = []
        for result in results:
            data.append({
                'url': result.url,
                'title': result.title,
                'content': result.content,
                'word_count': result.word_count,
                'timestamp': result.timestamp,
                'content_hash': result.content_hash,
                'metadata': result.metadata
            })
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        app_logger.info(f"Saved {len(results)} scraped contents to {output_file}")
        return output_file
    
    def load_results(self, input_file: str) -> List[ScrapedContent]:
        """Load scraping results from JSON file."""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            results.append(ScrapedContent(
                url=item['url'],
                title=item['title'],
                content=item['content'],
                word_count=item['word_count'],
                timestamp=item['timestamp'],
                content_hash=item['content_hash'],
                metadata=item['metadata']
            ))
        
        app_logger.info(f"Loaded {len(results)} scraped contents from {input_file}")
        return results


def scrape_from_file(url_file: str, output_file: str = None) -> List[ScrapedContent]:
    """Convenience function to scrape URLs from a text file."""
    # Read URLs from file
    with open(url_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Initialize scraper and scrape
    scraper = WebScraper()
    results = scraper.scrape_urls(urls)
    
    # Save results
    if output_file:
        scraper.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    # Example usage
    test_urls = [
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://docs.python.org/3/tutorial/index.html"
    ]
    
    scraper = WebScraper()
    results = scraper.scrape_urls(test_urls)
    
    if results:
        output_file = scraper.save_results(results)
        print(f"Scraped {len(results)} pages and saved to {output_file}")
        
        # Print summary
        total_words = sum(r.word_count for r in results)
        print(f"Total words scraped: {total_words}")
        
        for result in results:
            print(f"- {result.title} ({result.word_count} words): {result.url}")
    else:
        print("No content was successfully scraped")