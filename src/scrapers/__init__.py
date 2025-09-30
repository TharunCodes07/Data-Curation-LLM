"""
Scrapers package initialization.
"""
from .web_scraper import WebScraper, ScrapedContent, scrape_from_file

__all__ = ['WebScraper', 'ScrapedContent', 'scrape_from_file']