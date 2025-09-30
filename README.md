# Data Curation for LLM Pipeline

A comprehensive system for web scraping, grammar checking with Context-Free Grammar (CFG), text correction using Google Gemini API, and RAG-based chatbot interaction.

## Features

🌐 **Web Scraping**: Extract clean text content from websites with deduplication  
📝 **CFG Grammar Checking**: Generate and apply custom grammar rules using LLM  
🔧 **Text Correction**: Context-aware text correction using Google Gemini API  
🗄️ **RAG Database**: Vector storage with ChromaDB for similarity search  
🤖 **Interactive Chatbot**: Conversational AI with chat history and RAG integration  
⚡ **CLI Interface**: Complete command-line interface for pipeline orchestration

## Architecture

```
URLs → Web Scraper → Text Chunks → CFG Checker → Error Detection
                                       ↓
RAG Database ← Text Correction ← Gemini API
     ↓
RAG Chatbot
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd data-curation-llm
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env and add your Google Gemini API key
   ```

4. **Initialize NLTK data** (automatic during first run):
   ```bash
   python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng'])"
   ```

## Quick Start

### 1. Complete Pipeline (Recommended)

Run the entire data curation pipeline with one command:

```bash
python main.py pipeline --urls-file example_urls.txt
```

This will:

- Scrape content from URLs
- Generate CFG grammar rules
- Check and correct grammar
- Build RAG database
- Make chatbot ready for interaction

### 2. Interactive Chatbot

Start chatting with your curated knowledge base:

```bash
python main.py chat
```

### 3. Step-by-Step Usage

**Scrape websites**:

```bash
python main.py scrape --urls-file example_urls.txt --output data/scraped.json
```

**Generate CFG rules**:

```bash
python main.py generate-cfg --input data/scraped.json --output data/cfg_rules.json
```

**Check grammar**:

```bash
python main.py check-grammar --input data/scraped.json --output data/errors.json
```

**Correct text**:

```bash
python main.py correct-text --input data/scraped.json --output data/corrected.json
```

**Build RAG database**:

```bash
python main.py build-rag --scraped data/scraped.json --corrected data/corrected.json
```

**Check system status**:

```bash
python main.py status
```

## Configuration

### Environment Variables (.env)

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
LOG_LEVEL=INFO
```

### Configuration File (config.yaml)

```yaml
web_scraper:
  max_pages: 100
  delay: 1
  timeout: 30
  user_agent: "DataCurationBot/1.0"

gemini:
  model: "gemini-1.5-flash"
  temperature: 0.3
  max_output_tokens: 1000

rag:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "all-MiniLM-L6-v2"
  collection_name: "curated_content"

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Project Structure

```
data-curation-llm/
├── main.py                 # CLI interface
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
├── example_urls.txt       # Sample URLs
├── .env                   # Environment variables
├── data/                  # Data storage
├── logs/                  # Application logs
├── chroma_db/            # Vector database
└── src/
    ├── scrapers/         # Web scraping modules
    ├── grammar/          # CFG generation and checking
    ├── correction/       # Text correction with Gemini
    ├── rag/             # RAG data management
    ├── chatbot/         # Interactive chatbot
    ├── config.py        # Configuration loader
    └── logger.py        # Logging setup
```

## Usage Examples

### Example URLs File

Create a file with URLs to scrape:

```
# Technology websites
https://openai.com/blog
https://blog.google/technology/ai/

# Educational content
https://en.wikipedia.org/wiki/Artificial_intelligence
```

### Pipeline with Custom Options

```bash
# Skip correction step and reset database
python main.py pipeline --urls-file urls.txt --skip-correction --reset-db

# Run only scraping and RAG building
python main.py pipeline --urls-file urls.txt --skip-cfg --skip-correction
```

### Direct URL Scraping

```bash
python main.py scrape --urls https://example.com --urls https://example2.com
```

### Advanced Grammar Checking

```bash
# Use custom CFG rules
python main.py check-grammar --input data/content.json --rules custom_rules.json

# Generate comprehensive error report
python main.py check-grammar --input data/content.json --output detailed_errors.json
```

## Components Deep Dive

### 1. Web Scraper (`src/scrapers/`)

- Extracts clean text from HTML pages
- Handles rate limiting and retries
- Deduplicates content using hashing
- Supports multiple content formats

### 2. CFG Grammar System (`src/grammar/`)

- **CFGRuleGenerator**: Creates grammar rules using LLM analysis
- **CFGGrammarChecker**: Applies rules to detect errors
- Supports custom rule patterns and confidence scoring

### 3. Text Correction (`src/correction/`)

- Uses Google Gemini API for context-aware corrections
- Preserves original meaning while fixing grammar
- Provides detailed reasoning for changes

### 4. RAG System (`src/rag/`)

- ChromaDB vector database for document storage
- Semantic similarity search with embeddings
- Supports both scraped and corrected content

### 5. Chatbot (`src/chatbot/`)

- Conversational interface with memory
- RAG-enhanced responses using relevant context
- Chat history and session management

## API Integration

### Google Gemini Setup

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Add to `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

### Supported Models

- `gemini-1.5-flash` (default, fast)
- `gemini-1.5-pro` (more capable, slower)

## Troubleshooting

### Common Issues

**"No API key found"**:

- Ensure `.env` file exists with valid `GOOGLE_API_KEY`
- Check API key permissions and quota

**"NLTK data not found"**:

- Run: `python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng'])"`

**"ChromaDB collection error"**:

- Delete `chroma_db/` folder and rebuild with `--reset-db`

**Rate limiting errors**:

- Increase delay in config.yaml: `web_scraper.delay: 2`
- Use smaller batch sizes for correction

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py pipeline --urls-file urls.txt
```

### Logs Location

Check `logs/app.log` for detailed operation logs.

## Performance Tips

1. **Batch Processing**: Use pipeline command for efficiency
2. **Rate Limiting**: Adjust delays for API calls
3. **Chunk Size**: Optimize `rag.chunk_size` for your content
4. **Model Selection**: Use `gemini-1.5-flash` for speed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://python.langchain.com/) for RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector database
- [Google Gemini](https://ai.google/discover/generativeai/) for text correction
- [Rich](https://rich.readthedocs.io/) for beautiful CLI interface
