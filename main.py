"""
Main CLI interface for the Data Curation for LLM pipeline.
Orchestrates the entire workflow: scraping -> CFG generation -> grammar checking -> correction -> RAG storage -> chatbot.
"""
import click
import time
import json
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from src.logger import app_logger
from src.scrapers import WebScraper, ScrapedContent
from src.grammar import CFGRuleGenerator, CFGGrammarChecker, GrammarError
from src.correction import GeminiTextCorrector, CorrectionResult
from src.rag import RAGDataManager
from src.chatbot import RAGChatbot, create_interactive_chat_session

console = Console()


@click.group()
@click.version_option()
def cli():
    """Data Curation for LLM Pipeline - A comprehensive system for web scraping, 
    grammar checking, text correction, and RAG-based chatbot interaction."""
    pass


@cli.command()
@click.option('--urls-file', '-f', type=click.Path(exists=True), 
              help='File containing URLs to scrape (one per line)')
@click.option('--urls', '-u', multiple=True, help='URLs to scrape directly')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for scraped content')
@click.option('--delay', '-d', default=1, help='Delay between requests (seconds)')
@click.option('--max-pages', '-m', default=100, help='Maximum pages to scrape')
def scrape(urls_file: Optional[str], urls: tuple, output: Optional[str], 
          delay: int, max_pages: int):
    """Scrape content from websites."""
    console.print("[bold blue]🌐 Starting Web Scraping[/bold blue]")
    
    # Collect URLs
    all_urls = list(urls)
    if urls_file:
        with open(urls_file, 'r') as f:
            file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            all_urls.extend(file_urls)
    
    if not all_urls:
        console.print("[red]❌ No URLs provided. Use --urls or --urls-file[/red]")
        return
    
    console.print(f"📋 Found {len(all_urls)} URLs to scrape")
    
    # Initialize scraper
    scraper = WebScraper(delay=delay)
    scraper.max_pages = max_pages
    
    # Scrape with progress bar
    with Progress() as progress:
        task = progress.add_task("Scraping...", total=len(all_urls))
        
        results = []
        for i, url in enumerate(all_urls):
            progress.update(task, description=f"Scraping {i+1}/{len(all_urls)}")
            content = scraper.scrape_url(url)
            if content:
                results.append(content)
            progress.advance(task)
    
    # Save results
    if not output:
        output = f"data/scraped_content_{int(time.time())}.json"
    
    output_file = scraper.save_results(results, output)
    
    # Display summary
    table = Table(title="Scraping Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("URLs Attempted", str(len(all_urls)))
    table.add_row("Successfully Scraped", str(len(results)))
    table.add_row("Total Words", str(sum(r.word_count for r in results)))
    table.add_row("Output File", output_file)
    
    console.print(table)
    console.print(f"[green]✅ Scraping completed! Results saved to {output_file}[/green]")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input file with scraped content')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for generated CFG rules')
def generate_cfg(input: str, output: Optional[str]):
    """Generate CFG grammar rules from scraped content."""
    console.print("[bold blue]🔧 Generating CFG Grammar Rules[/bold blue]")
    
    # Load scraped content
    with open(input, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)
    
    console.print(f"📁 Loaded {len(scraped_data)} documents")
    
    # Combine all content
    all_text = ""
    for item in scraped_data:
        all_text += item.get('content', '') + "\n\n"
    
    console.print(f"📝 Total text length: {len(all_text)} characters")
    
    # Generate rules
    try:
        with console.status("[bold green]Generating CFG rules using LLM..."):
            generator = CFGRuleGenerator()
            rules = generator.generate_rules_from_text(all_text)
        
        # Save rules
        if not output:
            output = generator.save_rules()
        else:
            output = generator.save_rules(output)
        
        # Display results
        table = Table(title="Generated CFG Rules")
        table.add_column("Rule Name", style="cyan")
        table.add_column("Severity", style="yellow")
        table.add_column("Confidence", style="green")
        table.add_column("Description", style="white", max_width=50)
        
        for rule in rules[:10]:  # Show first 10 rules
            table.add_row(
                rule.name,
                rule.severity,
                f"{rule.confidence:.2f}",
                rule.description[:100] + "..." if len(rule.description) > 100 else rule.description
            )
        
        console.print(table)
        console.print(f"[green]✅ Generated {len(rules)} CFG rules and saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error generating CFG rules: {e}[/red]")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input file with content to check')
@click.option('--rules', '-r', type=click.Path(exists=True),
              help='CFG rules file (optional)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for grammar check results')
def check_grammar(input: str, rules: Optional[str], output: Optional[str]):
    """Check grammar using CFG rules."""
    console.print("[bold blue]📝 Checking Grammar[/bold blue]")
    
    # Load content
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    console.print(f"📁 Loaded {len(data)} documents")
    
    # Initialize grammar checker
    checker = CFGGrammarChecker(rules_file=rules, use_api=False)
    
    all_errors = []
    
    # Check each document
    with Progress() as progress:
        task = progress.add_task("Checking grammar...", total=len(data))
        
        for item in data:
            content = item.get('content', '')
            if content:
                errors = checker.check_text(content)
                all_errors.extend(errors)
            progress.advance(task)
    
    # Save results if requested
    if output:
        error_data = []
        for error in all_errors:
            error_data.append({
                'rule_id': error.rule_id,
                'rule_name': error.rule_name,
                'error_type': error.error_type,
                'severity': error.severity,
                'description': error.description,
                'text_snippet': error.text_snippet,
                'suggestion': error.suggestion,
                'confidence': error.confidence,
                'context': error.context
            })
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
    
    # Display summary
    summary = checker.get_error_summary(all_errors)
    
    table = Table(title="Grammar Check Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Errors Found", str(summary['total_errors']))
    table.add_row("Average Confidence", f"{summary.get('avg_confidence', 0):.2f}")
    
    for severity, count in summary.get('by_severity', {}).items():
        table.add_row(f"{severity.title()} Severity", str(count))
    
    console.print(table)
    
    if all_errors:
        console.print("\n[yellow]Top 5 Errors:[/yellow]")
        for i, error in enumerate(all_errors[:5], 1):
            console.print(f"{i}. [red]{error.rule_name}[/red]: '{error.text_snippet}' - {error.description}")
    
    console.print(f"[green]✅ Grammar check completed! Found {len(all_errors)} errors[/green]")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input file with content to correct')
@click.option('--grammar-errors', '-g', type=click.Path(exists=True),
              help='Grammar errors file from check-grammar command')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for corrected content')
def correct_text(input: str, grammar_errors: Optional[str], output: Optional[str]):
    """Correct text using Gemini API."""
    console.print("[bold blue]🔧 Correcting Text with Gemini[/bold blue]")
    
    # Load content
    with open(input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load grammar errors if provided
    errors_by_content = {}
    if grammar_errors:
        with open(grammar_errors, 'r', encoding='utf-8') as f:
            error_data = json.load(f)
        console.print(f"📋 Loaded {len(error_data)} grammar errors")
    
    # Initialize text corrector
    try:
        corrector = GeminiTextCorrector()
        console.print("🤖 Gemini text corrector initialized")
        
        corrections = []
        
        # Correct each document
        with Progress() as progress:
            task = progress.add_task("Correcting text...", total=len(data))
            
            for item in data:
                content = item.get('content', '')
                if content:
                    # Simple correction for now
                    result = corrector.correct_text_simple(content)
                    corrections.append(result)
                progress.advance(task)
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
        
        # Save results
        if not output:
            output = f"data/corrected_content_{int(time.time())}.json"
        
        correction_data = []
        for correction in corrections:
            correction_data.append({
                'original_text': correction.original_text,
                'corrected_text': correction.corrected_text,
                'changes_made': correction.changes_made,
                'confidence': correction.confidence,
                'reasoning': correction.reasoning,
                'processing_time': correction.processing_time
            })
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(correction_data, f, indent=2, ensure_ascii=False)
        
        # Display summary
        table = Table(title="Text Correction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        corrected_count = sum(1 for c in corrections if c.corrected_text != c.original_text)
        avg_confidence = sum(c.confidence for c in corrections) / len(corrections) if corrections else 0
        total_time = sum(c.processing_time for c in corrections)
        
        table.add_row("Total Documents", str(len(corrections)))
        table.add_row("Documents Corrected", str(corrected_count))
        table.add_row("Average Confidence", f"{avg_confidence:.2f}")
        table.add_row("Total Processing Time", f"{total_time:.1f}s")
        table.add_row("Output File", output)
        
        console.print(table)
        console.print(f"[green]✅ Text correction completed! Results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error correcting text: {e}[/red]")


@cli.command()
@click.option('--scraped', '-s', type=click.Path(exists=True),
              help='Scraped content file')
@click.option('--corrected', '-c', type=click.Path(exists=True),
              help='Corrected content file')
@click.option('--reset', is_flag=True, help='Reset the database before adding')
def build_rag(scraped: Optional[str], corrected: Optional[str], reset: bool):
    """Build RAG database from scraped and corrected content."""
    console.print("[bold blue]🗄️ Building RAG Database[/bold blue]")
    
    if not scraped and not corrected:
        console.print("[red]❌ Provide at least one of --scraped or --corrected[/red]")
        return
    
    # Initialize RAG manager
    rag_manager = RAGDataManager()
    
    if reset:
        if Confirm.ask("Are you sure you want to reset the database?"):
            rag_manager.reset_collection()
            console.print("[yellow]🧹 Database reset[/yellow]")
    
    # Add scraped content
    if scraped:
        console.print(f"📁 Loading scraped content from {scraped}")
        
        with open(scraped, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        
        # Convert to ScrapedContent objects
        scraped_content = []
        for item in scraped_data:
            content = ScrapedContent(
                url=item.get('url', ''),
                title=item.get('title', ''),
                content=item.get('content', ''),
                word_count=item.get('word_count', 0),
                timestamp=item.get('timestamp', time.time()),
                content_hash=item.get('content_hash', ''),
                metadata=item.get('metadata', {})
            )
            scraped_content.append(content)
        
        with console.status("[bold green]Adding scraped content to RAG database..."):
            result = rag_manager.add_scraped_content(scraped_content)
        
        console.print(f"[green]✅ Added {result['total_chunks']} chunks from {result['total_documents']} documents[/green]")
    
    # Add corrected content
    if corrected:
        console.print(f"📁 Loading corrected content from {corrected}")
        
        with open(corrected, 'r', encoding='utf-8') as f:
            corrected_data = json.load(f)
        
        # Convert to CorrectionResult objects
        correction_results = []
        for item in corrected_data:
            result = CorrectionResult(
                original_text=item.get('original_text', ''),
                corrected_text=item.get('corrected_text', ''),
                changes_made=item.get('changes_made', []),
                confidence=item.get('confidence', 0.0),
                reasoning=item.get('reasoning', ''),
                processing_time=item.get('processing_time', 0.0),
                errors_addressed=item.get('errors_addressed', [])
            )
            correction_results.append(result)
        
        with console.status("[bold green]Adding corrected content to RAG database..."):
            result = rag_manager.add_corrected_content(correction_results)
        
        console.print(f"[green]✅ Added {result['total_chunks_added']} corrected chunks[/green]")
    
    # Display final stats
    stats = rag_manager.get_collection_stats()
    
    table = Table(title="RAG Database Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Documents", str(stats['total_documents']))
    table.add_row("Embedding Model", stats['embedding_model'])
    table.add_row("Chunk Size", str(stats['chunk_size']))
    table.add_row("Database Path", stats['db_path'])
    
    console.print(table)
    console.print("[green]✅ RAG database built successfully![/green]")


@cli.command()
def chat():
    """Start interactive RAG chatbot session."""
    console.print("[bold blue]🤖 Starting RAG Chatbot[/bold blue]")
    
    try:
        # Initialize RAG manager and check stats
        rag_manager = RAGDataManager()
        stats = rag_manager.get_collection_stats()
        
        if stats.get('total_documents', 0) == 0:
            console.print("[yellow]⚠️  No documents found in RAG database.[/yellow]")
            if Confirm.ask("Continue with empty database? (responses will be limited)"):
                pass  # Continue with empty database
            else:
                console.print("Run 'build-rag' command first to populate the database.")
                return
        else:
            console.print(f"[green]✅ Found {stats['total_documents']} documents in knowledge base[/green]")
        
        # Start interactive session
        create_interactive_chat_session(rag_manager)
        
    except Exception as e:
        console.print(f"[red]❌ Error starting chatbot: {e}[/red]")


@cli.command()
@click.option('--urls-file', '-f', type=click.Path(exists=True), required=True,
              help='File containing URLs to scrape')
@click.option('--skip-scraping', is_flag=True, help='Skip scraping step')
@click.option('--skip-cfg', is_flag=True, help='Skip CFG generation step')
@click.option('--skip-correction', is_flag=True, help='Skip text correction step')
@click.option('--reset-db', is_flag=True, help='Reset RAG database')
def pipeline(urls_file: str, skip_scraping: bool, skip_cfg: bool, 
            skip_correction: bool, reset_db: bool):
    """Run the complete data curation pipeline with centralized rule bank and smart correction."""
    console.print(Panel.fit("[bold green]🚀 Smart Data Curation Pipeline[/bold green]", 
                           title="Starting", border_style="green"))
    
    timestamp = int(time.time())
    scraped_file = f"data/pipeline_scraped_{timestamp}.json"
    corrected_file = f"data/pipeline_corrected_{timestamp}.json"
    
    # Use centralized rule bank (no more separate files)
    centralized_rule_bank = "data/centralized_rule_bank.json"
    
    try:
        # Step 1: Web Scraping
        if not skip_scraping:
            console.print("\n[bold cyan]Step 1: Web Scraping[/bold cyan]")
            
            with open(urls_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            scraper = WebScraper()
            results = scraper.scrape_urls(urls)
            scraper.save_results(results, scraped_file)
            
            console.print(f"[green]✅ Scraped {len(results)} pages → {scraped_file}[/green]")
        
        # Step 2: Add to Centralized CFG Rule Bank
        if not skip_cfg and not skip_scraping:
            console.print("\n[bold cyan]Step 2: Adding to Centralized CFG Rule Bank[/bold cyan]")
            
            # Load scraped content for CFG generation
            with open(scraped_file, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            all_text = "\n\n".join(item.get('content', '') for item in scraped_data)
            
            generator = CFGRuleGenerator()
            # Check current rule bank size
            current_rules = len(generator.get_all_rules())
            
            # Generate and add new rules to centralized bank
            new_rules = generator.generate_rules_from_text(all_text)
            total_rules = len(generator.get_all_rules())
            
            console.print(f"[green]✅ Added {len(new_rules)} new rules to centralized bank (total: {total_rules}, was: {current_rules})[/green]")
            console.print(f"[cyan]📁 Centralized rule bank: {centralized_rule_bank}[/cyan]")
        
        # Step 3: Smart Text Correction (Only CFG Error Chunks)
        if not skip_correction and not skip_scraping:
            console.print("\n[bold cyan]Step 3: Smart Text Correction (CFG Error Chunks Only)[/bold cyan]")
            
            from src.correction.smart_corrector import SmartTextCorrector
            corrector = SmartTextCorrector()
            
            # Load content and apply smart correction
            with open(scraped_file, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            # Apply smart correction - only chunks with CFG errors get sent to LLM
            corrected_data = corrector.correct_scraped_data(scraped_data)
            
            # Get correction summary
            summary = corrector.get_correction_summary(corrected_data)
            
            # Save corrected data
            with open(corrected_file, 'w', encoding='utf-8') as f:
                json.dump(corrected_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✅ Smart correction completed → {corrected_file}[/green]")
            console.print(f"[cyan]📊 Efficiency: {summary['chunks_corrected']}/{summary['total_chunks']} chunks corrected ({summary['efficiency_percentage']}%)[/cyan]")
            console.print(f"[blue]💰 LLM calls saved: {summary['llm_calls_saved']}[/blue]")
        
        # Step 4: Build RAG Database
        console.print("\n[bold cyan]Step 4: Building RAG Database[/bold cyan]")
        
        rag_manager = RAGDataManager()
        
        if reset_db:
            rag_manager.reset_collection()
        
        # Add scraped content if available
        if not skip_scraping and Path(scraped_file).exists():
            with open(scraped_file, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            scraped_content = [
                ScrapedContent(
                    url=item.get('url', ''),
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    word_count=item.get('word_count', 0),
                    timestamp=item.get('timestamp', time.time()),
                    content_hash=item.get('content_hash', ''),
                    metadata=item.get('metadata', {})
                )
                for item in scraped_data
            ]
            
            rag_manager.add_scraped_content(scraped_content)
        
        # Add corrected content if available
        if not skip_correction and Path(corrected_file).exists():
            with open(corrected_file, 'r', encoding='utf-8') as f:
                corrected_data = json.load(f)
            
            correction_results = [
                CorrectionResult(
                    original_text=item.get('original_text', ''),
                    corrected_text=item.get('corrected_text', ''),
                    changes_made=item.get('changes_made', []),
                    confidence=item.get('confidence', 0.0),
                    reasoning=item.get('reasoning', ''),
                    processing_time=item.get('processing_time', 0.0),
                    errors_addressed=item.get('errors_addressed', [])
                )
                for item in corrected_data
            ]
            
            rag_manager.add_corrected_content(correction_results)
        
        stats = rag_manager.get_collection_stats()
        console.print(f"[green]✅ RAG database built with {stats['total_documents']} documents[/green]")
        
        # Final summary
        console.print(Panel.fit(
            f"[bold green]Smart Pipeline Complete![/bold green]\n\n"
            f"📁 Files created:\n"
            f"• Scraped content: {scraped_file if not skip_scraping else 'Skipped'}\n"
            f"• Centralized rule bank: {centralized_rule_bank if not skip_cfg else 'Skipped'}\n"
            f"• Smart corrected content: {corrected_file if not skip_correction else 'Skipped'}\n"
            f"• RAG database: {stats['total_documents']} documents\n\n"
            f"🤖 Ready for chatbot interaction!",
            title="Success",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Pipeline failed: {e}[/red]")
        app_logger.error(f"Pipeline error: {e}")


@cli.command()
@click.option('--urls', '-u', multiple=True, help='URLs to add to database')
@click.option('--urls-file', '-f', type=click.Path(exists=True), help='File containing URLs to add')
def add_data(urls: tuple, urls_file: Optional[str]):
    """Add new data to the system through the complete pipeline: scrape -> CFG rules -> check -> correct -> RAG."""
    console.print(Panel.fit("[bold green]📥 Adding New Data to System[/bold green]", 
                           title="Add Data", border_style="green"))
    
    if not urls and not urls_file:
        console.print("[red]❌ Please provide URLs either via --urls or --urls-file[/red]")
        return
    
    try:
        from src.correction.smart_corrector import SmartTextCorrector
        from src.scrapers.web_scraper import WebScraper
        from src.grammar.cfg_generator import CFGRuleGenerator
        from src.rag.data_manager import RAGDataManager
        
        # Step 1: Collect URLs
        all_urls = list(urls) if urls else []
        if urls_file:
            with open(urls_file, 'r', encoding='utf-8') as f:
                file_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                all_urls.extend(file_urls)
        
        console.print(f"📝 Processing {len(all_urls)} URLs")
        
        # Step 2: Scrape new content
        console.print("\n[bold cyan]Step 1: Scraping New Content[/bold cyan]")
        scraper = WebScraper()
        scraped_results = []
        
        with console.status("[bold green]Scraping..."):
            for url in all_urls:
                try:
                    result = scraper.scrape_url(url)
                    if result and result.get('content'):
                        scraped_results.append(result)
                        console.print(f"✅ {url}")
                    else:
                        console.print(f"⚠️  {url} - No content")
                except Exception as e:
                    console.print(f"❌ {url} - Error: {e}")
        
        if not scraped_results:
            console.print("[red]❌ No content scraped successfully[/red]")
            return
        
        console.print(f"[green]✅ Scraped {len(scraped_results)} pages[/green]")
        
        # Step 3: Add to centralized CFG rule bank
        console.print("\n[bold cyan]Step 2: Adding to CFG Rule Bank[/bold cyan]")
        generator = CFGRuleGenerator()
        current_rules = len(generator.get_all_rules())
        
        # Combine all text for rule generation
        all_text = "\n\n".join(item.get('content', '') for item in scraped_results)
        new_rules = generator.generate_rules_from_text(all_text)
        total_rules = len(generator.get_all_rules())
        
        console.print(f"[green]✅ Added {len(new_rules)} new rules (total: {total_rules}, was: {current_rules})[/green]")
        
        # Step 4: Smart correction (only error chunks)
        console.print("\n[bold cyan]Step 3: Smart Text Correction[/bold cyan]")
        corrector = SmartTextCorrector()
        corrected_data = corrector.correct_scraped_data(scraped_results)
        
        # Get correction summary
        summary = corrector.get_correction_summary(corrected_data)
        console.print(f"[green]✅ Corrected {summary['chunks_corrected']}/{summary['total_chunks']} chunks ({summary['efficiency_percentage']}% efficiency)[/green]")
        console.print(f"[blue]💰 Saved {summary['llm_calls_saved']} LLM calls[/blue]")
        
        # Step 5: Add to RAG database
        console.print("\n[bold cyan]Step 4: Adding to RAG Database[/bold cyan]")
        rag_manager = RAGDataManager()
        
        # Add both original and corrected content
        documents_added = 0
        for original, corrected in zip(scraped_results, corrected_data):
            # Use corrected content if available, otherwise original
            final_content = corrected.get('content', original.get('content', ''))
            
            if final_content.strip():
                rag_manager.add_document(
                    content=final_content,
                    metadata={
                        'title': corrected.get('title', original.get('title', '')),
                        'url': corrected.get('url', original.get('url', '')),
                        'source': 'add_data_command',
                        'had_corrections': corrected.get('correction_stats', {}).get('total_errors_found', 0) > 0
                    }
                )
                documents_added += 1
        
        console.print(f"[green]✅ Added {documents_added} documents to RAG database[/green]")
        
        # Final stats
        final_stats = rag_manager.get_collection_stats()
        console.print(Panel.fit(
            f"[bold green]Data Successfully Added![/bold green]\n\n"
            f"📊 Summary:\n"
            f"• URLs processed: {len(all_urls)}\n"
            f"• Content scraped: {len(scraped_results)} pages\n"
            f"• New CFG rules: {len(new_rules)} (total: {total_rules})\n"
            f"• Text correction efficiency: {summary['efficiency_percentage']}%\n"
            f"• Documents in RAG database: {final_stats} total\n\n"
            f"🤖 New data ready for chatbot queries!",
            title="Success",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Failed to add data: {e}[/red]")
        app_logger.error(f"Add data error: {e}")

@cli.command()
def status():
    """Show system status and statistics."""
    console.print("[bold blue]📊 System Status[/bold blue]")
    
    # Check RAG database
    try:
        rag_manager = RAGDataManager()
        stats = rag_manager.get_collection_stats()
        
        table = Table(title="RAG Database Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Documents", str(stats.get('total_documents', 0)))
        table.add_row("Embedding Model", stats.get('embedding_model', 'Unknown'))
        table.add_row("Database Path", stats.get('db_path', 'Unknown'))
        
        if 'source_types' in stats:
            for source_type, count in stats['source_types'].items():
                table.add_row(f"{source_type.title()} Documents", str(count))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]❌ Error checking RAG database: {e}[/red]")
    
    # Check CFG rules
    try:
        generator = CFGRuleGenerator()
        rules = generator.get_all_rules()
        
        if rules:
            console.print(f"\n[green]✅ CFG Rules: {len(rules)} rules loaded[/green]")
        else:
            console.print(f"\n[yellow]⚠️  CFG Rules: No rules found[/yellow]")
    
    except Exception as e:
        console.print(f"\n[red]❌ Error checking CFG rules: {e}[/red]")


if __name__ == '__main__':
    cli()