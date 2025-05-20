import asyncio
import base64
import csv
import logging
from pathlib import Path
from typing import List, Set

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

# for jupyter only
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraping.log'),
        logging.StreamHandler()
    ]
)


class Scraper_crawl4ai:
    def __init__(self, urls_file: str, log_dir: str, output_dir: str):
        self.urls_file = urls_file
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.successful_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.processed_urls: List[tuple] = []  # List of (filename, url) tuples
        self.queue = []
        self.fail_check = 0

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
    async def load_urls(self):
        """Load URLs from file into queue, skipping those with existing files"""
        # Get list of existing files
        existing_files = set(f.stem for f in self.output_dir.glob('*.md'))
        
        with open(self.urls_file, 'r') as f:
            # specify the number of urls to process
            urls = [line.strip() for line in f if line.strip()]
        
        skipped = 0
        queued = 0
        
        for url in urls:
            # Generate the expected filename using same logic as in process_url
            path = url.split('://', 1)[-1].split('/', 1)[-1]
            if path.startswith(('video/', 'node/')):
                skipped += 1
                continue
            parts = [p for p in path.split('/') if p]
            expected_filename = self.sanitize_filename("_".join(parts))
            
            # Check if file already exists
            if expected_filename in existing_files:
                skipped += 1
                continue

            # Queue URL if file doesn't exist
            self.queue.append(url)
            queued += 1
        
        logging.info(f"Loaded {queued} URLs into queue, skipped {skipped} existing files")
    
    def sanitize_filename(self,filename: str) -> str:
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    def convert_url_to_filename(self, url: str) -> str:
        path = url.split('://', 1)[-1].split('/', 1)[-1]

        # Split by '/' and filter out empty parts
        parts = [p for p in path.split('/') if p]

        return self.sanitize_filename("_".join(parts))

    # https://docs.crawl4ai.com/api/parameters/
    async def process_url_with_crawl4ai(self):
        async with AsyncWebCrawler() as crawler:
            run_config = CrawlerRunConfig(
                only_text=True,          # If True, tries to remove non-text elements
                target_elements=['main'],
                excluded_tags=['a', 'img', 'nav'], # Remove entire tag blocks
                excluded_selector='.latest-article-slider, #block-views-block-recent-videos-recent-videos, .product-recipes-container',
                remove_overlay_elements=True,
                screenshot=True,
                stream=True
            )

            dispatcher = MemoryAdaptiveDispatcher(
                check_interval=20.0,
                max_session_permit=5,
                rate_limiter=RateLimiter(
                    base_delay=(3, 7),
                    max_delay=40.0
                ),
            )

            async for result in await crawler.arun_many(
                self.queue, 
                config=run_config,
                dispatcher=dispatcher
            ):

                if result.success:
                    filename = self.convert_url_to_filename(result.url)
                    # If empty or just a slash, use 'index'
                    if not filename:
                        filename = 'index'
                    filepath = self.output_dir / f"{filename}.md"
                    
                    # Save content
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result.markdown)
                    
                    if result.screenshot:
                        # Decode base64 string to bytes
                        screenshot_bytes = base64.b64decode(result.screenshot)
                        screenshot_path = Path(f'../data/screenshot/{filename}.png')
                        with open(screenshot_path, "wb") as f:
                            f.write(screenshot_bytes)
                        print(f"Screenshot saved to: {screenshot_path}")
                    
                    self.processed_urls.append((f"{filename}.md", result.url))

                    
                    self.successful_urls.add(result.url)
                    logging.info(f"Successfully processed {result.url}")
                    # # simple mechanism to stop the crawler if it's failing repeatedly
                    # if self.fail_check > 0:
                    #     self.fail_check -= 1
                else:
                    logging.error(f"Failed to process {result.url}")
                    self.failed_urls.add(result.url)
                    # self.fail_check += 1
                    # if self.fail_check > 10:
                    #     logging.error("Failed to process 10 urls, stopping")
                    #     raise Exception("Failed to process 10 urls, stopping")

    async def run(self):
        """Run the scraper"""

        await self.process_url_with_crawl4ai()
    
    def save_processed_urls(self):
        """Save processed URLs to CSV file"""
        with open(self.log_dir / 'processed_urls.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'url'])
            writer.writerows(self.processed_urls)
            logging.info(f"Saved {len(self.processed_urls)} processed URLs")
        
        with open(self.log_dir / 'successful_urls.txt', 'w') as f:
            for url in self.successful_urls:
                f.write(f"{url}\n")
        
        # Save failed URLs
        with open(self.log_dir / 'failed_urls.txt', 'w') as f:
            for url in self.failed_urls:
                f.write(f"{url}\n")
        
        logging.info(f"Processed {len(self.successful_urls)} URLs successfully")
        logging.info(f"Failed to process {len(self.failed_urls)} URLs")


async def main():
    scraper2 = Scraper_crawl4ai(
        urls_file='../data/english_urls.txt',
        output_dir='../data/site',
        log_dir='./logs',
    )
    await scraper2.load_urls()
    await scraper2.run()
    scraper2.save_processed_urls()

if __name__ == "__main__":
    asyncio.run(main())