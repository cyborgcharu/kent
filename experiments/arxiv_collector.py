# arxiv_collector.py

import arxiv
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_collection.log'),
        logging.StreamHandler()
    ]
)

def fetch_papers(start_year=2015, end_year=2024, batch_size=100):
    """
    Fetch ML/AI papers with detailed logging and error handling
    """
    logging.info(f"Starting paper collection from {start_year} to {end_year}")
    
    client = arxiv.Client()
    papers = []
    
    # Build search query
    query = 'cat:cs.AI OR cat:cs.LG OR cat:cs.NE OR cat:stat.ML'
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=batch_size,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        logging.info(f"Search query: {query}")
        logging.info(f"Attempting to fetch {batch_size} papers")
        
        for i, paper in enumerate(client.results(search)):
            try:
                paper_data = {
                    'id': paper.entry_id,
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [author.name for author in paper.authors],
                    'categories': paper.categories,
                    'date': paper.published,
                }
                papers.append(paper_data)
                
                if i % 10 == 0:  # Log progress every 10 papers
                    logging.info(f"Processed {i} papers")
                
                time.sleep(0.33)  # Rate limiting - 3 requests per second
                
            except Exception as e:
                logging.error(f"Error processing paper: {e}")
                continue
        
        # Save results periodically
        df = pd.DataFrame(papers)
        output_file = f'arxiv_papers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)
        logging.info(f"Saved {len(papers)} papers to {output_file}")
        
        return df
    
    except Exception as e:
        logging.error(f"Major error in paper collection: {e}")
        return pd.DataFrame(papers)  # Return what we have so far

if __name__ == "__main__":
    logging.info("Starting script")
    
    try:
        papers = fetch_papers(batch_size=100)  # Start with a small batch for testing
        logging.info(f"Collection complete. Total papers: {len(papers)}")
        
        if not papers.empty:
            logging.info("\nSample of collected data:")
            logging.info(papers.head())
            logging.info("\nColumns:")
            logging.info(papers.columns.tolist())
        else:
            logging.warning("No papers were collected")
            
    except Exception as e:
        logging.error(f"Script failed: {e}")