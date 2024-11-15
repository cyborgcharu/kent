# historical_collector.py
import arxiv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_historical.log'),
        logging.StreamHandler()
    ]
)

def fetch_papers_for_period(start_date, end_date, batch_size=100):
    """
    Fetch papers for a specific time period
    """
    client = arxiv.Client()
    papers = []
    
    # Build search query with date range
    query = f'cat:cs.AI OR cat:cs.LG OR cat:cs.NE OR cat:stat.ML AND ' \
            f'submittedDate:[{start_date.strftime("%Y%m%d")}000000 TO ' \
            f'{end_date.strftime("%Y%m%d")}235959]'
    
    try:
        search = arxiv.Search(
            query=query,
            max_results=batch_size,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        logging.info(f"Fetching papers from {start_date} to {end_date}")
        logging.info(f"Query: {query}")
        
        for paper in client.results(search):
            try:
                paper_data = {
                    'id': paper.entry_id,
                    'title': paper.title,
                    'abstract': paper.summary,
                    'authors': [author.name for author in paper.authors],
                    'categories': paper.categories,
                    'date': paper.published
                }
                papers.append(paper_data)
                
                if len(papers) % 10 == 0:
                    logging.info(f"Processed {len(papers)} papers")
                
                time.sleep(0.33)  # Rate limiting
                
            except Exception as e:
                logging.error(f"Error processing paper: {e}")
                continue
                
        return pd.DataFrame(papers)
        
    except Exception as e:
        logging.error(f"Error in fetch_papers_for_period: {e}")
        return pd.DataFrame(papers)

def collect_historical_data(start_year=2015, end_year=2024, period_days=30):
    """
    Collect papers in chunks, saving periodically
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    current_date = start_date
    all_papers = []
    
    while current_date < end_date:
        period_end = min(current_date + timedelta(days=period_days), end_date)
        
        # Fetch papers for this period
        df_period = fetch_papers_for_period(current_date, period_end)
        
        if not df_period.empty:
            # Save this period's data
            filename = f'arxiv_papers_{current_date.strftime("%Y%m")}.csv'
            df_period.to_csv(filename, index=False)
            logging.info(f"Saved {len(df_period)} papers to {filename}")
            
            all_papers.append(df_period)
        
        # Wait between periods to be nice to ArXiv
        time.sleep(1)
        current_date = period_end
    
    # Combine all papers
    if all_papers:
        df_all = pd.concat(all_papers, ignore_index=True)
        df_all.to_csv('arxiv_papers_all.csv', index=False)
        logging.info(f"Saved total of {len(df_all)} papers")
        return df_all
    
    return pd.DataFrame()

def analyze_coverage(df):
    """
    Analyze the coverage of our data collection
    """
    print("\nData Coverage Analysis:")
    print("Date range:", df['date'].min(), "to", df['date'].max())
    print("\nPapers per year:")
    print(df['date'].dt.year.value_counts().sort_index())
    print("\nCategories distribution:")
    print(df['categories'].value_counts().head())
    
    # Plot papers over time
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    df['date'].dt.year.value_counts().sort_index().plot(kind='bar')
    plt.title('Papers per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig('papers_per_year.png')

if __name__ == "__main__":
    logging.info("Starting historical data collection")
    
    # Collect data in chunks
    df = collect_historical_data(start_year=2015, end_year=2024)
    
    if not df.empty:
        analyze_coverage(df)
    else:
        logging.error("No data collected")