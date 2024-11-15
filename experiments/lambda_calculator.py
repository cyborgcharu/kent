import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
import glob
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys

# Add provitas package to path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_dir))

from provitas.core.kent import EnhancedLambdaCalculator
from provitas.utils.metrics import calculate_information_gradient, calculate_attention

class EnhancedLambdaCalculator:
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        self.d = 100  # dimension of feature space
        
        # Critical thresholds
        self.critical_thresholds = {
            'tau_attention': 0.7,
            'tau_distinctness': 0.3,
            'tau_persistence': 0.5
        }
        
        # Adjusted sector boundaries for better Ω3 detection
        self.sector_thresholds = {
            'omega1': 2.0,      # Information-dominated threshold
            'omega2_upper': 2.0, # Upper bound for balanced zone
            'omega2_lower': 1.0, # Lower bound for balanced zone
            'omega3': 1.0       # Attention-dominated threshold
        }

    def _calculate_temporal_weights(self, dates):
        """Calculate temporal weights giving higher importance to recent papers"""
        time_deltas = (dates - dates.min()).dt.total_seconds()
        return np.array(time_deltas / time_deltas.max())

    def calculate_information_gradient(self, window_papers):
        """Calculate information gradient with reduced scaling"""
        if len(window_papers) < 2:
            return 0
            
        try:
            # Calculate TF-IDF with bigrams
            vectorizer = TfidfVectorizer(
                max_features=self.d,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(window_papers['abstract'])
            
            # Apply temporal weighting
            temporal_weights = self._calculate_temporal_weights(window_papers['date'])
            temporal_weights = temporal_weights.reshape(-1, 1)
            weighted_tfidf = tfidf_matrix.multiply(temporal_weights)
            
            # Calculate term significance with reduced scaling
            term_weights = np.var(tfidf_matrix.toarray(), axis=0)
            gradient_matrix = weighted_tfidf.multiply(term_weights)
            gradient = np.linalg.norm(gradient_matrix.mean(axis=0)) * 35  # Reduced scaling
            
            return gradient
            
        except Exception as e:
            print(f"Error in gradient calculation: {str(e)}")
            return 0

    def calculate_attention(self, window_papers):
        """Calculate attention measure with enhanced weights"""
        if len(window_papers) < 2:
            return 1
            
        try:
            # Category attention with stronger weight
            category_counts = window_papers['categories'].value_counts()
            category_probs = category_counts / len(window_papers)
            category_attention = (1 / (1 + entropy(category_probs))) * 1.5
            
            # Temporal attention with clustering emphasis
            dates = window_papers['date']
            time_diffs = np.diff(dates.astype(np.int64) // 10**9)
            temporal_attention = (1 / (1 + np.std(time_diffs) / (24 * 3600))) * 1.3 if len(time_diffs) > 0 else 1
            
            # Term attention with enhanced focus detection
            vectorizer = TfidfVectorizer(max_features=50)
            tfidf_matrix = vectorizer.fit_transform(window_papers['abstract'])
            term_frequencies = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            term_attention = (1 / (1 + entropy(term_frequencies + 1e-10))) * 1.2
            
            # Combine with adjusted geometric mean
            attention = (category_attention * temporal_attention * term_attention) ** (1/3) * 0.2
            
            return attention
            
        except Exception as e:
            print(f"Error in attention calculation: {str(e)}")
            return 1

    def identify_topological_sector(self, lambda_value):
        """Map λ values to topological sectors with clear boundaries"""
        if lambda_value >= self.sector_thresholds['omega1']:
            return "Ω1 (Information-Dominated)"
        elif (self.sector_thresholds['omega2_lower'] <= lambda_value < 
              self.sector_thresholds['omega2_upper']):
            return "Ω2 (Balanced)"
        else:
            return "Ω3 (Attention-Dominated)"

    def detect_phase_transition(self, lambda_series):
        """Detect phase transitions using statistical measures"""
        if len(lambda_series) < 3:
            return False
            
        delta_lambda = lambda_series.diff()
        mean_change = delta_lambda.mean()
        std_change = delta_lambda.std()
        threshold = 2 * std_change
        
        return abs(delta_lambda - mean_change).max() > threshold

    def calculate_lambda(self, window_size=30):
        """Calculate λ values with overlapping windows"""
        results = []
        lambda_values = []
        
        for i in range(0, len(self.df) - window_size + 1, max(1, window_size // 2)):
            window_papers = self.df.iloc[i:i+window_size]
            
            gradient = self.calculate_information_gradient(window_papers)
            attention = self.calculate_attention(window_papers)
            
            lambda_value = gradient / attention if attention > 0 else 0
            lambda_values.append(lambda_value)
            
            sector = self.identify_topological_sector(lambda_value)
            lambda_series = pd.Series(lambda_values[-10:] if len(lambda_values) > 10 else lambda_values)
            is_transition = self.detect_phase_transition(lambda_series)
            
            results.append({
                'window_start': window_papers['date'].min(),
                'window_end': window_papers['date'].max(),
                'lambda': lambda_value,
                'gradient': gradient,
                'attention': attention,
                'num_papers': len(window_papers),
                'topological_sector': sector,
                'phase_transition': is_transition
            })
            
        return pd.DataFrame(results)

def combine_historical_data():
    """Combine all historical arxiv paper CSVs"""
    all_files = sorted(glob.glob("arxiv_papers_2*.csv"))
    
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            print(f"Loading {filename}: {len(df)} papers")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    # Load and combine data
    df = combine_historical_data()
    print(f"\nAnalyzing {len(df)} papers from {df['date'].min()} to {df['date'].max()}")
    
    # Calculate lambda values
    calculator = EnhancedLambdaCalculator(df)
    lambda_df = calculator.calculate_lambda(window_size=30)
    
    # Basic statistics
    print("\nFirst few windows:")
    print(lambda_df[['window_start', 'window_end', 'lambda', 
                    'topological_sector', 'phase_transition']].head().to_string())
    
    print("\nSummary Statistics:")
    print(lambda_df['lambda'].describe())
    
    # Sector distribution analysis
    sector_counts = lambda_df['topological_sector'].value_counts()
    print("\nSector Distribution:")
    for sector, count in sector_counts.items():
        percentage = (count / len(lambda_df)) * 100
        print(f"{sector}: {count} windows ({percentage:.1f}%)")
    
    # Yearly sector analysis
    lambda_df['year'] = pd.to_datetime(lambda_df['window_start']).dt.year
    yearly_sectors = pd.crosstab(lambda_df['year'], lambda_df['topological_sector'])
    print("\nYearly Sector Distribution:")
    print(yearly_sectors)
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Lambda values with sector boundaries
    plt.subplot(5, 1, 1)
    plt.plot(pd.to_datetime(lambda_df['window_start']), lambda_df['lambda'], 
             'b-', alpha=0.6, label='λ Value')
    plt.axhline(y=calculator.sector_thresholds['omega1'], color='r', 
                linestyle='--', alpha=0.3, label='Ω1 boundary')
    plt.axhline(y=calculator.sector_thresholds['omega2_lower'], color='g', 
                linestyle='--', alpha=0.3, label='Ω2 boundary')
    plt.axhline(y=calculator.sector_thresholds['omega3'], color='b', 
                linestyle='--', alpha=0.3, label='Ω3 boundary')
    plt.title('λ Over Time with Topological Sectors')
    plt.ylabel('λ Value')
    plt.legend()
    
    # Plot 2: Information gradient
    plt.subplot(5, 1, 2)
    plt.plot(pd.to_datetime(lambda_df['window_start']), lambda_df['gradient'], 'g-')
    plt.title('Information Gradient (‖∇I‖)')
    plt.ylabel('Gradient')
    
    # Plot 3: Attention measure
    plt.subplot(5, 1, 3)
    plt.plot(pd.to_datetime(lambda_df['window_start']), lambda_df['attention'], 'r-')
    plt.title('Attention Measure (‖A(Q,K,V)‖)')
    plt.ylabel('Attention')
    
    # Plot 4: Sector distribution over time
    plt.subplot(5, 1, 4)
    sector_colors = {'Ω1 (Information-Dominated)': 'red', 
                    'Ω2 (Balanced)': 'green', 
                    'Ω3 (Attention-Dominated)': 'blue'}
    for sector in sector_colors:
        mask = lambda_df['topological_sector'] == sector
        plt.scatter(pd.to_datetime(lambda_df[mask]['window_start']), 
                   [1]*sum(mask), 
                   c=sector_colors[sector], 
                   label=sector, 
                   alpha=0.6)
    plt.title('Sector Distribution Over Time')
    plt.legend()
    
    # Plot 5: Phase transitions
    plt.subplot(5, 1, 5)
    transitions = lambda_df[lambda_df['phase_transition']]
    plt.plot(pd.to_datetime(lambda_df['window_start']), lambda_df['lambda'], 
             'b-', alpha=0.3)
    plt.scatter(pd.to_datetime(transitions['window_start']), 
                transitions['lambda'], 
                color='red', 
                label='Phase Transitions')
    plt.title('Phase Transitions at Critical Surfaces')
    plt.ylabel('λ Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lambda_analysis_all_sectors.png')
    
    # Print phase transitions
    transitions_df = lambda_df[lambda_df['phase_transition']]
    print("\nDetected Phase Transitions:")
    if not transitions_df.empty:
        print(transitions_df[['window_start', 'window_end', 'lambda', 
                            'topological_sector']].to_string())
    else:
        print("No clear phase transitions detected in this dataset")