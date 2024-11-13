import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import glob

def combine_historical_data():
    """Combine all historical arxiv paper CSVs"""
    all_files = sorted(glob.glob("arxiv_papers_201*.csv") + 
                      glob.glob("arxiv_papers_202*.csv"))
    
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            print(f"Loading {filename}: {len(df)} papers")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return pd.concat(dfs, ignore_index=True)

class LambdaCalculator:
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # Add KENT framework constants
        self.critical_thresholds = {
            'tau_attention': 0.7,  # τa: attention threshold
            'tau_distinctness': 0.3,  # τd: pattern distinctness threshold
            'tau_persistence': 0.5   # τp: temporal persistence threshold
        }
        
        # Define topological sector boundaries
        self.sector_thresholds = {
            'omega1': 2.0,  # λ ≫ 1 threshold
            'omega2_lower': 0.5,  # λ ≈ 1 lower bound
            'omega2_upper': 2.0,  # λ ≈ 1 upper bound
        }
    
    def identify_topological_sector(self, lambda_value: float) -> str:
        """
        Identifies which topological sector (Ω1, Ω2, Ω3) the system is in
        based on λ value, following Section 5.4 of the paper.
        """
        if lambda_value > self.sector_thresholds['omega1']:
            return "Ω1 (Information-Dominated)"
        elif (self.sector_thresholds['omega2_lower'] <= lambda_value <= 
              self.sector_thresholds['omega2_upper']):
            return "Ω2 (Balanced)"
        else:
            return "Ω3 (Attention-Dominated)"
    
    def calculate_information_gradient(self, window_papers):
        """
        Enhanced information gradient calculation incorporating
        KENT's gradient formulation from Section 5.2
        """
        if len(window_papers) < 2:
            return 0
            
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            # Get unique terms and calculate TFIDF
            tfidf_matrix = vectorizer.fit_transform(window_papers['abstract'])
            terms = vectorizer.get_feature_names_out()
            
            # Calculate term frequency changes over time
            term_freqs = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            
            # Enhanced gradient calculation using KENT's formulation
            gradient = np.linalg.norm(term_freqs) / len(window_papers)
            
            return gradient
            
        except Exception as e:
            print(f"Error in gradient calculation: {e}")
            return 0
    
    def calculate_attention(self, window_papers):
        """
        Enhanced attention calculation based on KENT's A(Q,K,V) mechanism
        from Section 5.1
        """
        if len(window_papers) < 2:
            return 1
            
        try:
            # Calculate attention using category distribution
            category_counts = window_papers['categories'].value_counts()
            category_probs = category_counts / len(window_papers)
            
            # Calculate attention focus using entropy
            entropy = -np.sum(category_probs * np.log2(category_probs + 1e-10))
            
            # Map to attention space using KENT's formulation
            attention = 1 / (1 + entropy)
            
            # Check if this is a focus node per Definition 6.1
            is_focus_node = (
                attention > self.critical_thresholds['tau_attention'] and
                entropy < self.critical_thresholds['tau_distinctness']
            )
            
            return attention
            
        except Exception as e:
            print(f"Error in attention calculation: {e}")
            return 1
    
    def detect_phase_transition(self, lambda_series: pd.Series) -> bool:
        """
        Detects phase transitions at critical surfaces Σi as defined
        in Section 5.4 of the paper.
        """
        if len(lambda_series) < 3:
            return False
            
        # Calculate rate of change
        delta_lambda = lambda_series.diff()
        
        # Calculate statistical measures
        mean_change = delta_lambda.mean()
        std_change = delta_lambda.std()
        
        # Define transition threshold based on paper's critical surfaces
        threshold = 2 * std_change
        
        # Check for significant changes indicating phase transition
        transitions = abs(delta_lambda - mean_change) > threshold
        
        return transitions.any()
    
    def calculate_lambda(self, window_size=10):
        """
        Enhanced λ calculation incorporating KENT's topological sectors
        """
        results = []
        lambda_values = []
        
        # Create overlapping windows of papers
        for i in range(0, len(self.df) - window_size + 1, max(1, window_size // 2)):
            window_papers = self.df.iloc[i:i+window_size]
            
            gradient = self.calculate_information_gradient(window_papers)
            attention = self.calculate_attention(window_papers)
            
            lambda_value = gradient / attention if attention > 0 else 0
            lambda_values.append(lambda_value)
            
            # Identify topological sector
            sector = self.identify_topological_sector(lambda_value)
            
            # Detect phase transitions
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

if __name__ == "__main__":
    df = combine_historical_data()
    print(f"\nAnalyzing {len(df)} papers from {df['date'].min()} to {df['date'].max()}")
    
    # Calculate lambda values with enhanced KENT framework
    calculator = LambdaCalculator(df)
    lambda_df = calculator.calculate_lambda(window_size=30)
    
    # Print detailed information
    print("\nFirst few windows:")
    print(lambda_df[['window_start', 'window_end', 'lambda', 
                    'topological_sector', 'phase_transition']].head().to_string())
    
    print("\nSummary Statistics:")
    print(lambda_df['lambda'].describe())
    
    # Plot results with enhanced visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 12))
    
    # Plot lambda values with sector boundaries
    plt.subplot(4, 1, 1)
    plt.plot(range(len(lambda_df)), lambda_df['lambda'], marker='o', alpha=0.6)
    plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.3, label='Ω1 boundary')
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='Ω2 boundary')
    plt.title('λ Over Time with Topological Sectors')
    plt.ylabel('λ Value')
    plt.legend()
    
    # Plot components
    plt.subplot(4, 1, 2)
    plt.plot(range(len(lambda_df)), lambda_df['gradient'], marker='o')
    plt.title('Information Gradient (‖∇I‖)')
    plt.ylabel('Gradient')
    
    plt.subplot(4, 1, 3)
    plt.plot(range(len(lambda_df)), lambda_df['attention'], marker='o')
    plt.title('Attention Measure (‖A(Q,K,V)‖)')
    plt.ylabel('Attention')
    
    # Plot phase transitions
    plt.subplot(4, 1, 4)
    transitions = lambda_df[lambda_df['phase_transition']].index
    plt.plot(range(len(lambda_df)), lambda_df['lambda'], 'b-', alpha=0.5)
    plt.scatter(transitions, lambda_df.loc[transitions, 'lambda'], 
                color='red', label='Phase Transitions')
    plt.title('Phase Transitions at Critical Surfaces')
    plt.ylabel('λ Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('lambda_analysis_v3.png')
    
    # Print phase transition analysis
    transitions_df = lambda_df[lambda_df['phase_transition']]
    if not transitions_df.empty:
        print("\nDetected Phase Transitions:")
        print(transitions_df[['window_start', 'window_end', 'lambda', 
                            'topological_sector']].to_string())
    else:
        print("\nNo clear phase transitions detected in this dataset")