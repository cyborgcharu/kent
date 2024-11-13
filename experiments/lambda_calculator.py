# lambda_calculator_v2.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta

class LambdaCalculator:
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
    def calculate_information_gradient(self, window_papers):
        """
        Enhanced information gradient calculation
        """
        if len(window_papers) < 2:
            return 0
            
        # Calculate concept diversity
        vectorizer = TfidfVectorizer(
            max_features=100,  # Reduced for smaller dataset
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            # Get unique terms
            tfidf_matrix = vectorizer.fit_transform(window_papers['abstract'])
            terms = vectorizer.get_feature_names_out()
            
            # Calculate term frequency changes
            term_freqs = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            
            # Normalize by window size
            gradient = np.sum(term_freqs) / len(window_papers)
            
            return gradient
            
        except Exception as e:
            print(f"Error in gradient calculation: {e}")
            return 0
    
    def calculate_attention(self, window_papers):
        """
        Enhanced attention calculation
        """
        if len(window_papers) < 2:
            return 1
            
        try:
            # Use categories as attention indicator
            category_counts = window_papers['categories'].value_counts()
            category_probs = category_counts / len(window_papers)
            
            # Calculate focus using category distribution
            entropy = -np.sum(category_probs * np.log2(category_probs + 1e-10))
            attention = 1 / (1 + entropy)
            
            return attention
            
        except Exception as e:
            print(f"Error in attention calculation: {e}")
            return 1
    
    def calculate_lambda(self, window_size=10):  # Smaller window size
        """
        Calculate λ using sliding windows
        """
        results = []
        
        # Create overlapping windows of papers
        for i in range(0, len(self.df) - window_size + 1, max(1, window_size // 2)):
            window_papers = self.df.iloc[i:i+window_size]
            
            gradient = self.calculate_information_gradient(window_papers)
            attention = self.calculate_attention(window_papers)
            
            lambda_value = gradient / attention if attention > 0 else 0
            
            results.append({
                'window_start': window_papers['date'].min(),
                'window_end': window_papers['date'].max(),
                'lambda': lambda_value,
                'gradient': gradient,
                'attention': attention,
                'num_papers': len(window_papers)
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('arxiv_papers_20241113_091847.csv')
    
    # Calculate lambda values
    calculator = LambdaCalculator(df)
    lambda_df = calculator.calculate_lambda(window_size=10)  # Smaller windows
    
    # Print detailed information
    print("\nFirst few windows:")
    print(lambda_df.head().to_string())
    
    print("\nSummary Statistics:")
    print(lambda_df.describe())
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 10))
    
    # Plot lambda values
    plt.subplot(3, 1, 1)
    plt.plot(range(len(lambda_df)), lambda_df['lambda'], marker='o')
    plt.title('λ Over Time')
    plt.ylabel('λ Value')
    
    # Plot components
    plt.subplot(3, 1, 2)
    plt.plot(range(len(lambda_df)), lambda_df['gradient'], marker='o')
    plt.title('Information Gradient')
    plt.ylabel('‖∇I‖')
    
    plt.subplot(3, 1, 3)
    plt.plot(range(len(lambda_df)), lambda_df['attention'], marker='o')
    plt.title('Attention Measure')
    plt.ylabel('‖A(Q,K,V)‖')
    
    plt.tight_layout()
    plt.savefig('lambda_analysis_v2.png')
    
    # Identify potential phase transitions
    lambda_mean = lambda_df['lambda'].mean()
    lambda_std = lambda_df['lambda'].std()
    
    print("\nPotential Phase Transitions (λ > 2σ from mean):")
    transitions = lambda_df[abs(lambda_df['lambda'] - lambda_mean) > 2*lambda_std]
    if not transitions.empty:
        print(transitions[['window_start', 'window_end', 'lambda']].to_string())
    else:
        print("No clear phase transitions detected in this dataset")