# category_analysis.py
import pandas as pd
import matplotlib.pyplot as plt

def analyze_categories(df):
    # Create separate boolean columns for each major category
    df['has_AI'] = df['categories'].str.contains('cs.AI')
    df['has_LG'] = df['categories'].str.contains('cs.LG')
    df['has_CL'] = df['categories'].str.contains('cs.CL')  # Computational Linguistics
    df['has_CV'] = df['categories'].str.contains('cs.CV')  # Computer Vision
    df['has_stat_ML'] = df['categories'].str.contains('stat.ML')

    # Count papers in each major category
    major_categories = {
        'Machine Learning (cs.LG)': df['has_LG'].sum(),
        'Artificial Intelligence (cs.AI)': df['has_AI'].sum(),
        'Computational Linguistics (cs.CL)': df['has_CL'].sum(),
        'Computer Vision (cs.CV)': df['has_CV'].sum(),
        'Statistics ML (stat.ML)': df['has_stat_ML'].sum()
    }

    # Calculate intersections
    intersections = {
        'AI + ML': (df['has_AI'] & df['has_LG']).sum(),
        'AI + CL': (df['has_AI'] & df['has_CL']).sum(),
        'ML + CL': (df['has_LG'] & df['has_CL']).sum()
    }

    return major_categories, intersections

def plot_category_distribution(major_categories):
    plt.figure(figsize=(12, 6))
    plt.bar(major_categories.keys(), major_categories.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Papers by Major Category')
    plt.tight_layout()
    plt.savefig('category_distribution.png')

# Load and analyze
df = pd.read_csv('arxiv_papers_20241113_091847.csv')
major_cats, intersections = analyze_categories(df)

print("Major Categories:")
for cat, count in major_cats.items():
    print(f"{cat}: {count} papers")

print("\nKey Intersections:")
for inter, count in intersections.items():
    print(f"{inter}: {count} papers")