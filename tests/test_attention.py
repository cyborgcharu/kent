# test_attention.py
from src.attention import FalsificationAttention
from transformers import AutoTokenizer
import torch.cuda

def print_test_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def test_statement(attention, tokenizer, components, test_name):
    print_test_header(test_name)
    print("\nAnalyzing components:")
    for i, comp in enumerate(components, 1):
        print(f"{i}. {comp}")
        
    results = attention.analyze_statement(components, tokenizer)
    
    print("\nCritical Component Pairs:")
    for pair in results['critical_pairs'][:3]:
        print(f"\nRelationship ({pair['relationship_type']}):")
        print(f"  Component 1: {pair['component1']}")
        print(f"  Component 2: {pair['component2']}")
        print(f"  Attention Score: {pair['attention_score']:.3f}")
    
    print("\nPotential Falsification Points:")
    for point in results['falsification_points']:
        print(f"\nPriority: {point['priority']}")
        print(f"Components: {point['components'][0]} ⟷ {point['components'][1]}")
        print(f"Approach: {point['falsification_approach']}")

def run_tests():
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Initialize models
    print("\nInitializing tokenizer and attention mechanism...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    attention = FalsificationAttention()
    
    # Test Case 1: Scientific Claim
    scientific_components = [...]
    test_statement(attention, tokenizer, scientific_components, "Scientific Health Claim")
    
    # Test Case 2: Technical System
    technical_components = [...]
    test_statement(attention, tokenizer, technical_components, "Technical Performance Claim")
    
    # Test Case 3: Causal Relationship
    causal_components = [...]
    test_statement(attention, tokenizer, causal_components, "Causal Relationship Claim")
    
    # Test Case 4: Complex System
    complex_components = [...]
    test_statement(attention, tokenizer, complex_components, "Complex System Claim")

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise