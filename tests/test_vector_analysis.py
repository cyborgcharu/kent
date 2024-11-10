from src.vector_analysis import VectorFalsification
from pprint import pprint

def print_section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}\n")

def test_falsification():
    analyzer = VectorFalsification()
    
    # Test Case 1: Scientific Claim
    print_section("Scientific Health Claims")
    scientific_components = [
        "Regular exercise improves cardiovascular health",
        "Higher heart rate during exercise strengthens the heart",
        "Exercise increases blood flow to muscles",
        "Resting heart rate decreases with regular exercise",
        "Blood pressure normalizes after consistent training"
    ]
    
    print("Analyzing relationships...")
    analysis = analyzer.analyze_relationships(scientific_components)
    
    print("\nTop Relationships:")
    for rel in analysis['relationships'][:3]:
        print(f"\nRelationship Type: {rel['relationship_type']}")
        print(f"Components:")
        print(f"1. {rel['component1']}")
        print(f"2. {rel['component2']}")
        print(f"Similarity Score: {rel['similarity']:.3f}")
    
    print("\nGenerating Falsification Tests...")
    tests = analyzer.generate_falsification_tests(scientific_components)
    
    for test in tests[:2]:
        print(f"\nTest for: {test['relationship']}")
        print(f"Components involved:")
        print(f"1. {test['components'][0]}")
        print(f"2. {test['components'][1]}")
        print("\nProposed Tests:")
        for t in test['falsification_tests']:
            print(f"- {t['test_type']}: {t['description']}")
            print(f"  Method: {t['methodology']}")
    
    # Test Case 2: Technical System
    print_section("Technical Performance Claims")
    technical_components = [
        "System processes 1000 requests per second",
        "Average response time is under 100ms",
        "Database queries complete within 50ms",
        "Memory usage stays under 4GB",
        "CPU utilization remains below 80%"
    ]
    
    analyzer.build_index(technical_components)
    print("\nFinding related components for 'slow response time'...")
    similar = analyzer.find_similar_statements("slow response time")
    
    print("\nRelated Components:")
    for result in similar:
        print(f"\nStatement: {result['statement']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Relationship: {result['relationship_type']}")

if __name__ == "__main__":
    test_falsification()