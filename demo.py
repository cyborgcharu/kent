# demo.py
from src.core import ProvitasCore
import time
import statistics
import numpy as np
from colorama import init, Fore, Style

init()  # Initialize colorama

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

def print_step(number, description):
    print(f"{Fore.GREEN}Step {number}: {description}{Style.RESET_ALL}")

def print_insight(text):
    print(f"{Fore.YELLOW}💡 Insight: {text}{Style.RESET_ALL}")

def print_metric(label, value, explanation=""):
    print(f"{Fore.MAGENTA}{label}:{Style.RESET_ALL} {value}")
    if explanation:
        print(f"   {Fore.WHITE}{explanation}{Style.RESET_ALL}")

def run_demo():
    print_header("PROVITAS SYSTEM DEMONSTRATION")
    print("This demo shows how Provitas evaluates trust based on:")
    print("1. Accuracy of contributions")
    print("2. Uniqueness of insights")
    print("3. Consistent performance over time")
    
    print_step(1, "Initializing System")
    core = ProvitasCore()
    
    print_step(2, "Adding Different Types of Participants")
    nodes = {
        "expert_alice": {
            "accuracy": 0.95, 
            "unique": True,
            "description": "High accuracy, unique insights"
        },
        "expert_bob": {
            "accuracy": 0.88, 
            "unique": True,
            "description": "Good accuracy, unique insights"
        },
        "user_charlie": {
            "accuracy": 0.75, 
            "unique": False,
            "description": "Average accuracy, common observations"
        },
        "user_dave": {
            "accuracy": 0.65, 
            "unique": False,
            "description": "Lower accuracy, common observations"
        },
        "user_eve": {
            "accuracy": 0.82, 
            "unique": False,
            "description": "Good accuracy but conformist"
        }
    }
    
    for node, traits in nodes.items():
        core.add_node(node)
        print(f"\n{Fore.WHITE}Added: {node}")
        print(f"Profile: {traits['description']}{Style.RESET_ALL}")
    
    print_step(3, "Running Trust Evolution Simulation")
    trust_history = {node: [] for node in nodes}
    
    for round in range(15):
        print(f"\n{Fore.BLUE}Round {round + 1}{Style.RESET_ALL}")
        print("-" * 20)
        
        round_insights = []
        
        for node, traits in nodes.items():
            # Add small random variation to accuracy
            round_accuracy = min(1.0, max(0.0, 
                traits["accuracy"] + np.random.normal(0, 0.02)))
            
            # Generate and classify statement
            if traits["unique"]:
                statement = f"Unique insight {round}_{node}"
                statement_type = "UNIQUE"
            else:
                statement = f"Common observation {round % 3}"
                statement_type = "COMMON"
            
            # Get current trust before action
            old_trust = trust_history[node][-1] if trust_history[node] else 0.5
            
            # Record action and get updated trust
            trust = core.record_action(
                node,
                accuracy=round_accuracy,
                statement=statement
            )
            
            # Store new trust value
            trust_history[node].append(trust)
            
            # Calculate change
            trust_change = trust - old_trust
            
            # Format output
            change_color = Fore.GREEN if trust_change > 0 else Fore.RED
            print(f"\n{Fore.WHITE}{node}:")
            print(f"  Statement Type: {statement_type}")
            print(f"  Accuracy: {round_accuracy:.3f}")
            print(f"  Trust: {old_trust:.3f} → {trust:.3f}")
            print(f"  Change: {change_color}{trust_change:+.3f}{Style.RESET_ALL}")
            
            # Record interesting patterns
            if abs(trust_change) > 0.02:
                round_insights.append(
                    f"{node}: Large trust {'increase' if trust_change > 0 else 'decrease'} "
                    f"due to {'high' if round_accuracy > 0.8 else 'low'} accuracy "
                    f"and {'unique' if traits['unique'] else 'common'} contribution"
                )
        
        # Show insights for the round
        if round_insights:
            print("\n🔍 Round Insights:")
            for insight in round_insights:
                print(f"  • {insight}")
        
        time.sleep(1)
    
    print_header("FINAL ANALYSIS")
    
    status = core.get_network_status()
    
    print_step(4, "Trust Evolution Results")
    for node in nodes:
        initial_trust = trust_history[node][0]
        final_trust = trust_history[node][-1]
        trust_growth = final_trust - initial_trust
        
        print(f"\n{Fore.WHITE}{node}:{Style.RESET_ALL}")
        print(f"  Initial Trust: {initial_trust:.3f}")
        print(f"  Final Trust:   {final_trust:.3f}")
        print(f"  Growth:        {trust_growth:+.3f}")
        print(f"  Average:       {statistics.mean(trust_history[node]):.3f}")
        
        # Analyze performance
        if trust_growth > 0.2:
            print_insight("Strong trust growth due to consistent high-quality contributions")
        elif trust_growth > 0:
            print_insight("Moderate trust growth - room for improvement")
        else:
            print_insight("Trust decline - needs attention")
    
    print_step(5, "System Performance Metrics")
    
    trust_values = list(status['trust_scores'].values())
    trust_spread = max(trust_values) - min(trust_values)
    expert_ratio = len(status['experts'])/len(nodes)
    
    print_metric("Trust Spread", f"{trust_spread:.3f}",
                "Difference between highest and lowest trust scores")
    print_metric("Expert Ratio", f"{expert_ratio:.2f}",
                "Proportion of nodes identified as experts")
    print_metric("Experts Identified", status['experts'],
                "Nodes that consistently provide accurate, unique insights")
    
    print_header("KEY TAKEAWAYS")
    print("1. Trust grows fastest with accurate, unique contributions")
    print("2. Consistent performance is rewarded over time")
    print("3. System differentiates between experts and regular users")
    print("4. Both accuracy and uniqueness matter for trust building")

if __name__ == "__main__":
    run_demo()