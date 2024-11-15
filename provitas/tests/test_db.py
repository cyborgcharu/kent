# tests/test_db.py
from src.core import ProvitasCore
from datetime import datetime

def format_timestamp(dt):
    return dt.strftime("%H:%M:%S")

def test_database():
    print("Creating Provitas Core with SQLite database...")
    core = ProvitasCore(db_url="sqlite:///provitas_test.db")
    
    print("\nTesting database operations...")
    core.add_node("test_user_1")
    core.add_node("test_user_2")
    
    for i in range(5):
        print(f"\nRound {i+1}:")
        # User 1: High accuracy, unique statements
        trust1 = core.record_action(
            "test_user_1", 
            accuracy=0.9, 
            statement=f"Unique insight {i}"
        )
        
        # User 2: Lower accuracy, repeated statements
        trust2 = core.record_action(
            "test_user_2", 
            accuracy=0.7, 
            statement="Common statement"
        )
        
        print(f"User 1 trust: {trust1:.3f}")
        print(f"User 2 trust: {trust2:.3f}")
    
    print("\nNetwork status:")
    status = core.get_network_status()
    for node, trust in status['trust_scores'].items():
        print(f"{node}: {trust:.3f}")
    
    print("\nUser 1 Detailed History:")
    history = core.get_node_history("test_user_1")
    
    print("\nActions:")
    for action in history['actions']:
        print(f"Time: {format_timestamp(action['timestamp'])}")
        print(f"  Accuracy: {action['accuracy']:.2f}")
        print(f"  Conformity: {action['conformity']:.2f}")
        print(f"  Statement: {action['statement']}")
        print("---")
    
    print("\nTrust Evolution:")
    for update in history['trust_history']:
        print(f"Time: {format_timestamp(update['timestamp'])}")
        print(f"  Trust: {update['old_trust']:.3f} → {update['new_trust']:.3f}")
        print(f"  Change: {(update['new_trust'] - update['old_trust']):.3f}")
        print("---")
    
    core.cleanup()

if __name__ == "__main__":
    test_database()