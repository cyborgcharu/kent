# test_client.py
import requests
import time

def test_api():
    base_url = "http://localhost:8000"
    
    # Create test nodes
    nodes = {
        "expert_alice": {"accuracy": 0.95, "unique": True},
        "expert_bob": {"accuracy": 0.88, "unique": True},
        "user_charlie": {"accuracy": 0.75, "unique": False}
    }
    
    print("Creating nodes...")
    for node_id in nodes:
        response = requests.post(
            f"{base_url}/nodes/",
            json={"node_id": node_id}
        )
        print(f"Created {node_id}: {response.json()}")
    
    print("\nSimulating actions...")
    for _ in range(5):
        for node_id, traits in nodes.items():
            response = requests.post(
                f"{base_url}/nodes/{node_id}/actions",
                json={
                    "accuracy": traits["accuracy"],
                    "statement": f"Statement from {node_id}" if traits["unique"] else "Common statement"
                }
            )
            print(f"{node_id} trust: {response.json()['new_trust']:.3f}")
        time.sleep(1)
    
    print("\nFinal network status:")
    status = requests.get(f"{base_url}/network/status").json()
    print(status)

if __name__ == "__main__":
    test_api()