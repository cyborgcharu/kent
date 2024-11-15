# tests/test_core.py
import pytest
from src.core import ProvitasCore

def test_basic_trust_evolution():
    """Test that trust evolves as expected"""
    core = ProvitasCore()
    
    # Add test node
    core.add_node("test1")
    
    # Simulate high accuracy, low conformity expert
    trust_values = []
    for _ in range(10):
        trust = core.record_action(
            "test1", 
            accuracy=0.9,  # High accuracy
            statement=f"Unique statement {_}"  # Unique statements
        )
        trust_values.append(trust)
    
    # Trust should increase
    assert trust_values[-1] > trust_values[0]
    assert 0.6 < trust_values[-1] < 1.0

def test_expert_emergence():
    """Test that experts emerge with good behavior"""
    core = ProvitasCore()
    
    # Add two nodes
    core.add_node("expert")
    core.add_node("regular")
    
    # Simulate actions
    for _ in range(10):
        # Expert: high accuracy, low conformity
        core.record_action(
            "expert",
            accuracy=0.95,
            statement=f"Expert statement {_}"
        )
        
        # Regular: medium accuracy, high conformity
        core.record_action(
            "regular",
            accuracy=0.6,
            statement="Common statement"
        )
    
    status = core.get_network_status()
    
    # Expert should have higher trust
    assert status['trust_scores']['expert'] > status['trust_scores']['regular']
    assert 'expert' in status['experts']
    assert 'regular' not in status['experts']

def test_trust_bounds():
    """Test that trust stays within [0,1]"""
    core = ProvitasCore()
    core.add_node("test")
    
    # Try to push trust very high
    for _ in range(20):
        trust = core.record_action("test", accuracy=1.0, statement=f"Unique {_}")
        assert 0 <= trust <= 1
    
    # Try to push trust very low
    for _ in range(20):
        trust = core.record_action("test", accuracy=0.0, statement="Common")
        assert 0 <= trust <= 1