# src/core.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
from .database import Node as NodeDB, Action as ActionDB, TrustUpdate as TrustUpdateDB
from .vector_analysis import VectorAnalysis
from attention import FalsificationAttention

@dataclass
class NetworkState:
    """Represents the current state of the network"""
    lambda_value: float
    topological_sector: str
    gradient: float
    attention: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

class ProvitasCore:
    def __init__(self, alpha: float = 0.08, beta: float = 0.07, 
                 db_session=None, embed_dim: int = 64):
        """Initialize Provitas core with trust and KENT components"""
        self.alpha = alpha
        self.beta = beta
        self.db_session = db_session
        
        # Initialize analysis components
        self.vector_analyzer = VectorFalsification()
        self.attention_analyzer = FalsificationAttention(embed_dim=embed_dim)
        
        # Network state tracking
        self.states: List[NetworkState] = []
        
        # Critical thresholds from KENT paper
        self.sector_thresholds = {
            'omega1': 2.0,
            'omega2_upper': 2.0,
            'omega2_lower': 1.0,
            'omega3': 1.0
        }

    def calculate_lambda(self, nodes: List[NodeDB]) -> NetworkState:
        """Calculate current λ value and determine topological sector"""
        # Calculate information gradient using trust scores
        trust_scores = np.array([node.current_trust for node in nodes])
        gradient = np.gradient(trust_scores).mean()
        
        # Calculate attention using recent actions
        recent_actions = []
        for node in nodes:
            if node.actions:
                recent_actions.extend([a.accuracy for a in node.actions[-5:]])
        attention = np.mean(recent_actions) if recent_actions else 0.5
        
        # Calculate λ
        lambda_value = gradient / attention if attention > 0 else float('inf')
        
        # Determine topological sector
        if lambda_value >= self.sector_thresholds['omega1']:
            sector = "Ω1 (Information-Dominated)"
        elif (self.sector_thresholds['omega2_lower'] <= lambda_value < 
              self.sector_thresholds['omega2_upper']):
            sector = "Ω2 (Balanced)"
        else:
            sector = "Ω3 (Attention-Dominated)"
            
        state = NetworkState(
            lambda_value=lambda_value,
            topological_sector=sector,
            gradient=gradient,
            attention=attention
        )
        
        self.states.append(state)
        return state

    def record_action(self, node_id: str, accuracy: float, 
                     statement: Optional[str] = None) -> Dict:
        """Record an action and update network state"""
        if not self.db_session:
            raise ValueError("Database session required")
            
        # Get or create node
        node = self.db_session.query(NodeDB).filter_by(id=node_id).first()
        if not node:
            node = NodeDB(id=node_id)
            self.db_session.add(node)
        
        # Record action
        action = ActionDB(
            node_id=node_id,
            accuracy=accuracy,
            statement=statement,
            conformity_score=self._calculate_conformity(node_id, statement)
        )
        self.db_session.add(action)
        
        # Update trust
        old_trust = node.current_trust
        delta_trust = self._calculate_trust_update(node, accuracy, action.conformity_score)
        node.current_trust = max(0, min(1, old_trust + delta_trust))
        
        # Record trust update
        trust_update = TrustUpdateDB(
            node_id=node_id,
            old_trust=old_trust,
            new_trust=node.current_trust,
            reason=f"Action recorded: acc={accuracy:.2f}"
        )
        self.db_session.add(trust_update)
        
        # Calculate new network state
        all_nodes = self.db_session.query(NodeDB).all()
        network_state = self.calculate_lambda(all_nodes)
        
        self.db_session.commit()
        
        return {
            'node_id': node_id,
            'new_trust': node.current_trust,
            'network_state': {
                'lambda': network_state.lambda_value,
                'sector': network_state.topological_sector,
                'gradient': network_state.gradient,
                'attention': network_state.attention
            }
        }

    def _calculate_trust_update(self, node: NodeDB, accuracy: float, 
                              conformity: float) -> float:
        """Calculate trust update with KENT-aware adjustments"""
        # Get current network state
        if self.states:
            current_state = self.states[-1]
            
            # Adjust trust update based on topological sector
            if current_state.topological_sector == "Ω1 (Information-Dominated)":
                # Higher weight to accuracy in information-dominated regime
                return self.alpha * accuracy - (self.beta * conformity)
            elif current_state.topological_sector == "Ω2 (Balanced)":
                # Balanced consideration in stable regime
                return (self.alpha * accuracy - self.beta * node.current_trust) * 0.8
            else:
                # Conservative updates in attention-dominated regime
                return (self.alpha * accuracy - self.beta * node.current_trust) * 0.6
        
        # Default update if no state history
        return self.alpha * accuracy - (self.beta * node.current_trust)

    def _calculate_conformity(self, node_id: str, statement: Optional[str]) -> float:
        """Calculate statement conformity using vector analysis"""
        if not statement:
            return 0.5
            
        # Get recent statements from other nodes
        recent_statements = []
        nodes = self.db_session.query(NodeDB).filter(NodeDB.id != node_id).all()
        for node in nodes:
            if node.actions:
                recent = [a.statement for a in node.actions if a.statement]
                if recent:
                    recent_statements.append(recent[-1])
        
        if not recent_statements:
            return 0.5
            
        # Use vector analysis to calculate conformity
        analysis = self.vector_analyzer.analyze_relationships(
            [statement] + recent_statements
        )
        
        # Calculate average similarity to other statements
        similarities = [r['similarity'] for r in analysis['relationships']
                      if statement in (r['component1'], r['component2'])]
        
        return np.mean(similarities) if similarities else 0.5

    def get_network_status(self) -> Dict:
        """Get comprehensive network status including KENT metrics"""
        if not self.db_session:
            raise ValueError("Database session required")
            
        nodes = self.db_session.query(NodeDB).all()
        current_state = self.calculate_lambda(nodes)
        
        trust_scores = {node.id: node.current_trust for node in nodes}
        
        return {
            'node_count': len(nodes),
            'trust_scores': trust_scores,
            'average_trust': np.mean(list(trust_scores.values())),
            'network_state': {
                'lambda': current_state.lambda_value,
                'sector': current_state.topological_sector,
                'gradient': current_state.gradient,
                'attention': current_state.attention
            }
        }