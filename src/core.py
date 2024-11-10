# src/core.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import networkx as nx
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database Models
Base = declarative_base()

class NodeDB(Base):
    __tablename__ = "nodes"
    id = Column(String, primary_key=True)
    current_trust = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    actions = relationship("ActionDB", back_populates="node")
    trust_updates = relationship("TrustUpdateDB", back_populates="node")

class ActionDB(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"))
    accuracy = Column(Float)
    statement = Column(String, nullable=True)
    conformity_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    node = relationship("NodeDB", back_populates="actions")

class TrustUpdateDB(Base):
    __tablename__ = "trust_updates"
    id = Column(Integer, primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"))
    old_trust = Column(Float)
    new_trust = Column(Float)
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    node = relationship("NodeDB", back_populates="trust_updates")

# Memory Model
@dataclass
class Node:
    id: str
    trust: float = 0.5
    accuracy_history: List[float] = field(default_factory=list)
    statements: List[str] = field(default_factory=list)

class ProvitasCore:
    def __init__(self, alpha: float = 0.08, beta: float = 0.07, db_url: Optional[str] = None):
        """
        Initialize Provitas core with optional database connection
        """
        self.network = nx.DiGraph()
        self.alpha = alpha
        self.beta = beta
        self.db_session = None
        
        if db_url:
            engine = create_engine(db_url)
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine)
            self.db_session = Session()
            self._load_state()

    def _load_state(self):
        """Load existing nodes from database"""
        if self.db_session:
            nodes = self.db_session.query(NodeDB).all()
            for node in nodes:
                self.network.add_node(node.id, data=Node(node.id, node.current_trust))

    def add_node(self, node_id: str) -> bool:
        """Add new node to network"""
        if node_id not in self.network:
            self.network.add_node(node_id, data=Node(node_id))
            
            if self.db_session:
                node_db = NodeDB(id=node_id, current_trust=0.5)
                self.db_session.add(node_db)
                self.db_session.commit()
            
            return True
        return False

    def record_action(
        self, 
        node_id: str, 
        accuracy: float, 
        statement: Optional[str] = None
    ) -> float:
        """Record an action and update trust scores"""
        if node_id not in self.network:
            self.add_node(node_id)
            
        node = self.network.nodes[node_id]['data']
        
        # Calculate conformity
        conformity = (
            self._calculate_conformity(node_id, statement) 
            if statement else 0.5
        )
        
        conformity_penalty = conformity * 1.2
        
        # Calculate trust update
        old_trust = node.trust
        delta_trust = (
            self.alpha * (1 - conformity_penalty) * accuracy - 
            self.beta * old_trust
        )
        
        # Apply mediocrity penalty
        if len(node.accuracy_history) > 3:
            recent_accuracy = sum(node.accuracy_history[-3:]) / 3
            if 0.5 <= recent_accuracy <= 0.7:
                delta_trust *= 0.9
        
        # Update trust
        node.trust = max(0, min(1, old_trust + delta_trust))
        
        # Update history
        node.accuracy_history.append(accuracy)
        if statement:
            node.statements.append(statement)
        
        # Record in database
        if self.db_session:
            action_db = ActionDB(
                node_id=node_id,
                accuracy=accuracy,
                statement=statement,
                conformity_score=conformity
            )
            self.db_session.add(action_db)
            
            trust_update_db = TrustUpdateDB(
                node_id=node_id,
                old_trust=old_trust,
                new_trust=node.trust,
                reason=f"Action: acc={accuracy:.2f}, conf={conformity:.2f}"
            )
            self.db_session.add(trust_update_db)
            
            # Update node's current trust
            node_db = self.db_session.query(NodeDB).filter_by(id=node_id).first()
            node_db.current_trust = node.trust
            
            self.db_session.commit()
            
        return node.trust

    def _calculate_conformity(self, node_id: str, statement: str) -> float:
        """Calculate how much a statement conforms to others"""
        other_statements = [
            n['data'].statements[-1] 
            for n in self.network.nodes.values()
            if (n['data'].id != node_id and n['data'].statements)
        ]
        
        if not other_statements:
            return 0
            
        return sum(statement == s for s in other_statements) / len(other_statements)

    def get_network_status(self) -> Dict:
        """Get current network status"""
        # Calculate average trust for baseline
        trust_scores = {
            node_id: data['data'].trust 
            for node_id, data in self.network.nodes(data=True)
        }
        avg_trust = sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0
        
        # Expert identification with multiple criteria
        experts = [
            node_id for node_id, data in self.network.nodes(data=True)
            if (
                data['data'].trust > 0.75 and  # Lowered base threshold
                data['data'].trust > (avg_trust * 1.2) and  # Must be significantly above average
                len(data['data'].accuracy_history) >= 5 and  # Must have sufficient history
                sum(data['data'].accuracy_history[-5:]) / 5 > 0.8  # Sustained high accuracy
            )
        ]
        
        return {
            'node_count': len(self.network),
            'trust_scores': trust_scores,
            'experts': experts,
            'average_trust': avg_trust,
            'trust_spread': max(trust_scores.values()) - min(trust_scores.values()) if trust_scores else 0
        }

    def get_node_history(self, node_id: str) -> Dict:
        """Get historical data for a node"""
        if not self.db_session:
            return None
            
        actions = self.db_session.query(ActionDB).filter_by(node_id=node_id).all()
        trust_updates = self.db_session.query(TrustUpdateDB).filter_by(node_id=node_id).all()
        
        return {
            'actions': [
                {
                    'timestamp': a.timestamp,
                    'accuracy': a.accuracy,
                    'conformity': a.conformity_score,
                    'statement': a.statement
                } for a in actions
            ],
            'trust_history': [
                {
                    'timestamp': t.timestamp,
                    'old_trust': t.old_trust,
                    'new_trust': t.new_trust,
                    'reason': t.reason
                } for t in trust_updates
            ]
        }

    def cleanup(self):
        """Cleanup database connection"""
        if self.db_session:
            self.db_session.close()