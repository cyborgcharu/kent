# src/database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Node(Base):
    __tablename__ = "nodes"
    
    id = Column(String, primary_key=True)
    current_trust = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    actions = relationship("Action", back_populates="node")
    trust_updates = relationship("TrustUpdate", back_populates="node")

class Action(Base):
    __tablename__ = "actions"
    
    id = Column(Integer, primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"))
    accuracy = Column(Float)
    statement = Column(String, nullable=True)
    conformity_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    node = relationship("Node", back_populates="actions")

class TrustUpdate(Base):
    __tablename__ = "trust_updates"
    
    id = Column(Integer, primary_key=True)
    node_id = Column(String, ForeignKey("nodes.id"))
    old_trust = Column(Float)
    new_trust = Column(Float)
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    node = relationship("Node", back_populates="trust_updates")

# Database setup function
def init_db():
    engine = create_engine("postgresql://user:password@localhost/provitas")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()