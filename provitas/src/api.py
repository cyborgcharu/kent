from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from .core import ProvitasCore

app = FastAPI(title="Provitas API")
system = ProvitasCore()

class Action(BaseModel):
    accuracy: float
    statement: Optional[str] = None

class NodeCreate(BaseModel):
    node_id: str
    initial_trust: Optional[float] = 0.5

@app.post("/nodes/", response_model=Dict)
async def create_node(node: NodeCreate):
    """Create a new node in the network"""
    success = system.add_node(node.node_id)
    if not success:
        raise HTTPException(status_code=400, detail="Node already exists")
    return {"node_id": node.node_id, "trust": node.initial_trust}

@app.post("/nodes/{node_id}/actions", response_model=Dict)
async def record_action(node_id: str, action: Action):
    """Record an action for a node"""
    try:
        new_trust = system.record_action(
            node_id=node_id,
            accuracy=action.accuracy,
            statement=action.statement
        )
        return {
            "node_id": node_id,
            "new_trust": new_trust,
            "action_recorded": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/network/status", response_model=Dict)
async def get_network_status():
    """Get current network status"""
    return system.get_network_status()

@app.get("/nodes/{node_id}/trust", response_model=Dict)
async def get_node_trust(node_id: str):
    """Get trust score for a specific node"""
    try:
        status = system.get_network_status()
        trust = status['trust_scores'].get(node_id)
        if trust is None:
            raise HTTPException(status_code=404, detail="Node not found")
        return {"node_id": node_id, "trust": trust}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))