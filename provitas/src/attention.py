import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class StatementEncoder(nn.Module):
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = self._create_position_encoding(1000, embed_dim)
        
    def _create_position_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x)
        positions = self.position_encoding[:x.size(1)].to(x.device)
        return embeddings + positions

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim)
        output = self.out_linear(attention_output)
        
        return output, attention_weights

class FalsificationAttention:
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 64, num_heads: int = 4):
        self.encoder = StatementEncoder(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.attention.to(self.device)

    def analyze_statement(self, components: List[str], tokenizer) -> Dict:
        # Get component-level representations
        component_tensors = []
        for comp in components:
            tokens = tokenizer(comp, return_tensors='pt', padding=True, truncation=True)
            component_tensors.append(tokens['input_ids'].to(self.device))
        
        # Pad to same length
        max_len = max(t.size(1) for t in component_tensors)
        padded_tensors = []
        for tensor in component_tensors:
            if tensor.size(1) < max_len:
                padding = torch.zeros((1, max_len - tensor.size(1)), dtype=torch.long).to(self.device)
                tensor = torch.cat([tensor, padding], dim=1)
            padded_tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.cat(padded_tensors, dim=0)
        
        # Get encodings
        with torch.no_grad():
            encoded = self.encoder(batch_tensor)
            
            # Get mean representation for each component
            component_reprs = encoded.mean(dim=1)  # Average over token dimension
            
            # Reshape for attention
            component_reprs = component_reprs.unsqueeze(0)  # Add batch dimension
            
            # Apply self-attention
            _, attention_weights = self.attention(component_reprs, component_reprs, component_reprs)
            
            # Get component-level attention scores
            attention_matrix = attention_weights.mean(dim=1).squeeze()  # Average over heads
            attention_scores = attention_matrix.cpu().numpy()

        # Process results
        critical_pairs = self._identify_critical_pairs(attention_scores, components)
        falsification_points = self._generate_falsification_points(critical_pairs)
        
        return {
            'attention_weights': attention_scores,
            'critical_pairs': critical_pairs,
            'falsification_points': falsification_points
        }

    def _identify_critical_pairs(self, attention_weights: np.ndarray, components: List[str]) -> List[Dict]:
        critical_pairs = []
        n_components = len(components)
        
        for i in range(n_components):
            for j in range(n_components):
                if i != j:
                    # Get single attention score
                    attention_score = float(attention_weights[i, j])
                    if attention_score > 0.2:
                        critical_pairs.append({
                            'component1': components[i],
                            'component2': components[j],
                            'attention_score': attention_score,
                            'relationship_type': self._classify_relationship(attention_score)
                        })
        
        return sorted(critical_pairs, key=lambda x: x['attention_score'], reverse=True)
    
    def _classify_relationship(self, attention_score: float) -> str:
        if attention_score > 0.6:
            return "Strong Dependency"
        elif attention_score > 0.4:
            return "Moderate Correlation"
        else:
            return "Weak Association"
    
    def _generate_falsification_points(self, critical_pairs: List[Dict]) -> List[Dict]:
        falsification_points = []
        
        for pair in critical_pairs:
            if pair['relationship_type'] == "Strong Dependency":
                falsification_points.append({
                    'components': (pair['component1'], pair['component2']),
                    'attention_score': pair['attention_score'],
                    'falsification_approach': "Test component independence",
                    'priority': "High"
                })
            elif pair['relationship_type'] == "Moderate Correlation":
                falsification_points.append({
                    'components': (pair['component1'], pair['component2']),
                    'attention_score': pair['attention_score'],
                    'falsification_approach': "Test for confounding variables",
                    'priority': "Medium"
                })
        
        return sorted(falsification_points, key=lambda x: x['attention_score'], reverse=True)