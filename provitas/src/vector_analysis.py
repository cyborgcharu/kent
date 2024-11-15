from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch

class VectorAnalysis:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with a sentence transformer model for embeddings"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.stored_components = []

    def _compute_embeddings(self, statements: List[str]) -> np.ndarray:
        """Compute embeddings for a list of statements"""
        return self.model.encode(statements, normalize_embeddings=True)

    def build_index(self, components: List[str]):
        """Build FAISS index from components"""
        embeddings = self._compute_embeddings(components)

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors = cosine similarity
        self.index.add(embeddings)
        self.stored_components = components

        return embeddings

    def analyze_relationships(self, components: List[str], k: int = 3) -> Dict:
        """Analyze relationships between components using vector similarity"""
        # Get embeddings and build index
        query_embeddings = self._compute_embeddings(components)

        # Compute all pairwise similarities
        similarities = cosine_similarity(query_embeddings)

        # Find relationships
        relationships = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                sim_score = similarities[i][j]
                relationship = {
                    'component1': components[i],
                    'component2': components[j],
                    'similarity': float(sim_score),
                    'relationship_type': self._classify_relationship(sim_score),
                    'falsification_potential': self._assess_falsification_potential(sim_score)
                }
                relationships.append(relationship)

        # Sort by similarity
        relationships.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'relationships': relationships,
            'embedding_space': query_embeddings,
            'similarity_matrix': similarities
        }

    def _classify_relationship(self, similarity: float) -> str:
        """Classify relationship based on similarity score"""
        if similarity > 0.8:
            return "Direct Dependency"
        elif similarity > 0.6:
            return "Strong Correlation"
        elif similarity > 0.4:
            return "Moderate Association"
        else:
            return "Weak or No Relation"

    def _assess_falsification_potential(self, similarity: float) -> Dict:
        """Assess potential for falsification based on similarity"""
        if similarity > 0.8:
            return {
                'potential': 'High',
                'reason': 'Direct dependency suggests clear falsification path',
                'approach': 'Test independence of components'
            }
        elif similarity > 0.6:
            return {
                'potential': 'Medium',
                'reason': 'Strong correlation may indicate shared mechanisms',
                'approach': 'Investigate confounding variables'
            }
        else:
            return {
                'potential': 'Low',
                'reason': 'Weak relationship suggests independent components',
                'approach': 'Verify absence of causal links'
            }

    def generate_falsification_tests(self, components: List[str]) -> List[Dict]:
        """Generate specific falsification tests based on component relationships"""
        analysis = self.analyze_relationships(components)
        tests = []

        for rel in analysis['relationships']:
            if rel['similarity'] > 0.4:  # Only generate tests for meaningful relationships
                test = {
                    'components': (rel['component1'], rel['component2']),
                    'similarity': rel['similarity'],
                    'relationship': rel['relationship_type'],
                    'falsification_tests': self._generate_tests(
                        rel['component1'],
                        rel['component2'],
                        rel['relationship_type']
                    )
                }
                tests.append(test)

        return tests

    def _generate_tests(self, comp1: str, comp2: str, relationship: str) -> List[Dict]:
        """Generate specific tests for falsifying relationship between components"""
        tests = []

        if relationship == "Direct Dependency":
            tests.append({
                'test_type': 'Independence Test',
                'description': f'Demonstrate {comp1} can occur without {comp2}',
                'methodology': 'Controlled experiment with isolation of components',
                'success_criteria': 'Component 1 observed without Component 2'
            })

        elif relationship == "Strong Correlation":
            tests.append({
                'test_type': 'Confounding Variable Analysis',
                'description': f'Identify hidden variables affecting both components',
                'methodology': 'Multiple regression analysis',
                'success_criteria': 'Relationship weakens when controlling for confounders'
            })

        tests.append({
            'test_type': 'Time Series Analysis',
            'description': f'Analyze temporal relationship between components',
            'methodology': 'Sequential observation with time delays',
            'success_criteria': 'No consistent temporal relationship found'
        })

        return tests

    def find_similar_statements(self, query: str, k: int = 5) -> List[Dict]:
        """Find similar statements in the indexed components"""
        if self.index is None:
            raise ValueError("No index built. Call build_index first.")

        # Get query embedding
        query_embedding = self._compute_embeddings([query])

        # Search index
        similarities, indices = self.index.search(query_embedding, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            results.append({
                'statement': self.stored_components[idx],
                'similarity': float(sim),
                'relationship_type': self._classify_relationship(sim)
            })

        return results