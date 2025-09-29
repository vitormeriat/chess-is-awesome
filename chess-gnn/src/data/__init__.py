"""
Módulo de dados para Chess-GNN

Contém classes para construção de grafos a partir de posições de xadrez
e datasets PyTorch para treinamento de GNNs.
"""

from .chess_graph import ChessGraphBuilder, ChessGraph
from .dataset import ChessGraphDataset, ChessGraphDataLoader

__all__ = [
    "ChessGraphBuilder",
    "ChessGraph",
    "ChessGraphDataset", 
    "ChessGraphDataLoader",
]
