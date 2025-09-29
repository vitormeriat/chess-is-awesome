"""
Chess Graph Neural Networks

Este pacote implementa Graph Neural Networks para análise de posições de xadrez.
"""

__version__ = "0.1.0"
__author__ = "Vitor Meriat"
__email__ = "vitor@vitormeriat.com"

from .data.chess_graph import ChessGraphBuilder
from .models.chess_gnn import ChessGNN
from .utils.visualization import GraphVisualizer

__all__ = [
    "ChessGraphBuilder",
    "ChessGNN", 
    "GraphVisualizer",
]
