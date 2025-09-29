"""
Implementações de modelos GNN básicos.

Este módulo contém implementações de Graph Neural Networks
usando PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List, Dict, Any


class GCN(nn.Module):
    """Graph Convolutional Network para classificação de grafos."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Inicializa o modelo GCN.
        
        Args:
            input_dim: Dimensão dos features de entrada
            hidden_dim: Dimensão das camadas ocultas
            output_dim: Dimensão da saída
            num_layers: Número de camadas GCN
            dropout: Taxa de dropout
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Camadas GCN
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Classificador
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Features dos nós [num_nodes, input_dim]
            edge_index: Índices das arestas [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Logits de classificação [batch_size, output_dim]
        """
        # Aplicar camadas GCN
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling global
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classificação
        x = self.classifier(x)
        
        return x


class GAT(nn.Module):
    """Graph Attention Network para classificação de grafos."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.5):
        """
        Inicializa o modelo GAT.
        
        Args:
            input_dim: Dimensão dos features de entrada
            hidden_dim: Dimensão das camadas ocultas
            output_dim: Dimensão da saída
            num_layers: Número de camadas GAT
            num_heads: Número de heads de atenção
            dropout: Taxa de dropout
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Camadas GAT
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                    heads=num_heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, 
                                 heads=1, dropout=dropout))
        
        # Classificador
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                  batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Features dos nós [num_nodes, input_dim]
            edge_index: Índices das arestas [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Logits de classificação [batch_size, output_dim]
        """
        # Aplicar camadas GAT
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling global
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classificação
        x = self.classifier(x)
        
        return x


class GraphSAGE(nn.Module):
    """Graph Sample and Aggregate Network para classificação de grafos."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        """
        Inicializa o modelo GraphSAGE.
        
        Args:
            input_dim: Dimensão dos features de entrada
            hidden_dim: Dimensão das camadas ocultas
            output_dim: Dimensão da saída
            num_layers: Número de camadas SAGE
            dropout: Taxa de dropout
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Camadas SAGE
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Classificador
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Features dos nós [num_nodes, input_dim]
            edge_index: Índices das arestas [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Logits de classificação [batch_size, output_dim]
        """
        # Aplicar camadas SAGE
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling global
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Classificação
        x = self.classifier(x)
        
        return x


class GraphClassifier(nn.Module):
    """Classificador de grafos com múltiplas opções de pooling."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 pooling: str = "mean"):
        """
        Inicializa o classificador.
        
        Args:
            input_dim: Dimensão dos features de entrada
            hidden_dim: Dimensão das camadas ocultas
            output_dim: Dimensão da saída
            pooling: Tipo de pooling ("mean", "max", "sum")
        """
        super(GraphClassifier, self).__init__()
        
        self.pooling = pooling
        
        # MLP para classificação
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do classificador.
        
        Args:
            x: Features dos nós [num_nodes, input_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Logits de classificação [batch_size, output_dim]
        """
        # Pooling global
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_mean_pool(x, batch) * x.size(0)  # Aproximação para sum
        
        # Classificação
        x = self.mlp(x)
        
        return x
