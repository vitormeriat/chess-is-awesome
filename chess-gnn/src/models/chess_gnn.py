"""
Modelo GNN customizado para xadrez.

Este módulo implementa um Graph Neural Network específico
para análise de posições de xadrez.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


class ChessGNN(nn.Module):
    """Graph Neural Network customizado para xadrez."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 use_attention: bool = True,
                 use_positional_encoding: bool = True):
        """
        Inicializa o modelo ChessGNN.
        
        Args:
            input_dim: Dimensão dos features de entrada
            hidden_dim: Dimensão das camadas ocultas
            output_dim: Dimensão da saída
            num_layers: Número de camadas
            dropout: Taxa de dropout
            use_attention: Se deve usar atenção
            use_positional_encoding: Se deve usar encoding posicional
        """
        super(ChessGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        
        # Encoding posicional
        if use_positional_encoding:
            self.pos_encoding = nn.Linear(2, hidden_dim // 4)  # (row, col) -> hidden_dim//4
        
        # Camadas de convolução
        self.convs = nn.ModuleList()
        
        if use_attention:
            # Usar GAT para atenção
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout))
        else:
            # Usar GCN padrão
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Camadas de normalização
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling e classificação
        self.pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Features dos nós [num_nodes, input_dim]
            edge_index: Índices das arestas [2, num_edges]
            edge_attr: Atributos das arestas [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            pos: Posições dos nós [num_nodes, 2]
            
        Returns:
            Logits de classificação [batch_size, output_dim]
        """
        # Adicionar encoding posicional se disponível
        if self.use_positional_encoding and pos is not None:
            pos_encoded = self.pos_encoding(pos)
            x = torch.cat([x, pos_encoded], dim=1)
        
        # Aplicar camadas de convolução
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling global
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Aplicar pooling e classificação
        x = self.pooling(x)
        x = self.classifier(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Retorna pesos de atenção se usando GAT."""
        if not self.use_attention:
            return None
        
        attention_weights = []
        for conv in self.convs:
            if hasattr(conv, 'get_attention'):
                _, att = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(att)
            x = conv(x, edge_index)
        
        return attention_weights


class ChessGNNTrainer:
    """Treinador para o modelo ChessGNN."""
    
    def __init__(self, model: ChessGNN, device: str = "cpu"):
        """
        Inicializa o treinador.
        
        Args:
            model: Modelo ChessGNN
            device: Dispositivo para treinamento
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate: float = 0.001, 
                      weight_decay: float = 1e-4,
                      scheduler_step: int = 50):
        """Configura otimizador e scheduler."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=scheduler_step,
            gamma=0.5
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Treina uma época."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                x=batch_data.x,
                edge_index=batch_data.edge_index,
                edge_attr=batch_data.edge_attr,
                batch=batch_data.batch,
                pos=batch_data.pos if hasattr(batch_data, 'pos') else None
            )
            
            # Calcular loss
            loss = F.cross_entropy(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Estatísticas
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100 * correct / total
        }
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Avalia o modelo."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    x=batch_data.x,
                    edge_index=batch_data.edge_index,
                    edge_attr=batch_data.edge_attr,
                    batch=batch_data.batch,
                    pos=batch_data.pos if hasattr(batch_data, 'pos') else None
                )
                
                # Calcular loss
                loss = F.cross_entropy(outputs, batch_labels)
                
                # Estatísticas
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100 * correct / total
        }
    
    def train(self, train_loader, val_loader, epochs: int = 100) -> Dict[str, List[float]]:
        """Treina o modelo."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Treinar
            train_metrics = self.train_epoch(train_loader)
            
            # Validar
            val_metrics = self.evaluate(val_loader)
            
            # Atualizar scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Salvar histórico
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Log
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        return history
