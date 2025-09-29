"""
Dataset PyTorch para grafos de xadrez.

Este módulo implementa datasets PyTorch para treinamento de GNNs
com dados de xadrez.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional, Dict, Any
import chess
import chess.pgn
import numpy as np
from .chess_graph import ChessGraph, ChessGraphBuilder


class ChessGraphDataset(Dataset):
    """Dataset PyTorch para grafos de xadrez."""
    
    def __init__(self, 
                 pgn_file: Optional[str] = None,
                 positions: Optional[List[chess.Board]] = None,
                 labels: Optional[List[int]] = None,
                 graph_type: str = "hybrid",
                 max_positions: int = 1000,
                 transform: Optional[callable] = None):
        """
        Inicializa o dataset.
        
        Args:
            pgn_file: Caminho para arquivo PGN
            positions: Lista de posições de xadrez
            labels: Labels para classificação
            graph_type: Tipo de grafo ('attack', 'defense', 'movement', 'hybrid')
            max_positions: Número máximo de posições
            transform: Transformação a ser aplicada aos dados
        """
        self.graph_type = graph_type
        self.transform = transform
        self.graphs = []
        self.labels = labels or []
        
        if pgn_file:
            self._load_from_pgn(pgn_file, max_positions)
        elif positions:
            self._load_from_positions(positions)
        else:
            raise ValueError("Deve fornecer pgn_file ou positions")
    
    def _load_from_pgn(self, pgn_file: str, max_positions: int):
        """Carrega dados de arquivo PGN."""
        builder = ChessGraphBuilder()
        self.graphs = builder.build_from_pgn(pgn_file, max_positions)
        
        # Gerar labels automáticos se não fornecidos
        if not self.labels:
            self.labels = self._generate_labels()
    
    def _load_from_positions(self, positions: List[chess.Board]):
        """Carrega dados de lista de posições."""
        builder = ChessGraphBuilder()
        self.graphs = builder.build_from_positions(positions, self.graph_type)
        
        # Gerar labels automáticos se não fornecidos
        if not self.labels:
            self.labels = self._generate_labels()
    
    def _generate_labels(self) -> List[int]:
        """Gera labels automáticos baseados nas posições."""
        labels = []
        for graph in self.graphs:
            # Classificar posição baseada em características simples
            label = self._classify_position(graph)
            labels.append(label)
        return labels
    
    def _classify_position(self, graph: ChessGraph) -> int:
        """
        Classifica posição baseada em características do grafo.
        
        Returns:
            0: Abertura
            1: Meio-jogo
            2: Final
        """
        num_pieces = graph.graph.number_of_nodes()
        
        if num_pieces >= 20:
            return 0  # Abertura
        elif num_pieces >= 10:
            return 1  # Meio-jogo
        else:
            return 2  # Final
    
    def __len__(self) -> int:
        """Retorna tamanho do dataset."""
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[Data, int]:
        """Retorna item do dataset."""
        graph = self.graphs[idx]
        label = self.labels[idx]
        
        # Converter para PyTorch Geometric Data
        data = graph.to_pytorch_geometric()
        data.y = torch.tensor(label, dtype=torch.long)
        
        # Aplicar transformação se fornecida
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do dataset."""
        if not self.graphs:
            return {}
        
        # Calcular estatísticas agregadas
        num_nodes = [g.graph.number_of_nodes() for g in self.graphs]
        num_edges = [g.graph.number_of_edges() for g in self.graphs]
        densities = [g.get_graph_statistics()['density'] for g in self.graphs]
        
        return {
            "num_graphs": len(self.graphs),
            "avg_nodes": np.mean(num_nodes),
            "std_nodes": np.std(num_nodes),
            "avg_edges": np.mean(num_edges),
            "std_edges": np.std(num_edges),
            "avg_density": np.mean(densities),
            "std_density": np.std(densities),
            "label_distribution": np.bincount(self.labels),
        }


class ChessGraphDataLoader:
    """DataLoader para grafos de xadrez."""
    
    def __init__(self, 
                 dataset: ChessGraphDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0):
        """
        Inicializa o DataLoader.
        
        Args:
            dataset: Dataset de grafos
            batch_size: Tamanho do batch
            shuffle: Se deve embaralhar os dados
            num_workers: Número de workers para carregamento
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_dataloader(self) -> DataLoader:
        """Retorna DataLoader PyTorch."""
        def collate_fn(batch):
            """Função de collate para batching de grafos."""
            data_list, labels = zip(*batch)
            batch_data = Batch.from_data_list(data_list)
            batch_labels = torch.tensor(labels, dtype=torch.long)
            return batch_data, batch_labels
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )


class ChessGraphTransform:
    """Transformações para dados de grafos de xadrez."""
    
    @staticmethod
    def normalize_features(data: Data) -> Data:
        """Normaliza features dos nós."""
        if data.x is not None:
            # Normalizar features (exceto cor da peça)
            x_normalized = data.x.clone()
            x_normalized[:, 0] = torch.nn.functional.normalize(x_normalized[:, 0:1], p=2, dim=0)
            x_normalized[:, 2:4] = torch.nn.functional.normalize(x_normalized[:, 2:4], p=2, dim=0)
            data.x = x_normalized
        return data
    
    @staticmethod
    def add_positional_encoding(data: Data) -> Data:
        """Adiciona encoding posicional aos nós."""
        if data.x is not None:
            # Adicionar encoding posicional simples
            num_nodes = data.x.size(0)
            pos_encoding = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)
            data.x = torch.cat([data.x, pos_encoding], dim=1)
        return data
    
    @staticmethod
    def add_edge_features(data: Data) -> Data:
        """Adiciona features às arestas."""
        if data.edge_attr is None:
            # Adicionar features básicas às arestas
            num_edges = data.edge_index.size(1)
            edge_features = torch.ones(num_edges, 1)
            data.edge_attr = edge_features
        return data


def create_chess_dataset(pgn_file: str, 
                        graph_type: str = "hybrid",
                        max_positions: int = 1000,
                        train_split: float = 0.8) -> Tuple[ChessGraphDataset, ChessGraphDataset]:
    """
    Cria dataset de treino e teste a partir de arquivo PGN.
    
    Args:
        pgn_file: Caminho para arquivo PGN
        graph_type: Tipo de grafo
        max_positions: Número máximo de posições
        train_split: Proporção de dados de treino
    
    Returns:
        Tuple com datasets de treino e teste
    """
    # Carregar dataset completo
    full_dataset = ChessGraphDataset(
        pgn_file=pgn_file,
        graph_type=graph_type,
        max_positions=max_positions
    )
    
    # Dividir em treino e teste
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    
    # Criar índices
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Criar datasets divididos
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    return train_dataset, test_dataset
