"""
Configurações do projeto Chess-GNN

Este arquivo contém todas as configurações do projeto,
incluindo parâmetros de modelos, caminhos de dados e configurações de treinamento.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configurações de dados."""
    # Caminhos
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    graphs_dir: str = "data/graphs"
    results_dir: str = "results"
    
    # Arquivos
    sample_positions_file: str = "data/raw/sample_positions.pkl"
    position_analyses_file: str = "data/raw/position_analyses.pkl"
    
    # Parâmetros
    max_positions: int = 1000
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class GraphConfig:
    """Configurações de grafos."""
    # Tipos de grafo
    graph_types: List[str] = None
    
    def __post_init__(self):
        if self.graph_types is None:
            self.graph_types = ["attack", "defense", "movement", "hybrid"]
    
    # Parâmetros de construção
    include_positional_encoding: bool = True
    normalize_features: bool = True
    add_edge_features: bool = True
    
    # Visualização
    default_layout: str = "spring"
    node_size_multiplier: int = 100
    edge_width_multiplier: int = 2


@dataclass
class ModelConfig:
    """Configurações de modelos."""
    # Dimensões
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 10
    
    # Arquitetura
    num_layers: int = 3
    dropout: float = 0.5
    use_attention: bool = True
    num_heads: int = 4
    
    # Pooling
    pooling_type: str = "mean"  # "mean", "max", "sum"
    
    # Regularização
    weight_decay: float = 1e-4
    batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Configurações de treinamento."""
    # Otimização
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    
    # Scheduler
    scheduler: str = "step"  # "step", "cosine", "plateau"
    scheduler_step: int = 50
    scheduler_gamma: float = 0.5
    
    # Treinamento
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Validação
    val_frequency: int = 1
    save_best_model: bool = True


@dataclass
class VisualizationConfig:
    """Configurações de visualização."""
    # Tamanhos
    figsize: tuple = (12, 8)
    dpi: int = 100
    
    # Cores
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'white': '#FFFFFF',
                'black': '#000000',
                'attack': '#FF0000',
                'defense': '#00FF00',
                'movement': '#0000FF',
                'hybrid': '#800080'
            }
    
    # Estilos
    style: str = "seaborn-v0_8"
    palette: str = "husl"
    
    # Salvar figuras
    save_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300


@dataclass
class MetricsConfig:
    """Configurações de métricas."""
    # Métricas básicas
    calculate_basic: bool = True
    calculate_centrality: bool = True
    calculate_structural: bool = True
    
    # Tipos de centralidade
    centrality_types: List[str] = None
    
    def __post_init__(self):
        if self.centrality_types is None:
            self.centrality_types = ["degree", "betweenness", "closeness", "eigenvector"]
    
    # Métricas de performance
    classification_metrics: List[str] = None
    
    def __post_init__(self):
        if self.classification_metrics is None:
            self.classification_metrics = ["accuracy", "precision", "recall", "f1_score"]


@dataclass
class DeviceConfig:
    """Configurações de dispositivo."""
    # Dispositivo
    device: str = "auto"  # "auto", "cpu", "cuda"
    cuda_visible_devices: str = None
    
    # Memória
    max_memory_usage: float = 0.8
    memory_cleanup: bool = True
    
    # Paralelização
    num_threads: int = None
    use_cuda: bool = True


class Config:
    """Classe principal de configuração."""
    
    def __init__(self):
        """Inicializa configurações."""
        self.data = DataConfig()
        self.graph = GraphConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.visualization = VisualizationConfig()
        self.metrics = MetricsConfig()
        self.device = DeviceConfig()
        
        # Criar diretórios
        self._create_directories()
    
    def _create_directories(self):
        """Cria diretórios necessários."""
        directories = [
            self.data.data_dir,
            self.data.raw_dir,
            self.data.graphs_dir,
            self.data.results_dir,
            "models",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configurações para dicionário."""
        return {
            'data': self.data.__dict__,
            'graph': self.graph.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'visualization': self.visualization.__dict__,
            'metrics': self.metrics.__dict__,
            'device': self.device.__dict__
        }
    
    def save(self, filepath: str):
        """Salva configurações em arquivo."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, filepath: str):
        """Carrega configurações de arquivo."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Atualizar configurações
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)


# Instância global de configuração
config = Config()


def get_config() -> Config:
    """Retorna instância global de configuração."""
    return config


def update_config(**kwargs):
    """Atualiza configurações."""
    global config
    
    for section, values in kwargs.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)


# Configurações específicas para diferentes cenários
def get_development_config() -> Config:
    """Configurações para desenvolvimento."""
    config = Config()
    
    # Reduzir tamanhos para desenvolvimento
    config.data.max_positions = 100
    config.training.epochs = 10
    config.training.batch_size = 16
    
    return config


def get_production_config() -> Config:
    """Configurações para produção."""
    config = Config()
    
    # Configurações otimizadas para produção
    config.training.epochs = 200
    config.training.batch_size = 64
    config.training.num_workers = 4
    
    return config


def get_research_config() -> Config:
    """Configurações para pesquisa."""
    config = Config()
    
    # Configurações para experimentos
    config.model.hidden_dim = 256
    config.model.num_layers = 5
    config.training.epochs = 500
    
    return config


if __name__ == "__main__":
    # Exemplo de uso
    config = get_config()
    print("Configurações carregadas:")
    print(f"  • Dados: {config.data.max_positions} posições")
    print(f"  • Modelo: {config.model.hidden_dim} dimensões ocultas")
    print(f"  • Treinamento: {config.training.epochs} épocas")
    
    # Salvar configurações
    config.save("config.json")
    print("✅ Configurações salvas em config.json")
