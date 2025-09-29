# Código Fonte - Chess Graph Neural Networks

Esta pasta contém o código fonte do projeto Chess-GNN, organizado em módulos para construção de grafos, implementação de GNNs e análise de dados.

## 📁 Estrutura do Código

```
src/
├── __init__.py              # Inicialização do pacote
├── data/                    # Módulos de dados
│   ├── __init__.py
│   ├── chess_graph.py        # Construção de grafos
│   └── dataset.py            # Dataset PyTorch
├── models/                   # Modelos GNN
│   ├── __init__.py
│   ├── gnn_models.py         # Modelos GNN básicos
│   └── chess_gnn.py         # GNN customizado para xadrez
└── utils/                    # Utilitários
    ├── __init__.py
    ├── visualization.py      # Visualização de grafos
    └── metrics.py            # Métricas de avaliação
```

## 🔧 Módulos Principais

### 1. **data/** - Processamento de Dados

#### `chess_graph.py`
- **Classe `ChessGraph`**: Representa um grafo de posição de xadrez
- **Classe `ChessGraphBuilder`**: Construtor de grafos
- **Tipos de Grafos**: Ataque, defesa, movimento, híbrido
- **Conversão**: NetworkX para PyTorch Geometric

#### `dataset.py`
- **Classe `ChessGraphDataset`**: Dataset PyTorch para grafos
- **Classe `ChessGraphDataLoader`**: DataLoader para batching
- **Transformações**: Normalização, encoding posicional
- **Divisão**: Train/validation/test splits

### 2. **models/** - Modelos GNN

#### `gnn_models.py`
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **GraphSAGE**: Graph Sample and Aggregate
- **GraphClassifier**: Classificador de grafos

#### `chess_gnn.py`
- **ChessGNN**: Modelo customizado para xadrez
- **ChessGNNTrainer**: Treinador do modelo
- **Features**: Encoding posicional, atenção
- **Métricas**: Loss, accuracy, F1-score

### 3. **utils/** - Utilitários

#### `visualization.py`
- **GraphVisualizer**: Visualização de grafos
- **Plot Functions**: Funções auxiliares
- **Attention Maps**: Visualização de atenção
- **Training History**: Gráficos de treinamento

#### `metrics.py`
- **GraphMetrics**: Métricas de grafos
- **Centrality**: Degree, betweenness, closeness
- **Connectivity**: Densidade, clustering
- **Performance**: Accuracy, precision, recall

## 🚀 Como Usar

### Importação Básica
```python
from src.data.chess_graph import ChessGraph, ChessGraphBuilder
from src.models.chess_gnn import ChessGNN, ChessGNNTrainer
from src.utils.visualization import GraphVisualizer
```

### Construção de Grafos
```python
# Criar grafo a partir de posição
board = chess.Board()
graph = ChessGraph(board, graph_type="hybrid")

# Converter para PyTorch Geometric
data = graph.to_pytorch_geometric()
```

### Treinamento de Modelo
```python
# Criar modelo
model = ChessGNN(input_dim=64, hidden_dim=128, output_dim=10)

# Treinar
trainer = ChessGNNTrainer(model)
trainer.setup_training()
history = trainer.train(train_loader, val_loader, epochs=100)
```

### Visualização
```python
# Visualizar grafo
visualizer = GraphVisualizer()
visualizer.plot_chess_graph(graph, title="Chess Position")

# Plotar atenção
visualizer.plot_attention_heatmap(attention_weights)
```

## 📊 Exemplos de Uso

### 1. Análise de Posição
```python
# Carregar posição
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Criar grafo
graph = ChessGraph(board, "attack")

# Analisar métricas
metrics = GraphMetrics()
stats = metrics.calculate_all_metrics(graph.graph)

# Visualizar
visualizer.plot_chess_graph(graph.graph)
```

### 2. Treinamento de Modelo
```python
# Carregar dataset
dataset = ChessGraphDataset("data/pgn/games.pgn")

# Criar modelo
model = ChessGNN(input_dim=64, hidden_dim=128, output_dim=3)

# Treinar
trainer = ChessGNNTrainer(model)
trainer.setup_training()
history = trainer.train(train_loader, val_loader, epochs=50)

# Plotar histórico
visualizer.plot_training_history(history)
```

### 3. Análise de Performance
```python
# Avaliar modelo
val_metrics = trainer.evaluate(val_loader)
print(f"Accuracy: {val_metrics['accuracy']:.2f}%")

# Calcular métricas detalhadas
from src.utils.metrics import evaluate_model_performance
performance = evaluate_model_performance(y_true, y_pred)
```

## 🔧 Configuração

### Variáveis de Ambiente
```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### Configuração de Dispositivo
```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Configuração de Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 📈 Métricas de Performance

### Modelos GNN
- **GCN**: Baseline para comparação
- **GAT**: Melhor para atenção
- **GraphSAGE**: Eficiente para grafos grandes
- **ChessGNN**: Otimizado para xadrez

### Métricas de Avaliação
- **Accuracy**: Precisão geral
- **Precision**: Precisão por classe
- **Recall**: Recall por classe
- **F1-Score**: Média harmônica

## 🐛 Troubleshooting

### Problemas Comuns

#### 1. Erro de Importação
```python
# Adicionar ao início do script
import sys
sys.path.append('.')
```

#### 2. Erro de CUDA
```python
# Forçar uso de CPU
device = torch.device('cpu')
model = model.to(device)
```

#### 3. Erro de Memória
```python
# Reduzir batch size
batch_size = 16  # em vez de 32
```

#### 4. Erro de Dependências
```bash
# Reinstalar dependências
pip install --upgrade -r ../requirements.txt
```

## 📚 Documentação

### Docstrings
Todos os módulos contêm docstrings detalhadas:
```python
def function(param1: int, param2: str) -> bool:
    """
    Descrição da função.
    
    Args:
        param1: Descrição do parâmetro 1
        param2: Descrição do parâmetro 2
        
    Returns:
        Descrição do retorno
        
    Raises:
        ValueError: Quando parâmetro é inválido
    """
```

### Type Hints
Todos os métodos usam type hints:
```python
from typing import List, Dict, Tuple, Optional

def process_graph(graph: nx.Graph) -> Dict[str, float]:
    """Processa grafo e retorna métricas."""
    pass
```

## 👥 Contribuições

Para contribuir com o código:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature
3. **Implemente** sua funcionalidade
4. **Adicione** testes
5. **Documente** o código
6. **Commit** suas mudanças
7. **Push** para a branch
8. **Abra** um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

*Para mais informações, consulte o README principal do projeto.*
