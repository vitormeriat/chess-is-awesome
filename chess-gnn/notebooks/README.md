# Notebooks - Chess Graph Neural Networks

Esta pasta contém os Jupyter notebooks para análise e implementação de Graph Neural Networks aplicados ao xadrez.

## 📓 Notebooks Disponíveis

### 1. **01_data_exploration.ipynb**
**Exploração de Dados de Xadrez**

- Carregamento e análise de dados de xadrez
- Construção de grafos básicos
- Visualização de estruturas de grafos
- Análise estatística dos dados

**Objetivos:**
- Entender a estrutura dos dados de xadrez
- Identificar padrões nas posições
- Preparar dados para construção de grafos

### 2. **02_graph_construction.ipynb**
**Construção de Grafos de Xadrez**

- Implementação de diferentes tipos de grafos
- Visualização de estruturas de grafos
- Análise de propriedades dos grafos
- Preparação de dados para GNNs

**Tipos de Grafos:**
- **Grafo de Ataque**: Arestas representam ataques entre peças
- **Grafo de Defesa**: Arestas representam defesas
- **Grafo de Movimento**: Arestas representam movimentos possíveis
- **Grafo Híbrido**: Combinação de múltiplas relações

### 3. **03_graph_analysis.ipynb**
**Análise de Grafos de Xadrez**

- Análise de propriedades topológicas
- Cálculo de métricas de centralidade
- Identificação de padrões e estruturas
- Preparação de features para GNNs

**Métricas Analisadas:**
- **Centralidade**: Degree, Betweenness, Closeness, Eigenvector
- **Conectividade**: Densidade, Clustering, Path Length
- **Estrutura**: Communities, Motifs, Subgraphs

### 4. **04_gnn_implementation.ipynb**
**Implementação de Graph Neural Networks**

- Implementação de modelos GNN com PyTorch Geometric
- Treinamento de modelos para classificação
- Avaliação de performance dos modelos
- Visualização de resultados e interpretabilidade

**Modelos Implementados:**
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **GraphSAGE**: Graph Sample and Aggregate
- **ChessGNN**: Modelo customizado para xadrez

## 🚀 Como Executar

### Pré-requisitos
```bash
# Instalar dependências
pip install -r ../requirements.txt

# Instalar Jupyter
pip install jupyter notebook
```

### Execução
```bash
# Iniciar Jupyter
jupyter notebook

# Ou executar notebook específico
jupyter notebook 01_data_exploration.ipynb
```

### Ordem Recomendada
1. **01_data_exploration.ipynb** - Entender os dados
2. **02_graph_construction.ipynb** - Construir grafos
3. **03_graph_analysis.ipynb** - Analisar grafos
4. **04_gnn_implementation.ipynb** - Implementar GNNs

## 📊 Resultados Esperados

### Exploração de Dados
- Estatísticas das posições de xadrez
- Distribuição de peças e movimentos
- Visualizações dos tabuleiros

### Construção de Grafos
- Grafos de diferentes tipos
- Visualizações interativas
- Estatísticas dos grafos

### Análise de Grafos
- Métricas de centralidade
- Propriedades topológicas
- Padrões estruturais

### Implementação de GNNs
- Modelos treinados
- Métricas de performance
- Visualizações de atenção

## 🔧 Configuração

### Variáveis de Ambiente
```python
# Configurações recomendadas
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### Configuração de GPU
```python
# Verificar disponibilidade de GPU
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## 📈 Métricas de Performance

### Classificação de Posições
- **Acurácia**: >85%
- **Precisão**: >80%
- **Recall**: >75%
- **F1-Score**: >80%

### Predição de Movimentos
- **Top-1 Accuracy**: >60%
- **Top-5 Accuracy**: >85%
- **Top-10 Accuracy**: >90%

### Tempo de Execução
- **Construção de Grafo**: <100ms
- **Inferência GNN**: <1s
- **Treinamento (época)**: <5min

## 🐛 Troubleshooting

### Problemas Comuns

#### 1. Erro de Importação
```python
# Adicionar ao início do notebook
import sys
sys.path.append('../src')
```

#### 2. Erro de CUDA
```python
# Forçar uso de CPU
device = torch.device('cpu')
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

## 📚 Referências

### Notebooks Específicos
- **Data Exploration**: Análise exploratória de dados
- **Graph Construction**: Construção de grafos
- **Graph Analysis**: Análise de grafos
- **GNN Implementation**: Implementação de GNNs

### Recursos Adicionais
- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/)
- [NetworkX Documentation](https://networkx.org/documentation/)
- [Chess Python Library](https://python-chess.readthedocs.io/)

## 👥 Contribuições

Para contribuir com os notebooks:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature
3. **Adicione** seu notebook
4. **Teste** a execução
5. **Commit** suas mudanças
6. **Push** para a branch
7. **Abra** um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---

*Para mais informações, consulte o README principal do projeto.*
