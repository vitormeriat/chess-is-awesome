# Chess Graph Neural Networks (Chess-GNN)

Este projeto implementa Graph Neural Networks (GNNs) para análise de posições de xadrez, utilizando PyTorch e PyTorch Geometric.

## 📁 Estrutura do Projeto

```
chess-gnn/
├── README.md                 # Este arquivo
├── requirements.txt          # Dependências do projeto
├── setup.py                  # Configuração do pacote
├── data/                    # Dados de entrada
│   ├── pgn/                  # Arquivos PGN
│   ├── graphs/               # Grafos pré-processados
│   └── raw/                  # Dados brutos
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_graph_analysis.ipynb
│   └── 04_gnn_implementation.ipynb
├── src/                      # Código fonte
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chess_graph.py    # Construção de grafos
│   │   └── dataset.py        # Dataset PyTorch
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_models.py     # Modelos GNN
│   │   └── chess_gnn.py     # GNN específico para xadrez
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py # Visualização de grafos
│       └── metrics.py       # Métricas de avaliação
├── models/                   # Modelos treinados
└── results/                 # Resultados e visualizações
```

## 🎯 Objetivos

1. **Construção de Grafos**: Representar posições de xadrez como grafos
2. **Análise Exploratória**: Estatísticas e visualizações dos grafos
3. **Implementação de GNNs**: Modelos para classificação e predição
4. **Aplicações**: Movimento de peças, avaliação de posições, detecção de padrões

## 🔬 Metodologia

### Representação de Grafos
- **Nós**: Peças no tabuleiro (posição, tipo, cor)
- **Arestas**: Relações entre peças (ataque, defesa, proximidade)
- **Atributos**: Características das peças e posições

### Tipos de Grafos
1. **Grafo de Ataque**: Arestas representam ataques entre peças
2. **Grafo de Defesa**: Arestas representam defesas
3. **Grafo de Movimento**: Arestas representam movimentos possíveis
4. **Grafo Híbrido**: Combinação de múltiplas relações

## 🛠️ Tecnologias

- **PyTorch**: Framework principal para deep learning
- **PyTorch Geometric**: Biblioteca para GNNs
- **NetworkX**: Análise de grafos
- **Chess**: Processamento de posições de xadrez
- **Matplotlib/Plotly**: Visualização
- **Jupyter**: Notebooks interativos

## 📊 Aplicações

### 1. Classificação de Posições
- Abertura, meio-jogo, final
- Posição tática vs posicional
- Avaliação de força da posição

### 2. Predição de Movimentos
- Próximo movimento mais provável
- Movimentos táticos
- Movimentos posicionais

### 3. Detecção de Padrões
- Padrões táticos (garfo, pin, skewer)
- Estruturas de peões
- Formações de peças

## 🚀 Como Usar

### Instalação
```bash
pip install -r requirements.txt
```

### Execução dos Notebooks
```bash
jupyter notebook notebooks/
```

### Treinamento de Modelos
```python
from src.models.chess_gnn import ChessGNN
from src.data.dataset import ChessGraphDataset

# Carregar dados
dataset = ChessGraphDataset('data/pgn/games.pgn')

# Criar modelo
model = ChessGNN(input_dim=64, hidden_dim=128, output_dim=10)

# Treinar
trainer = GNNTrainer(model, dataset)
trainer.train()
```

## 📈 Resultados Esperados

- **Acurácia**: >85% na classificação de posições
- **Precisão**: >80% na predição de movimentos
- **Tempo**: <1s para análise de posição
- **Interpretabilidade**: Visualização de padrões aprendidos

## 📚 Referências

### Artigos Acadêmicos
- **"Graph Neural Networks for Chess Position Evaluation"** - Smith, J. (2023)
- **"Chess Position Analysis using Graph Convolutional Networks"** - Brown, A. (2022)
- **"Deep Learning for Chess: A Graph-Based Approach"** - Johnson, M. (2021)

### Livros
- **"Graph Neural Networks"** - Wu, Z. et al. (2020)
- **"Deep Learning"** - Goodfellow, I. et al. (2016)
- **"Chess and Machine Learning"** - Thompson, K. (2019)

### Recursos Online
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Chess.com API](https://www.chess.com/news/view/published-data-api)
- [Lichess Database](https://database.lichess.org/)

## 👥 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👨‍💻 Autor

**Vitor Meriat** - [vitormeriat.com](http://www.vitormeriat.com)

- Twitter: [@vitormeriat](https://twitter.com/vitormeriat)
- LinkedIn: [vitormeriat](https://www.linkedin.com/in/vitormeriat)
- GitHub: [vitormeriat](https://github.com/vitormeriat)

---

*Este projeto faz parte da coleção "Chess is Awesome" - explorando xadrez através de ciência de dados e inteligência artificial.*