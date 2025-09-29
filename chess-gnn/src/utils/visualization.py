"""
Módulo de visualização para grafos de xadrez.

Este módulo contém funções para visualizar grafos de xadrez
e resultados de GNNs.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import List, Dict, Tuple, Optional, Any
import chess


class GraphVisualizer:
    """Classe para visualização de grafos de xadrez."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Inicializa o visualizador.
        
        Args:
            figsize: Tamanho da figura
        """
        self.figsize = figsize
        self.colors = {
            'white': '#FFFFFF',
            'black': '#000000',
            'attack': '#FF0000',
            'defense': '#00FF00',
            'movement': '#0000FF'
        }
    
    def plot_chess_graph(self, graph: nx.Graph, 
                        title: str = "Chess Graph",
                        layout: str = "spring",
                        show_labels: bool = True) -> None:
        """
        Plota um grafo de xadrez.
        
        Args:
            graph: Grafo NetworkX
            title: Título do gráfico
            layout: Layout do grafo ("spring", "circular", "random")
            show_labels: Se deve mostrar labels dos nós
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Escolher layout
        if layout == "spring":
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "random":
            pos = nx.random_layout(graph)
        else:
            pos = nx.spring_layout(graph)
        
        # Plotar nós
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node in graph.nodes():
            # Cor baseada no tipo de peça
            if 'piece' in graph.nodes[node]:
                piece = graph.nodes[node]['piece']
                if piece.color == chess.WHITE:
                    node_colors.append('#FFFFFF')
                else:
                    node_colors.append('#000000')
            else:
                node_colors.append('#808080')
            
            # Tamanho baseado no valor da peça
            if 'value' in graph.nodes[node]:
                node_sizes.append(graph.nodes[node]['value'] * 100)
            else:
                node_sizes.append(500)
            
            # Label
            if show_labels:
                node_labels[node] = str(node)
        
        # Plotar arestas
        edge_colors = []
        edge_widths = []
        
        for edge in graph.edges():
            if 'relation' in graph.edges[edge]:
                relation = graph.edges[edge]['relation']
                if relation == 'attack':
                    edge_colors.append('#FF0000')
                elif relation == 'defense':
                    edge_colors.append('#00FF00')
                elif relation == 'movement':
                    edge_colors.append('#0000FF')
                else:
                    edge_colors.append('#808080')
            else:
                edge_colors.append('#808080')
            
            # Largura baseada no peso
            if 'weight' in graph.edges[edge]:
                edge_widths.append(graph.edges[edge]['weight'] * 2)
            else:
                edge_widths.append(1)
        
        # Desenhar grafo
        nx.draw_networkx_nodes(graph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        nx.draw_networkx_edges(graph, pos,
                               edge_color=edge_colors,
                               width=edge_widths,
                               alpha=0.6,
                               ax=ax)
        
        if show_labels:
            nx.draw_networkx_labels(graph, pos, 
                                   labels=node_labels,
                                   font_size=8,
                                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Legenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#FFFFFF', markersize=10, label='Peças Brancas'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='#000000', markersize=10, label='Peças Pretas'),
            plt.Line2D([0], [0], color='#FF0000', linewidth=2, label='Ataque'),
            plt.Line2D([0], [0], color='#00FF00', linewidth=2, label='Defesa'),
            plt.Line2D([0], [0], color='#0000FF', linewidth=2, label='Movimento')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def plot_graph_statistics(self, graphs: List[nx.Graph], 
                            titles: List[str] = None) -> None:
        """
        Plota estatísticas de múltiplos grafos.
        
        Args:
            graphs: Lista de grafos
            titles: Títulos para cada grafo
        """
        n_graphs = len(graphs)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Estatísticas básicas
        num_nodes = [g.number_of_nodes() for g in graphs]
        num_edges = [g.number_of_edges() for g in graphs]
        densities = [nx.density(g) for g in graphs]
        clustering = [nx.average_clustering(g) for g in graphs]
        
        # 1. Número de nós
        axes[0, 0].bar(range(n_graphs), num_nodes, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Número de Nós')
        axes[0, 0].set_xlabel('Grafo')
        axes[0, 0].set_ylabel('Nós')
        if titles:
            axes[0, 0].set_xticks(range(n_graphs))
            axes[0, 0].set_xticklabels(titles, rotation=45)
        
        # 2. Número de arestas
        axes[0, 1].bar(range(n_graphs), num_edges, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Número de Arestas')
        axes[0, 1].set_xlabel('Grafo')
        axes[0, 1].set_ylabel('Arestas')
        if titles:
            axes[0, 1].set_xticks(range(n_graphs))
            axes[0, 1].set_xticklabels(titles, rotation=45)
        
        # 3. Densidade
        axes[1, 0].bar(range(n_graphs), densities, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Densidade')
        axes[1, 0].set_xlabel('Grafo')
        axes[1, 0].set_ylabel('Densidade')
        if titles:
            axes[1, 0].set_xticks(range(n_graphs))
            axes[1, 0].set_xticklabels(titles, rotation=45)
        
        # 4. Clustering
        axes[1, 1].bar(range(n_graphs), clustering, color='gold', alpha=0.7)
        axes[1, 1].set_title('Clustering Médio')
        axes[1, 1].set_xlabel('Grafo')
        axes[1, 1].set_ylabel('Clustering')
        if titles:
            axes[1, 1].set_xticks(range(n_graphs))
            axes[1, 1].set_xticklabels(titles, rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_attention_heatmap(self, attention_weights: torch.Tensor,
                              title: str = "Attention Weights") -> None:
        """
        Plota heatmap dos pesos de atenção.
        
        Args:
            attention_weights: Pesos de atenção [num_edges, num_heads]
            title: Título do gráfico
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Converter para numpy se necessário
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Plotar heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Configurar eixos
        ax.set_xlabel('Heads de Atenção')
        ax.set_ylabel('Arestas')
        ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Peso de Atenção')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plota histórico de treinamento.
        
        Args:
            history: Dicionário com histórico de treinamento
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()


def plot_graph(graph: nx.Graph, **kwargs) -> None:
    """Função auxiliar para plotar grafo."""
    visualizer = GraphVisualizer()
    visualizer.plot_chess_graph(graph, **kwargs)


def plot_attention(attention_weights: torch.Tensor, **kwargs) -> None:
    """Função auxiliar para plotar atenção."""
    visualizer = GraphVisualizer()
    visualizer.plot_attention_heatmap(attention_weights, **kwargs)
