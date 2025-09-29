"""
Módulo de métricas para análise de grafos de xadrez.

Este módulo contém funções para calcular métricas de grafos
e avaliar performance de modelos.
"""

import networkx as nx
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


class GraphMetrics:
    """Classe para cálculo de métricas de grafos."""
    
    def __init__(self):
        """Inicializa o calculador de métricas."""
        self.metrics = {}
    
    def calculate_basic_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Calcula métricas básicas do grafo.
        
        Args:
            graph: Grafo NetworkX
            
        Returns:
            Dicionário com métricas básicas
        """
        metrics = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'average_degree': np.mean([d for n, d in graph.degree()]),
            'max_degree': max([d for n, d in graph.degree()]) if graph.number_of_nodes() > 0 else 0,
            'min_degree': min([d for n, d in graph.degree()]) if graph.number_of_nodes() > 0 else 0,
        }
        
        # Métricas de conectividade
        if graph.number_of_nodes() > 0:
            try:
                metrics['average_clustering'] = nx.average_clustering(graph)
                metrics['transitivity'] = nx.transitivity(graph)
            except:
                metrics['average_clustering'] = 0.0
                metrics['transitivity'] = 0.0
            
            # Métricas de caminho
            if nx.is_connected(graph):
                metrics['average_path_length'] = nx.average_shortest_path_length(graph)
                metrics['diameter'] = nx.diameter(graph)
            else:
                metrics['average_path_length'] = float('inf')
                metrics['diameter'] = float('inf')
        
        return metrics
    
    def calculate_centrality_metrics(self, graph: nx.Graph) -> Dict[str, Dict[int, float]]:
        """
        Calcula métricas de centralidade.
        
        Args:
            graph: Grafo NetworkX
            
        Returns:
            Dicionário com métricas de centralidade
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        centrality_metrics = {}
        
        try:
            # Degree centrality
            centrality_metrics['degree'] = nx.degree_centrality(graph)
        except:
            centrality_metrics['degree'] = {}
        
        try:
            # Betweenness centrality
            centrality_metrics['betweenness'] = nx.betweenness_centrality(graph)
        except:
            centrality_metrics['betweenness'] = {}
        
        try:
            # Closeness centrality
            centrality_metrics['closeness'] = nx.closeness_centrality(graph)
        except:
            centrality_metrics['closeness'] = {}
        
        try:
            # Eigenvector centrality
            centrality_metrics['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            centrality_metrics['eigenvector'] = {}
        
        return centrality_metrics
    
    def calculate_structural_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Calcula métricas estruturais.
        
        Args:
            graph: Grafo NetworkX
            
        Returns:
            Dicionário com métricas estruturais
        """
        metrics = {}
        
        # Componentes conectados
        if graph.number_of_nodes() > 0:
            components = list(nx.connected_components(graph))
            metrics['num_components'] = len(components)
            metrics['largest_component_size'] = len(max(components, key=len)) if components else 0
        
        # Assortatividade
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(graph)
        except:
            metrics['assortativity'] = 0.0
        
        # Motifs (triângulos)
        try:
            metrics['num_triangles'] = sum(nx.triangles(graph).values()) // 3
        except:
            metrics['num_triangles'] = 0
        
        return metrics
    
    def calculate_all_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Calcula todas as métricas do grafo.
        
        Args:
            graph: Grafo NetworkX
            
        Returns:
            Dicionário com todas as métricas
        """
        all_metrics = {}
        
        # Métricas básicas
        all_metrics.update(self.calculate_basic_metrics(graph))
        
        # Métricas de centralidade
        centrality = self.calculate_centrality_metrics(graph)
        all_metrics['centrality'] = centrality
        
        # Métricas estruturais
        structural = self.calculate_structural_metrics(graph)
        all_metrics.update(structural)
        
        return all_metrics


def calculate_centrality(graph: nx.Graph, 
                        centrality_type: str = 'degree') -> Dict[int, float]:
    """
    Calcula centralidade de um tipo específico.
    
    Args:
        graph: Grafo NetworkX
        centrality_type: Tipo de centralidade
        
    Returns:
        Dicionário com centralidade de cada nó
    """
    if graph.number_of_nodes() == 0:
        return {}
    
    try:
        if centrality_type == 'degree':
            return nx.degree_centrality(graph)
        elif centrality_type == 'betweenness':
            return nx.betweenness_centrality(graph)
        elif centrality_type == 'closeness':
            return nx.closeness_centrality(graph)
        elif centrality_type == 'eigenvector':
            return nx.eigenvector_centrality(graph, max_iter=1000)
        else:
            raise ValueError(f"Tipo de centralidade inválido: {centrality_type}")
    except:
        return {}


def calculate_connectivity(graph: nx.Graph) -> Dict[str, float]:
    """
    Calcula métricas de conectividade.
    
    Args:
        graph: Grafo NetworkX
        
    Returns:
        Dicionário com métricas de conectividade
    """
    metrics = {}
    
    if graph.number_of_nodes() == 0:
        return metrics
    
    # Densidade
    metrics['density'] = nx.density(graph)
    
    # Clustering
    try:
        metrics['average_clustering'] = nx.average_clustering(graph)
        metrics['transitivity'] = nx.transitivity(graph)
    except:
        metrics['average_clustering'] = 0.0
        metrics['transitivity'] = 0.0
    
    # Conectividade
    metrics['is_connected'] = nx.is_connected(graph)
    
    if metrics['is_connected']:
        try:
            metrics['average_path_length'] = nx.average_shortest_path_length(graph)
            metrics['diameter'] = nx.diameter(graph)
        except:
            metrics['average_path_length'] = float('inf')
            metrics['diameter'] = float('inf')
    else:
        metrics['average_path_length'] = float('inf')
        metrics['diameter'] = float('inf')
    
    return metrics


def evaluate_model_performance(y_true: List[int], 
                             y_pred: List[int],
                             class_names: List[str] = None) -> Dict[str, Any]:
    """
    Avalia performance de um modelo.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        class_names: Nomes das classes
        
    Returns:
        Dicionário com métricas de performance
    """
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Relatório de classificação
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }


def calculate_graph_similarity(graph1: nx.Graph, 
                              graph2: nx.Graph,
                              method: str = 'jaccard') -> float:
    """
    Calcula similaridade entre dois grafos.
    
    Args:
        graph1: Primeiro grafo
        graph2: Segundo grafo
        method: Método de similaridade
        
    Returns:
        Score de similaridade
    """
    if method == 'jaccard':
        # Similaridade de Jaccard
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        
        intersection = len(edges1.intersection(edges2))
        union = len(edges1.union(edges2))
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    elif method == 'edit_distance':
        # Distância de edição (aproximada)
        try:
            return 1.0 - nx.graph_edit_distance(graph1, graph2) / max(
                graph1.number_of_nodes(), graph2.number_of_nodes()
            )
        except:
            return 0.0
    
    else:
        raise ValueError(f"Método de similaridade inválido: {method}")


def analyze_graph_evolution(graphs: List[nx.Graph]) -> Dict[str, List[float]]:
    """
    Analisa evolução de grafos ao longo do tempo.
    
    Args:
        graphs: Lista de grafos ordenados por tempo
        
    Returns:
        Dicionário com métricas de evolução
    """
    evolution = {
        'num_nodes': [],
        'num_edges': [],
        'density': [],
        'clustering': [],
        'diameter': []
    }
    
    for graph in graphs:
        metrics = GraphMetrics().calculate_basic_metrics(graph)
        
        evolution['num_nodes'].append(metrics['num_nodes'])
        evolution['num_edges'].append(metrics['num_edges'])
        evolution['density'].append(metrics['density'])
        evolution['clustering'].append(metrics['average_clustering'])
        evolution['diameter'].append(metrics['diameter'])
    
    return evolution
