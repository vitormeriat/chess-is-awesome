#!/usr/bin/env python3
"""
Script para investigar e corrigir problemas com o arquivo constructed_graphs.pkl
"""
import pickle
import sys
import os
import networkx as nx
import chess

# Definir a classe ChessGraph localmente para resolver o pickle
class ChessGraph:
    """Representa um grafo de posição de xadrez - versão compatível com notebook 02"""
    
    def __init__(self, board_fen: str, graph_type: str = "attack"):
        self.board_fen = board_fen
        try:
            self.board = chess.Board(board_fen)
        except Exception as e:
            print(f"Erro ao criar board com FEN '{board_fen}': {e}")
            self.board = chess.Board()
        self.graph_type = graph_type
        self.nodes = []
        self.edges = []
        self.node_features = {}
        self.edge_features = {}
        self.networkx_graph = nx.Graph()
    
    def to_networkx(self):
        """Converte para NetworkX Graph"""
        G = nx.Graph()
        
        # Adicionar nós
        for node in self.nodes:
            G.add_node(node, **self.node_features.get(node, {}))
        
        # Adicionar arestas
        for edge in self.edges:
            G.add_edge(edge[0], edge[1], **self.edge_features.get(edge, {}))
        
        return G
    
    def to_pytorch_geometric(self):
        """Converte para PyTorch Geometric Data - versão compatível"""
        import torch
        from torch_geometric.data import Data
        
        # Criar matriz de adjacência
        num_nodes = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Arestas
        edge_index = []
        edge_attr = []
        
        for edge in self.edges:
            if edge[0] in node_to_idx and edge[1] in node_to_idx:
                edge_index.append([node_to_idx[edge[0]], node_to_idx[edge[1]]])
                edge_attr.append(list(self.edge_features.get(edge, {}).values()))
        
        # Features dos nós
        node_features = []
        for node in self.nodes:
            features = self.node_features.get(node, {})
            node_features.append([
                features.get('piece_type', 0),
                features.get('color', 0),
                features.get('file', 0),
                features.get('rank', 0)
            ])
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
        )

def investigate_pickle_structure():
    """Investiga a estrutura dos objetos pickled"""
    print("=== INVESTIGANDO ESTRUTURA DO PICKLE ===")
    
    try:
        with open('data/constructed_graphs.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print(f"Tipo dos dados: {type(data)}")
        print(f"Chaves: {list(data.keys()) if isinstance(data, dict) else 'Não é dicionário'}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n{key}: {type(value)} com {len(value)} elementos")
                if len(value) > 0:
                    print(f"  Primeiro elemento: {type(value[0])}")
                    print(f"  Atributos: {list(value[0].__dict__.keys())}")
                    
                    # Verificar se tem atributo 'graph'
                    if hasattr(value[0], 'graph'):
                        print(f"  ✅ Tem atributo 'graph': {type(value[0].graph)}")
                    else:
                        print("  ❌ NÃO tem atributo 'graph'")
                        print(f"  Métodos disponíveis: {[m for m in dir(value[0]) if not m.startswith('_')]}")
                    
                    # Verificar se tem networkx_graph
                    if hasattr(value[0], 'networkx_graph'):
                        print(f"  ✅ Tem atributo 'networkx_graph': {type(value[0].networkx_graph)}")
                    else:
                        print("  ❌ NÃO tem atributo 'networkx_graph'")
                    
                    # Testar to_pytorch_geometric
                    try:
                        pytorch_data = value[0].to_pytorch_geometric()
                        print(f"  ✅ to_pytorch_geometric() funciona: {type(pytorch_data)}")
                    except Exception as e:
                        print(f"  ❌ to_pytorch_geometric() falha: {e}")
        
        return data
        
    except Exception as e:
        print(f"Erro ao carregar pickle: {e}")
        return None

def analyze_graph_construction():
    """Analisa a construção dos grafos"""
    print("\n=== ANÁLISE DA CONSTRUÇÃO DOS GRAFOS ===")
    
    data = investigate_pickle_structure()
    if not data:
        return
    
    for graph_type, graphs in data.items():
        if not graphs:
            continue
            
        print(f"\n📊 GRAFOS DE {graph_type.upper()}:")
        
        # Estatísticas básicas
        num_nodes = [len(g.nodes) for g in graphs]
        num_edges = [len(g.edges) for g in graphs]
        
        print(f"  Nós - Média: {sum(num_nodes)/len(num_nodes):.1f}, Min: {min(num_nodes)}, Max: {max(num_nodes)}")
        print(f"  Arestas - Média: {sum(num_edges)/len(num_edges):.1f}, Min: {min(num_edges)}, Max: {max(num_edges)}")
        
        # Verificar se apenas movement tem arestas
        has_edges = [len(g.edges) > 0 for g in graphs]
        print(f"  Grafos com arestas: {sum(has_edges)}/{len(graphs)}")
        
        if graph_type == 'movement':
            print("  ✅ Movement graphs têm arestas (esperado)")
        else:
            if sum(has_edges) == 0:
                print("  ⚠️ Attack/Defense/Hybrid graphs não têm arestas (pode ser problema)")
            else:
                print("  ✅ Attack/Defense/Hybrid graphs têm arestas")

if __name__ == "__main__":
    investigate_pickle_structure()
    analyze_graph_construction()