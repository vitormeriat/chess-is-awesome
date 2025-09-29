#!/usr/bin/env python3
"""
Script simples para investigar a estrutura dos objetos pickled
"""
import pickle
import sys
import os
import networkx as nx
import chess

# Definir a classe ChessGraph localmente para resolver o pickle
class ChessGraph:
    """Representa um grafo de posição de xadrez"""
    
    def __init__(self, board_fen: str, graph_type: str = "attack"):
        self.board_fen = board_fen
        self.board = chess.Board(board_fen)
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

def investigate_pickle_structure():
    """Investiga a estrutura dos objetos pickled"""
    print("=== INVESTIGANDO ESTRUTURA DOS OBJETOS PICKLED ===")
    
    try:
        # Carregar objetos pickled
        with open('data/constructed_graphs.pkl', 'rb') as f:
            chess_graphs = pickle.load(f)
        
        print("✅ Objetos carregados com sucesso!")
        
        # Examinar a estrutura do primeiro objeto
        graph = chess_graphs['attack'][0]
        print(f"\n=== ESTRUTURA DO OBJETO PICKLED ===")
        print(f"Tipo: {type(graph)}")
        print(f"Graph type: {getattr(graph, 'graph_type', 'N/A')}")
        
        # Verificar atributos
        print(f"\n=== ATRIBUTOS DISPONÍVEIS ===")
        attrs = [attr for attr in dir(graph) if not attr.startswith('_')]
        for attr in attrs:
            try:
                value = getattr(graph, attr)
                if callable(value):
                    print(f"{attr}: {type(value)} (método)")
                else:
                    print(f"{attr}: {type(value)} = {value}")
            except Exception as e:
                print(f"{attr}: ERRO - {e}")
        
        # Verificar se tem atributo graph
        print(f"\n=== VERIFICAÇÃO ESPECÍFICA ===")
        print(f"Tem atributo 'graph': {hasattr(graph, 'graph')}")
        print(f"Tem atributo 'nodes': {hasattr(graph, 'nodes')}")
        print(f"Tem atributo 'edges': {hasattr(graph, 'edges')}")
        print(f"Tem atributo 'networkx_graph': {hasattr(graph, 'networkx_graph')}")
        
        # Tentar acessar o grafo interno
        if hasattr(graph, 'graph'):
            print(f"Graph interno: {graph.graph}")
        elif hasattr(graph, 'networkx_graph'):
            print(f"NetworkX graph: {graph.networkx_graph}")
        else:
            print("❌ Nenhum grafo interno encontrado!")
            
    except Exception as e:
        print(f"❌ Erro ao carregar pickle: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_pickle_structure()
