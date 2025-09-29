#!/usr/bin/env python3
"""
Teste da implementação consolidada da classe ChessGraph
"""
import sys
sys.path.append('./src')

from data.chess_graph import ChessGraph
import chess

def test_chess_graph():
    """Testa a implementação consolidada"""
    print("=== TESTE DA IMPLEMENTAÇÃO CONSOLIDADA ===")
    
    # Teste com posição inicial
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print(f"Testando com FEN: {test_fen}")
    
    # Testar diferentes tipos de grafo
    graph_types = ['attack', 'defense', 'movement', 'hybrid']
    
    for graph_type in graph_types:
        print(f"\n--- Testando grafo {graph_type} ---")
        try:
            graph = ChessGraph(test_fen, graph_type)
            print(f"✅ Grafo criado com sucesso")
            print(f"   Nós: {len(graph.nodes)}")
            print(f"   Arestas: {len(graph.edges)}")
            print(f"   Tem atributo 'graph': {hasattr(graph, 'graph')}")
            print(f"   Tem atributo 'networkx_graph': {hasattr(graph, 'networkx_graph')}")
            
            # Testar to_pytorch_geometric
            try:
                pytorch_data = graph.to_pytorch_geometric()
                print(f"   ✅ to_pytorch_geometric() funciona: {type(pytorch_data)}")
                print(f"   Features shape: {pytorch_data.x.shape if pytorch_data.x is not None else 'None'}")
                print(f"   Edge index shape: {pytorch_data.edge_index.shape if pytorch_data.edge_index is not None else 'None'}")
            except Exception as e:
                print(f"   ❌ to_pytorch_geometric() falha: {e}")
            
            # Testar to_networkx
            try:
                nx_graph = graph.to_networkx()
                print(f"   ✅ to_networkx() funciona: {type(nx_graph)}")
                print(f"   NX nodes: {nx_graph.number_of_nodes()}")
                print(f"   NX edges: {nx_graph.number_of_edges()}")
            except Exception as e:
                print(f"   ❌ to_networkx() falha: {e}")
                
        except Exception as e:
            print(f"❌ Erro ao criar grafo {graph_type}: {e}")

def test_pickle_compatibility():
    """Testa compatibilidade com pickle existente"""
    print("\n=== TESTE DE COMPATIBILIDADE COM PICKLE ===")
    
    try:
        import pickle
        with open('data/constructed_graphs.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print("✅ Pickle carregado com sucesso")
        print(f"Tipo dos dados: {type(data)}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {len(value)} grafos")
                if len(value) > 0:
                    graph = value[0]
                    print(f"    Primeiro grafo: {type(graph)}")
                    print(f"    Atributos: {list(graph.__dict__.keys())}")
                    
                    # Testar se funciona com nova implementação
                    try:
                        # Tentar acessar atributos
                        print(f"    Tem 'graph': {hasattr(graph, 'graph')}")
                        print(f"    Tem 'nodes': {hasattr(graph, 'nodes')}")
                        print(f"    Tem 'edges': {hasattr(graph, 'edges')}")
                        
                        # Testar to_pytorch_geometric
                        pytorch_data = graph.to_pytorch_geometric()
                        print(f"    ✅ to_pytorch_geometric() funciona")
                        
                    except Exception as e:
                        print(f"    ❌ Erro ao processar grafo: {e}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar pickle: {e}")

if __name__ == "__main__":
    test_chess_graph()
    test_pickle_compatibility()
