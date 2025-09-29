#!/usr/bin/env python3
"""
Exemplo de uso do Chess-GNN

Este script demonstra como usar o Chess-GNN para análise de posições de xadrez.
"""

import sys
import os
sys.path.append('src')

import chess
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Importar módulos do projeto
from data.chess_graph import ChessGraph, ChessGraphBuilder
from data.dataset import ChessGraphDataset, ChessGraphDataLoader
from models.chess_gnn import ChessGNN, ChessGNNTrainer
from models.gnn_models import GCN, GAT, GraphSAGE
from utils.visualization import GraphVisualizer
from utils.metrics import GraphMetrics, evaluate_model_performance


def main():
    """Função principal do exemplo."""
    print("🏁 Iniciando exemplo do Chess-GNN")
    print("=" * 50)
    
    # 1. Criar posições de exemplo
    print("\n1. Criando posições de exemplo...")
    positions = create_sample_positions()
    print(f"✅ Criadas {len(positions)} posições")
    
    # 2. Construir grafos
    print("\n2. Construindo grafos...")
    graphs = build_graphs(positions)
    print(f"✅ Construídos {len(graphs)} grafos")
    
    # 3. Analisar grafos
    print("\n3. Analisando grafos...")
    analyze_graphs(graphs)
    
    # 4. Visualizar grafos
    print("\n4. Visualizando grafos...")
    visualize_graphs(graphs)
    
    # 5. Criar dataset
    print("\n5. Criando dataset...")
    dataset = create_dataset(positions)
    print(f"✅ Dataset criado com {len(dataset)} amostras")
    
    # 6. Treinar modelo
    print("\n6. Treinando modelo...")
    model, history = train_model(dataset)
    print("✅ Modelo treinado com sucesso")
    
    # 7. Avaliar modelo
    print("\n7. Avaliando modelo...")
    evaluate_model(model, dataset)
    
    print("\n🎉 Exemplo concluído com sucesso!")


def create_sample_positions() -> List[chess.Board]:
    """Cria posições de exemplo para análise."""
    positions = []
    
    # Posição inicial
    board = chess.Board()
    positions.append(board)
    
    # Algumas aberturas famosas
    openings = [
        # Ruy Lopez
        ["e4", "e5", "Nf3", "Nc6", "Bb5"],
        # Sicilian Defense
        ["e4", "c5", "Nf3", "d6", "d4"],
        # Queen's Gambit
        ["d4", "d5", "c4", "e6", "Nc3"],
    ]
    
    for opening in openings:
        board = chess.Board()
        for move_san in opening:
            try:
                move = board.parse_san(move_san)
                board.push(move)
            except:
                continue
        positions.append(board)
    
    return positions


def build_graphs(positions: List[chess.Board]) -> List[ChessGraph]:
    """Constrói grafos a partir das posições."""
    graphs = []
    
    for i, board in enumerate(positions):
        # Criar grafo híbrido
        graph = ChessGraph(board, graph_type="hybrid")
        graphs.append(graph)
        
        print(f"  📊 Grafo {i+1}: {graph.graph.number_of_nodes()} nós, {graph.graph.number_of_edges()} arestas")
    
    return graphs


def analyze_graphs(graphs: List[ChessGraph]) -> None:
    """Analisa propriedades dos grafos."""
    metrics_calculator = GraphMetrics()
    
    for i, graph in enumerate(graphs):
        print(f"\n  📈 Análise do Grafo {i+1}:")
        
        # Métricas básicas
        basic_metrics = metrics_calculator.calculate_basic_metrics(graph.graph)
        print(f"    • Nós: {basic_metrics['num_nodes']}")
        print(f"    • Arestas: {basic_metrics['num_edges']}")
        print(f"    • Densidade: {basic_metrics['density']:.3f}")
        print(f"    • Clustering: {basic_metrics['average_clustering']:.3f}")
        
        # Métricas de centralidade
        centrality = metrics_calculator.calculate_centrality_metrics(graph.graph)
        if centrality['degree']:
            max_degree = max(centrality['degree'].values())
            print(f"    • Centralidade máxima: {max_degree:.3f}")


def visualize_graphs(graphs: List[ChessGraph]) -> None:
    """Visualiza os grafos."""
    visualizer = GraphVisualizer()
    
    # Visualizar primeiro grafo
    if graphs:
        print("  🎨 Visualizando primeiro grafo...")
        visualizer.plot_chess_graph(
            graphs[0].graph,
            title="Posição Inicial - Grafo Híbrido",
            layout="spring"
        )
        
        # Estatísticas de todos os grafos
        print("  📊 Gerando estatísticas...")
        visualizer.plot_graph_statistics(
            [g.graph for g in graphs],
            titles=[f"Posição {i+1}" for i in range(len(graphs))]
        )


def create_dataset(positions: List[chess.Board]) -> ChessGraphDataset:
    """Cria dataset PyTorch."""
    # Gerar labels automáticos
    labels = []
    for board in positions:
        # Classificar baseado no número de peças
        num_pieces = len(board.piece_map())
        if num_pieces >= 20:
            labels.append(0)  # Abertura
        elif num_pieces >= 10:
            labels.append(1)  # Meio-jogo
        else:
            labels.append(2)  # Final
    
    # Criar dataset
    dataset = ChessGraphDataset(
        positions=positions,
        labels=labels,
        graph_type="hybrid"
    )
    
    return dataset


def train_model(dataset: ChessGraphDataset) -> tuple:
    """Treina um modelo GNN."""
    # Criar modelo
    model = ChessGNN(
        input_dim=64,  # Features dos nós
        hidden_dim=128,
        output_dim=3,   # 3 classes: abertura, meio-jogo, final
        num_layers=3,
        use_attention=True
    )
    
    # Criar trainer
    trainer = ChessGNNTrainer(model)
    trainer.setup_training(learning_rate=0.001)
    
    # Dividir dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Criar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Treinar modelo
    print("  🚀 Iniciando treinamento...")
    history = trainer.train(train_loader, val_loader, epochs=20)
    
    return model, history


def evaluate_model(model: ChessGNN, dataset: ChessGraphDataset) -> None:
    """Avalia o modelo treinado."""
    model.eval()
    
    # Fazer predições
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data, label = dataset[i]
            
            # Forward pass
            output = model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=torch.zeros(data.x.size(0), dtype=torch.long)
            )
            
            # Predição
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)
            true_labels.append(label)
    
    # Calcular métricas
    performance = evaluate_model_performance(true_labels, predictions)
    
    print(f"  📊 Resultados da Avaliação:")
    print(f"    • Accuracy: {performance['accuracy']:.3f}")
    print(f"    • Precision: {performance['precision']:.3f}")
    print(f"    • Recall: {performance['recall']:.3f}")
    print(f"    • F1-Score: {performance['f1_score']:.3f}")


if __name__ == "__main__":
    main()
