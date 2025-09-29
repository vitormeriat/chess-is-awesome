"""
Construção de grafos a partir de posições de xadrez.

Este módulo implementa diferentes tipos de grafos para representar posições de xadrez:
- Grafo de ataque: arestas representam ataques entre peças
- Grafo de defesa: arestas representam defesas
- Grafo de movimento: arestas representam movimentos possíveis
- Grafo híbrido: combinação de múltiplas relações
"""

import chess
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import torch
from torch_geometric.data import Data, Batch


@dataclass
class PieceInfo:
    """Informações sobre uma peça no tabuleiro."""
    square: int
    piece: chess.Piece
    color: chess.Color
    position: Tuple[int, int]  # (row, col)
    value: int  # Valor da peça (1-9)


class ChessGraph:
    """Representa um grafo de posição de xadrez - Versão Consolidada."""
    
    def __init__(self, board_fen: str, graph_type: str = "attack"):
        """Inicializa o grafo de xadrez.
        
        Args:
            board_fen: String FEN da posição do tabuleiro
            graph_type: Tipo de grafo ('attack', 'defense', 'movement', 'hybrid')
        """
        self.board_fen = board_fen
        try:
            self.board = chess.Board(board_fen)
        except Exception as e:
            print(f"Erro ao criar board com FEN '{board_fen}': {e}")
            # Usar posição inicial como fallback
            self.board = chess.Board()
        
        self.graph_type = graph_type
        self.graph = nx.Graph()  # NetworkX graph para compatibilidade
        self.pieces = {}
        self.nodes = []
        self.edges = []
        self.node_features = {}
        self.edge_features = {}
        self._build_graph()
    
    def _build_graph(self):
        """Constrói o grafo baseado no tipo especificado."""
        if self.graph_type == "attack":
            self._build_attack_graph()
        elif self.graph_type == "defense":
            self._build_defense_graph()
        elif self.graph_type == "movement":
            self._build_movement_graph()
        elif self.graph_type == "hybrid":
            self._build_hybrid_graph()
        else:
            raise ValueError(f"Tipo de grafo inválido: {self.graph_type}")
    
    def _build_attack_graph(self):
        """Constrói grafo de ataques."""
        # Verificar se board é válido
        if not hasattr(self.board, 'piece_at'):
            print(f"Erro: board não é válido. Tipo: {type(self.board)}")
            return
            
        # Nós: peças no tabuleiro
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.nodes.append(square)
                self.node_features[square] = {
                    'piece_type': piece.piece_type,
                    'color': piece.color,
                    'square': square,
                    'file': chess.square_file(square),
                    'rank': chess.square_rank(square)
                }
                self._add_piece_to_graph(square, piece)
        
        # Arestas: ataques entre peças
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Encontrar peças atacadas por esta peça
                for target_square in chess.SQUARES:
                    if target_square != square:
                        target_piece = self.board.piece_at(target_square)
                        if target_piece and target_piece.color != piece.color:
                            # Verificar se a peça pode atacar o alvo
                            if self._can_attack(square, target_square):
                                self.edges.append((square, target_square))
                                self.edge_features[(square, target_square)] = {
                                    'attack_type': 'direct',
                                    'distance': self._calculate_distance(square, target_square)
                                }
                                self.graph.add_edge(square, target_square, 
                                                  relation="attack", 
                                                  weight=1.0)
    
    def _build_defense_graph(self):
        """Constrói grafo de defesas."""
        # Verificar se board é válido
        if not hasattr(self.board, 'piece_at'):
            print(f"Erro: board não é válido. Tipo: {type(self.board)}")
            return
            
        # Nós: peças no tabuleiro
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.nodes.append(square)
                self.node_features[square] = {
                    'piece_type': piece.piece_type,
                    'color': piece.color,
                    'square': square,
                    'file': chess.square_file(square),
                    'rank': chess.square_rank(square)
                }
                self._add_piece_to_graph(square, piece)
        
        # Arestas: defesas entre peças
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # Encontrar peças defendidas por esta peça
                for target_square in chess.SQUARES:
                    if target_square != square:
                        target_piece = self.board.piece_at(target_square)
                        if target_piece and target_piece.color == piece.color:
                            # Verificar se a peça pode defender o alvo
                            if self._can_defend(square, target_square):
                                self.edges.append((square, target_square))
                                self.edge_features[(square, target_square)] = {
                                    'defense_type': 'direct',
                                    'distance': self._calculate_distance(square, target_square)
                                }
                                self.graph.add_edge(square, target_square,
                                                  relation="defense",
                                                  weight=0.5)
    
    def _build_movement_graph(self):
        """Constrói grafo de movimentos possíveis."""
        # Verificar se board é válido
        if not hasattr(self.board, 'piece_at'):
            print(f"Erro: board não é válido. Tipo: {type(self.board)}")
            return
            
        # Nós: casas do tabuleiro
        for square in chess.SQUARES:
            self.nodes.append(square)
            piece = self.board.piece_at(square)
            self.node_features[square] = {
                'piece_type': piece.piece_type if piece else 0,
                'color': piece.color if piece else 0,  # Usar 0 em vez de None
                'square': square,
                'file': chess.square_file(square),
                'rank': chess.square_rank(square)
            }
            if piece:
                self._add_piece_to_graph(square, piece)
        
        # Arestas: movimentos legais
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                for move in self.board.legal_moves:
                    if move.from_square == square:
                        self.edges.append((square, move.to_square))
                        self.edge_features[(square, move.to_square)] = {
                            'move_type': 'legal',
                            'piece_type': piece.piece_type
                        }
                        self.graph.add_edge(square, move.to_square,
                                          relation="movement",
                                          weight=1.0)
    
    def _build_hybrid_graph(self):
        """Constrói grafo híbrido combinando múltiplas relações."""
        # Adicionar todas as peças
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.nodes.append(square)
                self.node_features[square] = {
                    'piece_type': piece.piece_type,
                    'color': piece.color,
                    'square': square,
                    'file': chess.square_file(square),
                    'rank': chess.square_rank(square)
                }
                self._add_piece_to_graph(square, piece)
        
        # Adicionar relações de ataque
        self._add_attack_relations()
        
        # Adicionar relações de defesa
        self._add_defense_relations()
        
        # Adicionar relações de movimento
        self._add_movement_relations()
    
    def _add_piece_to_graph(self, square: int, piece: chess.Piece):
        """Adiciona uma peça ao grafo."""
        row, col = divmod(square, 8)
        piece_info = PieceInfo(
            square=square,
            piece=piece,
            color=piece.color,
            position=(row, col),
            value=self._get_piece_value(piece)
        )
        
        self.pieces[square] = piece_info
        self.graph.add_node(square, **piece_info.__dict__)
    
    def _get_piece_value(self, piece: chess.Piece) -> int:
        """Retorna o valor de uma peça."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 10
        }
        return values.get(piece.piece_type, 0)
    
    def _can_attack(self, from_square: int, to_square: int) -> bool:
        """Verifica se uma peça pode atacar outra."""
        try:
            move = chess.Move(from_square, to_square)
            return self.board.is_legal(move)
        except:
            return False
    
    def _can_defend(self, from_square: int, to_square: int) -> bool:
        """Verifica se uma peça pode defender outra."""
        # Implementação simplificada - pode ser expandida
        return self._can_attack(from_square, to_square)
    
    def _calculate_distance(self, square1: int, square2: int) -> float:
        """Calcula distância entre duas casas."""
        file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
        file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
        return np.sqrt((file1 - file2)**2 + (rank1 - rank2)**2)
    
    def _add_attack_relations(self):
        """Adiciona relações de ataque ao grafo."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                for target_square in chess.SQUARES:
                    if target_square != square:
                        target_piece = self.board.piece_at(target_square)
                        if target_piece and target_piece.color != piece.color:
                            if self._can_attack(square, target_square):
                                self.edges.append((square, target_square))
                                self.edge_features[(square, target_square)] = {
                                    'relation': 'attack',
                                    'weight': 1.0
                                }
                                self.graph.add_edge(square, target_square,
                                                  relation="attack",
                                                  weight=1.0)
    
    def _add_defense_relations(self):
        """Adiciona relações de defesa ao grafo."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                for target_square in chess.SQUARES:
                    if target_square != square:
                        target_piece = self.board.piece_at(target_square)
                        if target_piece and target_piece.color == piece.color:
                            if self._can_defend(square, target_square):
                                self.edges.append((square, target_square))
                                self.edge_features[(square, target_square)] = {
                                    'relation': 'defense',
                                    'weight': 0.5
                                }
                                self.graph.add_edge(square, target_square,
                                                  relation="defense",
                                                  weight=0.5)
    
    def _add_movement_relations(self):
        """Adiciona relações de movimento ao grafo."""
        for move in self.board.legal_moves:
            self.edges.append((move.from_square, move.to_square))
            self.edge_features[(move.from_square, move.to_square)] = {
                'relation': 'movement',
                'weight': 0.3
            }
            self.graph.add_edge(move.from_square, move.to_square,
                              relation="movement",
                              weight=0.3)
    
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
    
    def to_pytorch_geometric(self) -> Data:
        """Converte o grafo para PyTorch Geometric Data."""
        # Criar matriz de adjacência
        num_nodes = len(self.nodes)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Arestas
        edge_index = []
        edge_attr = []
        
        for edge in self.edges:
            if edge[0] in node_to_idx and edge[1] in node_to_idx:
                edge_index.append([node_to_idx[edge[0]], node_to_idx[edge[1]]])
                edge_features = self.edge_features.get(edge, {})
                edge_attr.append([
                    edge_features.get('weight', 1.0),
                    self._relation_to_int(edge_features.get('relation', 'unknown'))
                ])
        
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
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 2), dtype=torch.float)
        )
    
    def _relation_to_int(self, relation: str) -> int:
        """Converte relação para inteiro."""
        relations = {"attack": 0, "defense": 1, "movement": 2}
        return relations.get(relation, 0)
    
    def get_graph_statistics(self) -> Dict:
        """Retorna estatísticas do grafo."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "density": len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1) / 2) if len(self.nodes) > 1 else 0,
            "graph_type": self.graph_type,
        }


class ChessGraphBuilder:
    """Construtor de grafos de xadrez."""
    
    def __init__(self):
        self.graphs = []
    
    def build_from_fen(self, fen: str, graph_type: str = "attack") -> ChessGraph:
        """Constrói grafo a partir de FEN."""
        return ChessGraph(fen, graph_type)
    
    def build_from_pgn(self, pgn_file: str, max_positions: int = 1000) -> List[ChessGraph]:
        """Constrói grafos a partir de arquivo PGN."""
        import chess.pgn
        
        graphs = []
        with open(pgn_file) as f:
            game_count = 0
            while game_count < max_positions:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Processar posições do jogo
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    graph = ChessGraph(board.fen(), "hybrid")
                    graphs.append(graph)
                    game_count += 1
                    
                    if game_count >= max_positions:
                        break
        
        return graphs
    
    def build_from_positions(self, positions: List[chess.Board], 
                           graph_type: str = "attack") -> List[ChessGraph]:
        """Constrói grafos a partir de lista de posições."""
        graphs = []
        for board in positions:
            graph = ChessGraph(board.fen(), graph_type)
            graphs.append(graph)
        return graphs

