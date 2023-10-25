import pandas as pd
import numpy as np
import chess.pgn
import chess
import os


def get_board_features(board: chess.Board):
    return [str(board.piece_at(square)) for square in chess.SQUARES]


def get_move_features(move: chess.Move):
    from_ = np.zeros(64)
    to_ = np.zeros(64)
    from_[move.from_square] = 1
    to_[move.to_square] = 1
    return from_, to_


def play(board: chess.Board, nb_moves=0):

    if (nb_moves == len(game_moves)):
        return
    if ((white_won and board.turn) or (not white_won and not board.turn)):
        _extracted_from_play_7(board, nb_moves)
    board.push(game_moves[nb_moves])
    return play(board, nb_moves + 1)


# TODO Rename this here and in `play`
def _extracted_from_play_7(board: chess.Board, nb_moves: int):
    legal_moves = list(board.legal_moves)
    good_move = game_moves[nb_moves]
    bad_moves = list(filter(lambda x: x != good_move, legal_moves))

    board_features = get_board_features(board)
    line = np.array([], dtype=object)
    # append bad moves to data
    for move in bad_moves:
        from_square, to_square = get_move_features(move)
        line = np.concatenate(
            (board_features, from_square, to_square, [False]))
        data.append(line)

    # append good move to data
    from_square, to_square = get_move_features(good_move)
    line = np.concatenate((board_features, from_square, to_square, [True]))
    data.append(line)


def process_game_data(direname: str, filename: str):
    game_path = os.path.join(dirname, filename)
    pgn = open(game_path)
    game = chess.pgn.read_game(pgn)

    result = {'1-0': True, '1/2-1/2': None,
              '0-1': False}[game.headers['Result']]

    global white_won
    global game_moves
    global data

    data = []

    if (result is None):
        return
    elif (result):
        white_won = True
    else:
        white_won = False

    game_moves = list(game.mainline_moves())
    board = game.board()

    play(board)

    board_feature_names = chess.SQUARE_NAMES
    move_from_feature_names = [
        'from_' + square for square in chess.SQUARE_NAMES]
    move_to_feature_names = ['to_' + square for square in chess.SQUARE_NAMES]

    columns = (
        board_feature_names
        + move_from_feature_names
        + move_to_feature_names
        + ['good_move']
    )

    df = pd.DataFrame(data=data, columns=columns)
    print(df.shape)

    new_filename = filename.replace('pgn', 'csv')
    new_dirname = './data/CSV_BOTVINNIK'
    new_path = os.path.join(new_dirname, new_filename)

    df.to_csv(new_path, index=False)

    return


for dirname, _, filenames in os.walk('/data/PGN/Raw_game/Raw_game/Botvinnik'):
    for filename in filenames:
        process_game_data(dirname, filename)
