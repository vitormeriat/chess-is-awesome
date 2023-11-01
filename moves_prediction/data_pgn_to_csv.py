from slugify import slugify
from tqdm import tqdm
import pandas as pd
import numpy as np
import chess.pgn
import chess
import os
import io


def list_to_pgn(lines: list) -> io.TextIOWrapper:
    """Converte uma lista de strings para um TextIOWrapper
    """
    output = io.BytesIO()
    wrapper = io.TextIOWrapper(
        output,
        encoding='cp1252',
        line_buffering=True,
    )
    for l in lines:
        wrapper.write(l)
    wrapper.seek(0, 0)
    return wrapper


def get_board_features(board: chess.Board):
    return [str(board.piece_at(square)) for square in chess.SQUARES]


def get_move_features(move: chess.Move):
    from_ = np.zeros(64)
    to_ = np.zeros(64)
    from_[move.from_square] = 1
    to_[move.to_square] = 1
    return from_, to_


def play(board: chess.Board, game_moves: list, white_won: bool, nb_moves=0):

    if (nb_moves == len(game_moves)):
        return
    if ((white_won and board.turn) or (not white_won and not board.turn)):
        _extracted_from_play_7(
            board=board, nb_moves=nb_moves, game_moves=game_moves)
    board.push(game_moves[nb_moves])
    return play(board=board,
                game_moves=game_moves,
                white_won=white_won,
                nb_moves=nb_moves + 1)


# TODO Rename this here and in `play`
def _extracted_from_play_7(board: chess.Board, nb_moves: int, game_moves: list) -> list:
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
    return data


def generate_file_name(game: chess.pgn.Game, count: int) -> str:
    b = game.headers['Black'].replace(', ', '-').lower()
    w = game.headers['White'].replace(', ', '-').lower()
    r = game.headers['Result']
    d = game.headers['Date'].split('/')[0]
    c = str(count).zfill(4)
    return slugify(f"{c}-{b}-{w}-{r}-{d}")


def process_game(game: list, player: str, count: int):
    try:
        global data
        data = []

        game = chess.pgn.read_game(list_to_pgn(lines=game))
        fname = generate_file_name(game=game, count=count)

        white_won = {'1-0': True, '1/2-1/2': None,
                     '0-1': False}[game.headers['Result']]

        if white_won is None:
            return False

        game_moves = list(game.mainline_moves())
        board = game.board()

        play(board, game_moves=game_moves, white_won=white_won)

        board_feature_names = chess.SQUARE_NAMES
        move_from_feature_names = [
            f'from_{square}' for square in chess.SQUARE_NAMES]
        move_to_feature_names = [
            f'to_{square}' for square in chess.SQUARE_NAMES]

        columns = (
            board_feature_names
            + move_from_feature_names
            + move_to_feature_names
            + ['good_move']
        )

        if os.path.exists(f'./data/games/{player}') == False:
            os.makedirs(f'./data/games/{player}')

        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(f'./data/games/{player}/{fname}.csv', index=False)
        return True
    except Exception as ex:
        return False


def read_pgn_games(path: str) -> list:
    pgns = []
    game = []
    count = 0
    with open(path) as fp:
        while True:
            count += 1
            line = fp.readline()

            if '[Event ' in line and count > 5:
                pgns.append(game)
                game = []
            game.append(line)

            if not line:
                pgns.append(game)
                break
    return pgns


if __name__ == '__main__':
    pgns = read_pgn_games(path='./data/garry-kasparov-2508.pgn')

    print(f"Quantidade de jogos: {len(pgns)}")

    processed = 0

    pbar = tqdm(total=len(pgns))
    for i, pgn in enumerate(pgns):
        is_valid = process_game(game=pgn, player='garry-kasparov', count=i+1)
        if is_valid:
            processed += 1
        pbar.update(1)
    pbar.close()

    print(
        f"\nQuantidade de jogos processados com sucesso: {processed}\nDONE!!!")
