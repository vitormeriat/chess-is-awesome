# Chess Moves Prediction

Chess is a game that is popular for high intelligence and strategic thinking. There has been a lot of research on chess for predicting chess moves, applying chess game theory, and automating chess games. The art of playing chess using computer vision can be implemented using various learning algorithms. A class of Deep Learning has the ability to solve problems of predicting chess moves although facing the necessity of huge datasets.

## How complex is it to calculate moves in chess?

The complexity of calculating moves in chess depends on the algorithm used and the representation of the board and pieces. Here are some examples:

1. Brute-force approach: One simple approach is to calculate all possible moves for every piece on the board and evaluate the resulting positions. This would be an $\large O(n^2)$ operation, where n is the number of pieces on the board, because you would need to evaluate every piece and every possible move it can make.
2. Optimized approach: There are more efficient algorithms that can calculate legal moves in chess more quickly, such as the magic bitboards algorithm. This approach represents the board as a 64-bit integer, where each bit represents the presence or absence of a piece on a particular square. The advantage of this representation is that you can use bitwise operations to quickly calculate all possible moves for a particular piece type. The time complexity of this approach is $\large O(1)$ for each piece type.

The actual time complexity of calculating moves in chess would depend on the specific algorithm used and the implementation details, but it is possible to calculate moves much more quickly than the brute-force approach.

### Legal moves in chess optimized with python

Here is an example of how you can use the magic bitboards algorithm in Python to calculate legal moves for a rook piece:

```python
def get_rook_moves(board, rook_square):
    # Precomputed masks for the rook's moves
    horizontal_mask = 0x101010101010101
    vertical_mask = 0xff

    # The magic number for the rook
    magic_number = 0x2800000028000000
    magic_index = (rook_square * magic_number) & 0xffffffffffffff

    # Get the occupancy of the horizontal and vertical lines
    horizontal_occupancy = board & horizontal_mask
    vertical_occupancy = board & vertical_mask

    # Get the possible moves by using the magic index to access a precomputed table
    horizontal_moves = rook_attacks_table[magic_index][horizontal_occupancy]
    vertical_moves = rook_attacks_table[magic_index][vertical_occupancy]

    # Combine the moves and return the result
    return horizontal_moves | vertical_moves
```

This code uses a precomputed table of possible moves for each square, based on the occupancy of the horizontal and vertical lines. The magic number is used to calculate the index into the table for a given square, and the bitwise operations are used to extract the relevant information from the board representation.

Note that this is just one example of how you can use the magic bitboards algorithm to calculate legal moves in chess. You would need to create similar functions for the other pieces and handle special cases like castling and en passant. Additionally, you would need to generate the precomputed tables of possible moves and store them in your program.

### How to plot a possible movies for a rock with python

To plot possible moves for a rook piece in chess using Python, you can use a library like Matplotlib or Pygame to display a graphical representation of the chess board and the moves. Here is an example using Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_board(board, rook_square, possible_moves):
    # Create a chess board representation as a 2D array
    chess_board = np.zeros((8, 8))
    for i in range(64):
        x, y = i // 8, i % 8
        if (board >> i) & 1:
            chess_board[x][y] = 1

    # Plot the board and possible moves
    fig, ax = plt.subplots()
    ax.imshow(chess_board, cmap="gray")
    for i in range(64):
        x, y = i // 8, i % 8
        if (possible_moves >> i) & 1:
            ax.add_patch(plt.Rectangle((y, 7 - x), 1, 1, color="red", alpha=0.5))
    x, y = rook_square // 8, rook_square % 8
    ax.add_patch(plt.Rectangle((y, 7 - x), 1, 1, color="green"))
    plt.axis("off")
    plt.show()
```

This code uses the `plot_board` function to display a 2D array representation of the chess board, where `board` is a 64-bit integer that represents the pieces on the board, `rook_square` is the square where the rook is located, and `possible_moves` is a 64-bit integer that represents the possible moves for the rook. The function uses the Matplotlib library to display the board and highlight the rook's location in green and the possible moves in red.

Note that this is just a basic example to give you an idea of how you can plot possible moves for a rook piece in chess using Python. You would need to modify this code to handle other pieces and special cases, and you may want to add additional features like labeling the squares, zoom and pan capabilities, etc.

## First approach

### Data transformation 

It reads games in pgn format and transforms them into csvs.

## Bibliography

- H. Panchal, S. Mishra and V. Shrivastava, "[Chess Moves Prediction using Deep Learning Neural Networks](https://ieeexplore.ieee.org/document/9708405)," 2021 International Conference on Advances in Computing and Communications (ICACC), Kochi, Kakkanad, India, 2021, pp. 1-6, doi: 10.1109/ICACC-202152719.2021.9708405.
- 