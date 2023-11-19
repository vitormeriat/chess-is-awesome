from rich.console import Console
from rich.markdown import Markdown
from rich import print
import chess.pgn
import chess


MARKDOWN = """
# Chess Strategy

Every move is gold. You need plenty of time to ponder about a move. **EVERY MOVE MUST FOLLOW** a **`Chess Principle`**! If it does not, then this move is crap! Don’t become a wood pusher in chess.

Choose a principle:

01. Control the Center
02. Earning a FREE MOVE while attacking the INSTABLE BISHOP to make a LUFTLOCH
03. The dark-running Bishop has left the white Queenside and can’t retreat to the Queenside anymore
04. Gaining a Tempi to develop the black Bishop to g7
05. Develops a Piece quickly
06. Build a Wall to reduce the Bishops Power with c3
07. White wants to develop his Kingside pieces to be able to castle short
08. Misplaced Bishop
09. Develop a Piece!
10. Trade Knight for Bishop in open Position
"""

console = Console()
md = Markdown(MARKDOWN)
console.print(md)


expl01 = """ 
## 1. d5 Chess Principle: Control the Center.
> **(d4 d5 White moves)**

Black has answered `d4` with `d5` to control the squares `e4` and `c4` and to stop the white center pawn to advance further gaining center space.
"""

expl02 = """ 
## 2. h6 Chess Principle: Earning a FREE MOVE while attacking the INSTABLE BISHOP to make a LUFTLOCH.
[Bg5 h6] Black EARNS the free move (2..h6), where you don't lose a development tempi as you attack the INSTABLE BISHOP, which must retreat, losing a tempi for White.

The move h6 is useful as it controls **g5** and enables the follow up move **g5** later on, followed by the development of the inactive bishop to g7. In other words Black gains a tempi. One tempi = one move. The move h6 is NOT very effective but it is FREE! This is the point. It does not cost you anything, a tempi, I mean, but h6 is a nice little practical move, that improves your chances a little bit! It puts some spice into the game as it enables g5 AND if the bishop goes to h4 then it is cut off from the queenside! It can not come for help to the queenside when the house is on fire.
"""

explanations = [expl01, expl02]

boards = [
    ['d4', 'd5'],
    ['d4', 'd5', 'Bg5', 'h6'],
]

print("\n[yellow]Select a option[/yellow] ([bold]use the number[/bold]):")

input_message = "\tPick an number:"
user_input = input(input_message)


console.print()

explain = explanations[int(user_input)-1]

expl = Markdown(explain)
console.print(expl)


print("\n[bold yellow]Board :lemon:[/bold yellow]\n")

moves = boards[int(user_input)-1]

board = chess.Board()

for m in moves:
    board.push_san(m)
print(board)
