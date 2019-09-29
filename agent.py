import torch
import numpy as np
from model.model import predict, GOCNN, decode
from const import *

PATH = "model/model_3.pth"

print("PLEASE, WRITE YOUR PLAY COLOUR: ")

white = [0] * 16
black = [0] * 16
for i in range(16):
    white[i] = [0] * 16
    black[i] = [0] * 16
while True:
    computer_color = input()
    if computer_color.lower() == 'black':
        computer_color = 1
        computer_matrix = white
        opponent_color = -1
        opponent_matrix = black
        break
    elif computer_color.lower() == 'white':
        computer_color = -1
        computer_matrix = black
        opponent_color = 1
        opponent_matrix = white
        break
    else:
        print("PLEASE, WRITE BLACK OR WHITE:")


def show_board(white, black):
    print("    " + "1  " + "2  " + "3  " + "4  " + "5  " + "6  " + "7  " \
          + "8  " + "9  " + "10 " + "11 " + "12 " + "13 " + "14 " + "15 ")
    print("    " + "---------------------------------------------")
    print()
    for i in range(15):
        s = ""
        for j in range(15):
            if black[i][j] == -1:
                s += "-1 "
                continue
            elif white[i][j] == 1:
                s += "1  "
                continue
            else:
                s += "0  "
        print(dig_to_let[i + 1] + "|   " + s + "\n")


def check_win(array):
    array = np.array(array)
    for i in range(array.shape[0]):
        k = 0
        for j in range(array.shape[1]):
            if array[i][j] == 1 or array[i][j] == -1:
                k += 1
                if k == 5:
                    return 1
            else:
                k = 0

    for i in range(array.shape[1]):
        k = 0
        for j in range(array.shape[0]):
            if array[j][i] == 1 or array[j][i] == -1:
                k += 1
                if k == 5:
                    return 1
            else:
                k = 0

    diags = [array[::-1, :].diagonal(i) for i in range(-array.shape[0] + 1, array.shape[1])]

    diags.extend(array.diagonal(i) for i in range(array.shape[1] - 1, -array.shape[0], -1))
    for diag in diags:
        k = 0
        for i in diag:
            if i == 1 or i == -1:
                k += 1
                if k == 5:
                    return 1
            else:
                k = 0
    return 0


def main():
    model = GOCNN()
    model.load_state_dict(torch.load(PATH, map_location='cpu'))
    model.eval()

    if computer_color == 1:
        move = predict(model, [white, black, [[computer_color] * 16] * 16])
        move = decode(np.argmax(move).data.item())
        print(move)
        computer_matrix[let_to_dig[move[0]] - 1][int(move[1:]) - 1] = computer_color
        show_board(white, black)

    while True:
        opponents_move = input()

        if (opponents_move.lower() == "finish"):
            print("THE GAME WAS OVER YOU PREMATURELY")
            return 0
        if ('a' <= opponents_move[0] <= 'p') and (opponents_move[0] != 'i') and (1 <= int(opponents_move[1:]) <= 15):

            if black[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] == 0 and \
                    white[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] == 0:
                opponent_matrix[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] = opponent_color
            else:
                print("CELL IS BUSY")
                while black[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] != 0 or \
                        white[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] != 0:
                    opponents_move = input()
                    if black[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] == 0 and \
                            white[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] == 0:
                        opponent_matrix[let_to_dig[opponents_move[0]] - 1][int(opponents_move[1:]) - 1] = opponent_color
                        break
                    else:
                        print("CELL IS BUSY")
            if check_win(opponent_matrix) == 1:
                print('opponent(human) is winner!')
                return 0
            move = predict(model, [white, black, [[computer_color] * 16] * 16])
            indexes = np.argsort(move).data.tolist()
            for i in indexes:
                cell = decode(i)
                if black[let_to_dig[cell[0]] - 1][int(cell[1:]) - 1] == 0 and \
                        white[let_to_dig[cell[0]] - 1][int(cell[1:]) - 1] == 0:
                    computer_matrix[let_to_dig[cell[0]] - 1][int(cell[1:]) - 1] = computer_color
                    break
            print(cell)

            if check_win(computer_matrix) == 1:
                print('computer is winner!')
                return 1

            show_board(white, black)
        else:
            print("PLEASE, WRITE CORRECT MOVE")

main()
