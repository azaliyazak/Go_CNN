from const import *
import numpy as np


def correcting_data(board_state):
    winner = ''
    correct_game = False
    # for white cheking is game finished horizontally and vertically
    for i in range(size - 1):
        if not correct_game:
            game_finished = False
            tokens_in_row = 0
            for j in range(size - 1):
                if board_state[i][j][color_to_dig['white']] == 1:
                    tokens_in_row += 1
                    game_finished = True
                    if tokens_in_row == 5:
                        correct_game = True
                        winner = 'white'
                        break
                elif board_state[i][j][color_to_dig['white']] == 0:
                    tokens_in_row = 0
                    game_finished = False

    if not correct_game:
        for j in range(size - 1):
            if not correct_game:
                game_finished = False
                tokens_in_row = 0
                for i in range(size - 1):
                    if board_state[i][j][color_to_dig['white']] == 1:
                        tokens_in_row += 1
                        game_finished = True
                        if tokens_in_row == 5:
                            correct_game = True
                            winner = 'white'
                            break
                    elif board_state[i][j][color_to_dig['white']] == 0:
                        tokens_in_row = 0
                        game_finished = False

    # for balck cheking is game finished horizontally and vertically
    if not correct_game:
        for i in range(size - 1):
            if not correct_game:
                game_finished = False
                tokens_in_row = 0
                for j in range(size - 1):
                    if board_state[i][j][color_to_dig['black']] == -1:
                        tokens_in_row += 1
                        game_finished = True
                        if tokens_in_row == 5:
                            correct_game = True
                            winner = 'black'
                            break
                    elif board_state[i][j][color_to_dig['black']] == 0:
                        tokens_in_row = 0
                        game_finished = False

    if not correct_game:
        for j in range(size - 1):
            if not correct_game:
                game_finished = False
                tokens_in_row = 0
                for i in range(size - 1):
                    if board_state[i][j][color_to_dig['black']] == -1:
                        tokens_in_row += 1
                        game_finished = True
                        if tokens_in_row == 5:
                            correct_game = True
                            winner = 'black'
                            break
                    elif board_state[i][j][color_to_dig['black']] == 0:
                        tokens_in_row = 0
                        game_finished = False

    # for white

    # check diagonal lines from top-left to bottom-right /
    if not correct_game:
        for i in range(size - 1 - 4):  ##
            for j in range(4, size - 1):  ##
                if board_state[i][j][color_to_dig['white']] == 1 and board_state[i + 1][j - 1][color_to_dig['white']] == \
                        1 and board_state[i + 2][j - 2][color_to_dig['white']] == 1 and board_state[i + 3][j - 3][
                    color_to_dig['white']] \
                        == 1 and board_state[i + 4][j - 4][color_to_dig['white']] == 1:
                    correct_game = True
                    winner = 'white'

    # check diagonal lines from top-right to bottom-left \
    if not correct_game:
        for i in range(size - 1 - 4):  ##
            for j in range(size - 1 - 4):  ##
                if board_state[i][j][color_to_dig['white']] == 1 and board_state[i + 1][j + 1][
                    color_to_dig['white']] == 1 \
                        and board_state[i + 2][j + 2][color_to_dig['white']] == 1 and board_state[i + 3][j + 3][
                    color_to_dig['white']] \
                        == 1 and board_state[i + 4][j + 4][color_to_dig['white']] == 1:
                    correct_game = True
                    winner = 'white'

    # for black

    # check diagonal lines from top-left to bottom-right /
    if not correct_game:
        for i in range(size - 1 - 4):
            for j in range(4, size - 1):
                if board_state[i][j][color_to_dig['black']] == -1 and board_state[i + 1][j - 1][
                    color_to_dig['black']] == \
                        -1 and board_state[i + 2][j - 2][color_to_dig['black']] == -1 and board_state[i + 3][j - 3][
                    color_to_dig['black']] \
                        == -1 and board_state[i + 4][j - 4][color_to_dig['black']] == -1:
                    correct_game = True
                    winner = 'black'

    # check diagonal lines from top-right to bottom-left \
    if not correct_game:
        for i in range(size - 1 - 4):
            for j in range(size - 1 - 4):
                if board_state[i][j][color_to_dig['black']] == -1 and board_state[i + 1][j + 1][
                    color_to_dig['black']] == -1 \
                        and board_state[i + 2][j + 2][color_to_dig['black']] == -1 and board_state[i + 3][j + 3][
                    color_to_dig['black']] \
                        == -1 and board_state[i + 4][j + 4][color_to_dig['black']] == -1:
                    correct_game = True
                    winner = 'black'

    # print('game #', 'res:', correct_game)
    return correct_game, winner

file = open('train_full.renju', 'r+')
corrected_data = open('corrected_data.txt','w+')
lines = file.readlines()

for line in lines:

    board_state = np.zeros((size, size, deepth))
    line_splited = line.split()
    winner_start = line_splited[0]
    if winner_start == 'white':
        for i in range(size - 1):  # saving last row and column empty
            for j in range(size - 1):
                board_state[i][j][deepth - 1] = 1
    elif winner_start == 'black':
        for i in range(size - 1):  # saving last row and column empty
            for j in range(size - 1):
                board_state[i][j][deepth - 1] = -1
    # else:
    # print('ERROR', line_splited)

    # filling last 3rd low (who won)
    for i in range(1, len(line_splited), 2):
        move = info_to_move(line_splited[i])  # for white
        board_state[move[0]][move[1]][color_to_dig['white']] = 1
    for i in range(2, len(line_splited), 2):
        move = info_to_move(line_splited[i])  # for black
        # print(move)
        board_state[move[0]][move[1]][color_to_dig['black']] = -1
        # print(board_state[:,:,color_to_dig['black']])
        # print(dig_to_let[move[1] + 1], move[0] + 1)
        # print()

    is_game_finished, winner = correcting_data(board_state)

    # если игра закончена есть два варианта-либо победитель совпал с изначальными данными либо нет(надо поменять)

    if is_game_finished:
        str1 = ' '.join(line_splited[1:])
        corrected_data.write(winner + ' ' + str1 + '\n')

corrected_data.close()

# --------данные теперь оьбработаны и верны-----------------