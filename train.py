import time

import torch
import numpy as np

#  на кагле не забыть импорты скопировать
from const import *
from model.model import GOCNN


def preprocess(lines):
    data = []
    win_moves = []

    for line in lines:
        line_splited = line.split()
        winner = line_splited[0]

        if winner == 'white':
            board_state = np.zeros((size, size, deepth))  # заполнение последнего сдоя победителем
            for i in range(size - 1):  # saving last row and column empty
                for j in range(size - 1):
                    board_state[i][j][deepth - 1] = 1

            for i in range(1, len(line_splited)):
                if (i % 2) == 1:
                    win_move = info_to_move(line_splited[i])
                    data.append(board_state)
                    win_moves.append(win_move)
                elif (i % 2) == 0:
                    move = info_to_move(line_splited[i])
                    board_state[move[0]][move[1]][color_to_dig['black']] = -1
                    board_state[win_move[0]][win_move[1]][color_to_dig['white']] = 1

        if winner == 'black':
            board_state = np.zeros((size, size, deepth))
            for i in range(size - 1):  # saving last row and column empty
                for j in range(size - 1):
                    board_state[i][j][deepth - 1] = -1

            for i in range(1, len(line_splited)):
                if (i % 2) == 0:
                    move = info_to_move(line_splited[i])
                    board_state[move[0]][move[1]][color_to_dig['white']] = 1
                    board_state[win_move[0]][win_move[1]][color_to_dig['black']] = -1
                elif (i % 2) == 1:
                    win_move = info_to_move(line_splited[i])
                    data.append(board_state)
                    win_moves.append(win_move)

    data = np.array(data)
    win_moves = np.array(win_moves)

    temp = []
    for move in win_moves:
        m = move2d_to_1d(move)
        temp.append(move2d_to_1d(move))

    win_moves = np.array(temp)

    return data, win_moves


def train(net, batch_size, n_epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    training_start_time = time.time()

    start_time = time.time()

    batch_loss = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        with open('/kaggle/input/corrrr/corrected_data.txt') as f:
            lines = []
            i = 0
            for line in f:
                i += 1
                if i <= batch_size:
                    lines.append(line)
                    continue
                else:
                    i = 0
                data, win_moves = preprocess(lines)
                x = torch.from_numpy(data).type(torch.FloatTensor)
                y = torch.from_numpy(np.array(win_moves)).type(torch.LongTensor)

                optimizer.zero_grad()

                output = net(x)

                loss_val = loss(output, y)
                loss_val.backward()
                optimizer.step()

                epoch_loss += loss_val.data.item()
                batch_loss += loss_val.data.item()

                if i % 10 == 0:
                    print("Epoch {} \t batch_loss: {:.2f} took: {:.2f}s".format(epoch + 1, batch_loss,
                                                                                time.time() - start_time))

                batch_loss = 0
                lines = []
            start_time = time.time()

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


CNN = GOCNN()
train(CNN, batch_size=4096, n_epochs=15, learning_rate=0.001)

torch.save(CNN.state_dict(), 'model/model_2.pth')
