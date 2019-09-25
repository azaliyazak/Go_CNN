import time

import torch
import numpy as np

from const import *
from model.model import GOCNN


def train(net, train, rez, batch_size, n_epochs, learning_rate):
    train_temp = train.copy()
    rez_temp = rez.copy()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    training_start_time = time.time()

    start_time = time.time()

    batch_n = int(np.ceil(len(train) / batch_size))

    batch_loss = 0

    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(batch_n):
            if i == 0:
                x = torch.from_numpy(train_temp[:batch_size]).type(torch.FloatTensor)
                y = torch.from_numpy(np.array(rez_temp[:batch_size])).type(torch.LongTensor)
            elif i + 1 == batch_n:
                x = torch.from_numpy(train_temp[i * batch_size:]).type(torch.FloatTensor)
                y = torch.from_numpy(np.array(rez_temp[i * batch_size:])).type(torch.LongTensor)
            else:
                x = torch.from_numpy(train_temp[i*batch_size:(i+1)*batch_size]).type(torch.FloatTensor)
                y = torch.from_numpy(np.array(rez_temp[i*batch_size:(i+1)*batch_size])).type(torch.LongTensor)

            optimizer.zero_grad()

            output = net(x)
            try:
                loss_val = loss(output, y)
                loss_val.backward()
                optimizer.step()
            except RuntimeError:
                pass

            epoch_loss += loss_val.data.item()
            batch_loss += loss_val.data.item()

            epoch_loss += loss_val.data.item()
            batch_loss += loss_val.data.item()

            if i % 10 == 0:
                print("Epoch {}, batch {}/{} \t batch_loss: {:.2f} took: {:.2f}s".format(epoch + 1, i+1, batch_n,
                                                                                         batch_loss,
                                                                                         time.time() - start_time))
            batch_loss = 0

        print("Epoch {} \t epoch_loss: {:.2f} took: {:.2f}s".format(epoch + 1, epoch_loss, time.time() - start_time))
        start_time = time.time()

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

f = open('corrected_data.txt', 'r+')

lines = f.readlines()
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
            if (i % 2) == 1:
                move = info_to_move(line_splited[i])
                board_state[move[0]][move[1]][color_to_dig['white']] = 1
                board_state[win_move[0]][win_move[1]][color_to_dig['black']] = -1
            elif (i % 2) == 0:
                win_move = info_to_move(line_splited[i])
                data.append(board_state)
                win_moves.append(win_move)

data = np.array(data)
win_moves = np.array(win_moves)

temp = []
for move in win_moves:
    temp.append(move2d_to_1d(move))

win_moves = np.array(temp)

CNN = GOCNN()
train(CNN, data, win_moves, batch_size=16, n_epochs=15, learning_rate=0.001)

torch.save(CNN.state_dict(), 'model/model.pth')
