let_to_dig = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
              'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15}

dig_to_let = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h',
              9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p'}

color_to_dig = {'white': 0, 'black': 1} # layers

# Tensor
size = 16
deepth = 3


def info_to_move(str_move):
    y, x = let_to_dig[str_move[0]] - 1, int(str_move[1:]) - 1
    move = (x, y)
    return move


def move2d_to_1d(move):
    return (move[0]+1) * 15 + move[1]
