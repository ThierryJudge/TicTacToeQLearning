import numpy as np
import random

X = 1
O = -1
DRAW = 2


class Environment:
    REWARD_WIN = 1
    REWARD_LOSS = -1
    REWARD_DRAW = 0.5
    REWARD_DEFAULT = 0
    REWARD_ILLEGAL = -1

    X = X
    O = O
    DRAW = DRAW

    def __init__(self):
        self.board = get_new_board()

    def reset(self):
        self.board = get_new_board()

        turn = get_first_turn()
        if turn == self.O:
            o_action = self.get_sample_action(True)
            self.board[o_action] = self.O

        return self.board

    def draw_board(self):
        draw(self.board)

    def check_win(self):
        return check_win(self.board)

    def get_sample_action(self, force_legal=False):
        if force_legal:
            available_positions = []
            for index, item in enumerate(self.board):
                if item == 0:
                    available_positions.append(index)

            return random.choice(available_positions)
        else:
            return random.randint(0, 8)

    # return observation, reward, done
    def step(self, action, train=False):
        if self.board[action] != 0:
            return self.board, self.REWARD_ILLEGAL, train
        else:
            self.board[action] = self.X
            w = self.check_win()

            if w == self.X:
                return self.board, self.REWARD_WIN, True
            elif w == self.O:
                return self.board, self.REWARD_LOSS, True
            elif w == self.DRAW:
                return self.board, self.REWARD_DRAW, True
            else:
                o_action = self.get_sample_action(True)
                self.board[o_action] = self.O
                w = self.check_win()

                if w == self.O:
                    return self.board, self.REWARD_LOSS, True
                elif w == self.DRAW:
                    return self.board, self.REWARD_DRAW, True
                elif w == self.X:
                    return self.board, self.REWARD_WIN, True
                else:
                    return self.board, self.REWARD_DEFAULT, False

    # return observation, reward, done
    def step_sess(self, action, sess, inputs, predict, random_prob = 0,  train=False):
        if self.board[action] != 0:
            return self.board, self.REWARD_ILLEGAL, train
        else:
            self.board[action] = self.X
            w = self.check_win()

            if w == self.X:
                return self.board, self.REWARD_WIN, True
            elif w == self.O:
                return self.board, self.REWARD_LOSS, True
            elif w == self.DRAW:
                return self.board, self.REWARD_DRAW, True
            else:
                o_action = self.session_play(sess, inputs, predict, random_prob)
                self.board[o_action] = self.O
                w = self.check_win()

                if w == self.O:
                    return self.board, self.REWARD_LOSS, True
                elif w == self.DRAW:
                    return self.board, self.REWARD_DRAW, True
                elif w == self.X:
                    return self.board, self.REWARD_WIN, True
                else:
                    return self.board, self.REWARD_DEFAULT, False

    def session_play(self, sess, inputs, predict, random_prob):
        temp_board = self.board * -1
        action = sess.run(predict, feed_dict={inputs: temp_board.reshape(1, 9)})
        action = action[0]
        random_action = self.get_sample_action(True)

        if np.random.rand(1) < random_prob:
            action = random_action

        if temp_board[action] != 0:
            action = self.get_sample_action(True)

        return action

    # return observation, reward, done
    def step_minimax(self, action):
        if self.board[action] != 0:
            return self.board, self.REWARD_ILLEGAL, False
        else:
            self.board[action] = self.X
            w = self.check_win()

            if w == self.X:
                return self.board, self.REWARD_WIN, True
            elif w == self.O:
                return self.board, self.REWARD_LOSS, True
            elif w == self.DRAW:
                return self.board, self.REWARD_DRAW, True
            else:
                _, o_action = self.minimax(self.O)
                self.board[o_action] = self.O
                w = self.check_win()

                if w == self.O:
                    return self.board, self.REWARD_LOSS, True
                elif w == self.DRAW:
                    return self.board, self.REWARD_DRAW, True
                elif w == self.X:
                    return self.board, self.REWARD_WIN, True
                else:
                    return self.board, self.REWARD_DEFAULT, False

    def minimax(self, player):
        if player == self.O:
            best_value = -11
        else:
            best_value = 11

        c = self.check_win()
        if c != 0:
            if c == self.X:
                return 10, None
            elif c == self.O:
                return -10, None
            else:
                return 0, None
        else:
            for i in range(9):
                if self.board[i] == 0:
                    self.board[i] = player
                    value, _ = self.minimax(player * -1)
                    self.board[i] = 0

                    if player == self.O:
                        if value > best_value:
                            best_value, best_move = value, i
                    else:
                        if value < best_value:
                            best_value, best_move = value, i

        return best_value, best_move

    # return observation, winner, done
    # turn : X or O
    def step_player(self, action, turn):
        if self.board[action] != 0:
            return self.board, self.REWARD_ILLEGAL, False
        else:
            self.board[action] = turn
            w = self.check_win()

            if w == self.X:
                return self.board, self.X, True
            elif w == self.O:
                return self.board, self.O, True
            elif w == self.DRAW:
                return self.board, self.DRAW, True
            else:
                return self.board, self.REWARD_DEFAULT, False


def get_new_board():
    return np.zeros(9)


def get_first_turn():
    return random.choice([X, O])


def draw(board, numbers=False):
    char_board = []
    for i in range(len(board)):
        if board[i] == 0:
            if numbers:
                char_board.append(str(i))
            else:
                char_board.append(" ")
        elif board[i] == X:
            char_board.append("X")
        elif board[i] == O:
            char_board.append("O")

    print('   |   |')
    print(' ' + char_board[0] + ' | ' + char_board[1] + ' | ' + char_board[2])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + char_board[3] + ' | ' + char_board[4] + ' | ' + char_board[5])
    print('   |   |')
    print('-----------')
    print('   |   |')
    print(' ' + char_board[6] + ' | ' + char_board[7] + ' | ' + char_board[8])
    print('   |   |')


def check_win(board):
    if board[0] + board[1] + board[2] == 3 * X:
        return X
    if board[0] + board[1] + board[2] == 3 * O:
        return O
    if board[3] + board[4] + board[5] == 3 * X:
        return X
    if board[3] + board[4] + board[5] == 3 * O:
        return O
    if board[6] + board[7] + board[8] == 3 * X:
        return X
    if board[6] + board[7] + board[8] == 3 * O:
        return O

    if board[0] + board[3] + board[6] == 3 * X:
        return X
    if board[0] + board[3] + board[6] == 3 * O:
        return O
    if board[1] + board[4] + board[7] == 3 * X:
        return X
    if board[1] + board[4] + board[7] == 3 * O:
        return O
    if board[2] + board[5] + board[8] == 3 * X:
        return X
    if board[2] + board[5] + board[8] == 3 * O:
        return O

    if board[0] + board[4] + board[8] == 3 * X:
        return X
    if board[0] + board[4] + board[8] == 3 * O:
        return O
    if board[2] + board[4] + board[6] == 3 * X:
        return X
    if board[2] + board[4] + board[6] == 3 * O:
        return O

    if np.count_nonzero(board) == 9:
        return DRAW

    return 0