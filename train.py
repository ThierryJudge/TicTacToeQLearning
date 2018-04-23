import tensorflow as tf
from Environment import Environment
import numpy as np
import random
from ModelBuilder import build_model
import os


# returns list of [state, action, reward, next_state] for a full game
def play_game(sess, env, input, predict):
    global wins
    global losses
    global draws
    global e

    done = False
    state = env.reset()
    reward = env.REWARD_DEFAULT

    game_memory = []

    while not done:

        action = sess.run(predict, feed_dict={input: state.reshape(1, 9)})
        random_action = env.get_sample_action(True)

        if np.random.rand(1) < e:
            action = random_action

        # new_state, reward, done = env.step(action)  # random opponent
        new_state, reward, done = env.step_sess(action,sess, inputs, predict, train=True, random_prob=0.6)  # self opponent

        turn_memory = []
        turn_memory.append(np.copy(state))
        turn_memory.append(action)
        turn_memory.append(reward)
        turn_memory.append(np.copy(new_state))

        game_memory.append(turn_memory)

        state = new_state

        if done:
            break

    if reward == env.REWARD_WIN:
        wins += 1
    elif reward == env.REWARD_LOSS:
        losses += 1
    elif reward == env.REWARD_DRAW:
        draws += 1

    return game_memory


def run_test_games(num_tests, sess, env, input, predict, random_prob=0):
    wins = 0
    losses = 0
    draws = 0
    illegal_count = 0

    for _ in range(num_tests):

        done = False
        state = env.reset()
        state = state.reshape(1, 9)
        reward = env.REWARD_DEFAULT

        while not done:

            action = sess.run(predict, feed_dict={inputs: state})
            #state, reward, done = env.step(action)
            state, reward, done = env.step_sess(action,sess, input, predict, train=False, random_prob=random_prob)
            if reward == Environment.REWARD_ILLEGAL and not done:
                illegal_count += 1
                action = env.get_sample_action(True)
                state, reward, done = env.step(action)

            state = state.reshape(1, 9)

        if reward == env.REWARD_WIN:
            wins += 1
        elif reward == env.REWARD_LOSS:
            losses += 1
        elif reward == env.REWARD_DRAW:
            draws += 1

    print("Test results: random_prob: " + str(random_prob))
    print("X: " + str(wins) + ", O: " + str(losses) + ", Draw: " + str(draws))
    print("Illegal: " + str(illegal_count))


wins = 0
losses = 0
draws = 0

y = .90
e_initial = 0.9
e_downrate = 0.005
e = 0.9
learning_rate = 1e-4

EPOCHS = 10  # number of training iterations over each game list
EPISODES = 100  # number of games in each game list

tf.reset_default_graph()

layers = [729, 729, 729]
inputs, Q_out, predict = build_model(9, layers, 9)


nextQ = tf.placeholder(shape=[None, 9], dtype=tf.float32)
loss = tf.reduce_max(tf.square(tf.subtract(nextQ, Q_out)))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
env = Environment()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)


total_cost = 0
iteration = 0
step = 0

checkpoint = tf.train.get_checkpoint_state("model")
if checkpoint and checkpoint.model_checkpoint_path:
    s = saver.restore(sess,checkpoint.model_checkpoint_path)
    print("Loaded model:", checkpoint.model_checkpoint_path)
    step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
else:
    print("Can't find model")
iteration += step

while True:
    wins = 0
    losses = 0
    draws = 0

    game_list = []
    for _ in range(EPISODES):
        game_list.append(play_game(sess, env, inputs, predict))


    total_cost = 0

    for _ in range(EPOCHS):
        random.shuffle(game_list)

        for game in game_list:
            game_cost = 0
            game_length = len(game)
            turn_count = game_length
            game_reward = 0

            while turn_count > 0:
                turn = game[turn_count-1]
                state = turn[0].reshape(1, 9)
                action = turn[1]
                reward = turn[2]
                next_state = turn[3].reshape(1, 9)

                if turn_count == game_length:
                    game_reward = reward
                else:
                    new_Q_values = sess.run(Q_out, feed_dict={inputs: next_state})

                    max_Q = np.max(new_Q_values)
                    game_reward = reward + y * max_Q

                target_Q = sess.run(Q_out, feed_dict={inputs: state})

                for index, item in enumerate(state.reshape(9)):
                    if item != 0:
                        target_Q[0, index] = Environment.REWARD_ILLEGAL

                target_Q[0, action] = game_reward

                _, cost = sess.run([train, loss],
                                   feed_dict={inputs: state, nextQ: target_Q})
                total_cost += cost
                game_cost += cost
                turn_count -= 1

    print("Iteration : " + str(iteration) + ", X: " + str(wins) + ", O: " + str(losses) +
          ", Draw: " + str(draws) + ", e: " + str(e) + ", Cost: " + str(total_cost))

    iteration += 1

    if iteration % 10 == 0:
        run_test_games(100, sess, env, inputs, predict)
        run_test_games(100, sess, env, inputs, predict, random_prob=1)
        saver.save(sess, "./model/model.ckpt", global_step=iteration)

    if e > -0.2:
        e -= e_downrate
    else:
        e = random.choice([0.1, 0.05, 0.06, 0.07, 0.15, 0.03, 0.20, 0.25, 0.5, 0.4])