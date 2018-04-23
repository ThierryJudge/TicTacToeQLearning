import tensorflow as tf
from Environment import *
from ModelBuilder import build_model
import os


tf.reset_default_graph()

layers = [729, 729, 729]
inputs, Q_out, predict = build_model(9, layers, 9)

sess = tf.Session()
env = Environment()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)

checkpoint = tf.train.get_checkpoint_state("model")
if checkpoint and checkpoint.model_checkpoint_path:
    s = saver.restore(sess,checkpoint.model_checkpoint_path)
    print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
    step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
else:
    print("Could not find old network weights")

player = env.O
turn = get_first_turn()
done = False
state = get_new_board()
winner = env.REWARD_DEFAULT

while not done:
    print("------------------------")
    env.draw_board()
    if turn == env.X:
        action = sess.run(predict, feed_dict={inputs: state.reshape(1, 9)})
        action = action[0]
    else:
        action = int(input("Position?"))
    state, winner, done = env.step_player(action, turn)

    if winner != env.REWARD_ILLEGAL:
        turn = turn * -1
    else:
        if turn == player:
            print("Position is occupied: Enter new position")

    if done:
        break

print("------------------------")
env.draw_board()
if winner == player:
    print("Player win")
elif winner == env.DRAW:
    print("Draw")
else:
    print("Computer win")


