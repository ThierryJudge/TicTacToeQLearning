from Environment import Environment


NUMBER_OF_TESTS = 1000
env = Environment()


wins = 0
losses = 0
draws = 0

for _ in range(NUMBER_OF_TESTS):

    done = False
    state = env.reset()
    reward = env.REWARD_DEFAULT

    while not done:

        action = env.get_sample_action()
        state, reward, done = env.step(action)

    if reward == env.REWARD_WIN:
        wins += 1
    elif reward == env.REWARD_LOSS:
        losses += 1
    elif reward == env.REWARD_DRAW:
        draws += 1


print("Test results")
print("X: " + str(wins) + ", O: " + str(losses) + ", Draw: " + str(draws))



