import json
import numpy as np
from keras.models import model_from_json
from MountainCar import MountainCar


if __name__ == "__main__":
    # Initialize parameters

    # working with any parameters below, not nessecarily the parameters set during the training, it seems like neural network is able to generalize to othe initializations as well
    Xrange = [-1.5, 0.55]
    Vrange = [-0.7, 0.7]
    start = [-0.5, -0.1]
    goal = [0.45]

    with open("model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = MountainCar(start, goal, Xrange, Vrange)

    for e in range(10):
        c = 0
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        c += 1
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            c += 1
        print("Episode %d, Steps %d" %(e, c))