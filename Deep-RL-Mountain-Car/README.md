#Deep RL - Mountain Car Domain
This is my implementation of Mountain Car domain in reinforcement learning using neural network function approximation with Keras Deep Learning Library.  
*To the best of my knowledge, this is the first opensource code for solving Mountain Car RL problem using DQN.*    
I am motivated by this simple example of [Keras playing catch](https://edersantana.github.io/articles/keras_rl/)  
My code is adapted from the above incorporated with Mountain car domain.

### DQN implementation
DQN implementation is based on the paper:  
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

### Mountain Car Domain
Mountain car is standard platform for testing RL algorithms in which a underpowered car tries to reach a goal position uphill by moving to and fro the hill valley. The state space of the car is continuous and consist of its position and velocity. At every state, it can choose out of 3 possible actions -- move forward, backward or stay. Refer to this [Wikipedia article](https://en.wikipedia.org/wiki/Mountain_Car) for more information.  

![alt tag](Mcar.png) 

The figure above (from Wikipedia) shows the problem, where car in starting position and star is the goal position.

### Files
1. MountainCar.py -- Define the class of Mountain Car environment - transition from one state to another given an action and returning reward.
2. MCqlearn.py -- DQN implementation for Q-learning.
3. MCtest.py -- Testing the learned policy.

### Training
DQN is trained for 1000 successful episodes of the problem. The specific parameters of the algorithm are given in the MCqlearn.py file. To train the DQN network, symply run the training file:
```
python MCqlearn.py
```
After training, the network parameters are stored in .json and .h5 file.

### Testing
Once the network is trained and parameters are saved in .json and .h5 file, testing can be done. To test the network, run the file:
```
python MCqtest.py
```
The initial state and other parameters of Mountain Car domain can be set up in this file.  
**It is interesting to note that though the network is trained only for one initial state and one range of Mountain Car domain, it is able to generalize and success during testing for arbitrary initial states and range of the domain.**

### Dependencies
1. Python3
2. Keras
3. Numpy 




