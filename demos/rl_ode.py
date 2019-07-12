

'''
We propose a new method of learning to play games.
Instead of discretized timesteps for choosing actions, we evaluate an ODE
that gives the probability of taking an action at a given point in real-time.

We define the reward function as a trajectory leading to the definition of a
win.

Terminology:
We define the game space of a discretized game to be the set of all pairs
of actions and observations.

We must then define the action space as a differential equation wrt time.
The same goes for


'''
