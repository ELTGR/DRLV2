import gym
from gym import spaces
from numpy import inf
import numpy as np

from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete


Observation = Dict(
                    {
                        #X Y Z 
                        'depart' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3,), dtype=np.float32),
                        #X Y Z 
                        'arrive' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3,), dtype=np.float32),
                        #X Y Z 
                        'ma_pos' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3,), dtype=np.float32),
                        #Niveau de batterie 
                        'nivbat' : spaces.Box(low=float("inf"), high=float("inf"), shape=(1,), dtype=np.float32),
                        #X Y Z point de passage des cases
                        'waypnt' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3, 50), dtype=np.float32),
                        #data du velodymne, a determiner 
                        'data_v' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3, 50), dtype=np.float32),
                    }
                  )
# 1 si evitement 0 si scan
Action = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)


