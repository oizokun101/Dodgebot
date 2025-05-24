import numpy as np

# Direction definitions
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3

# Action definitions
FORWARD = 1
BACKWARD = -1
STOP = 0
TURN_LEFT = 2
TURN_RIGHT = 3

# Reward definitions
GOAL_REWARD = 20
CLOSER_REWARD = 0.2
TIMESTEP_REWARD = -0.1
COLLISION_REWARD = -30

class State:
  def __init__(self, radius_map, goal_dir):
    self.radius_map = radius_map
    self.goal_dir = goal_dir
  
  def __eq__(self, other):
    return (self.radius_map == other.radius_map).all() and self.goal_dir == other.goal_dir
  
  def __hash__(self):
    return hash((self.radius_map.tobytes(), self.goal_dir))
  

  

