import numpy as np
from collections import defaultdict
import math

# Direction definitions
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
SAME = 4

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
  
class Location:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
class RobotInfo:
  def __init__(self, location, direction, goal_loc, scan_radius, map_height, map_width, q_table = None):
    self.location = location
    self.direction = direction
    self.goal_loc = goal_loc
    self.scan_radius = scan_radius
    self.map_height = map_height
    self.map_width = map_width

    self.current_state = None

    self.q_table = defaultdict(lambda: 0)

  # This will have to be changed when we make it actually process laser scans
  def get_state(self, map):
    radius = map[max(0, self.goal_loc.y - self.scan_radius):min(self.map_height-1, self.goal_loc.y + self.scan_radius), 
               max(0, self.goal_loc.x - self.scan_radius):min(self.map_width-1, self.goal_loc.x + self.scan_radius)]
    
    scan = self.scan(radius)
    
    self.current_state = State(scan, self.get_goal_dir())
    return self.current_state
    
  def scan(self, radius,  num_rays=60):
    scan = np.ones_like(radius) * -1 # start as unknown

    cx, cy = self.location.x, self.location.y

    h, w, = radius.shape

    for angle in np.linspace(0, 2 * np.pi, num=num_rays, endpoint=False):
        dx = math.cos(angle)
        dy = math.sin(angle)

        x, y = cx, cy  # start exactly at the robot's position

        for step in range(max(h, w) * 4):  # march far enough
            ix, iy = int(x), int(y)
            if not (0 <= ix < h and 0 <= iy < w):
                break

            if radius[ix, iy] == 100:
                break

            scan[ix, iy] = 0  # mark visible
            x += dx * 0.2  # smaller steps for smoother rays
            y += dy * 0.2 

    return scan

    

  def get_goal_dir(self):
    dx = self.goal_loc.x - self.location.x
    dy = self.goal_loc.y - self.location.y

    if abs(dx) > abs(dy):  # horizontal move dominates
        return RIGHT if dx > 0 else LEFT
    elif abs(dy) > 0:  # vertical move dominates
        return DOWN if dy > 0 else UP
    else:
        return SAME 
    
  def turn_dir(self, turn_dir):
    if turn_dir == TURN_LEFT:
      self.direction = (self.direction - 1) % 4
    elif turn_dir ==TURN_RIGHT:
      self.direction = (self.direction + 1) % 4
    
  def move(self, direction):
    if direction == FORWARD:
      step = 1
    elif direction ==  BACKWARD:
      step = -1
    else:
      return
    
    if self.direction == RIGHT:
      self.location.x = max(0, min(self.location.x + step, self.map_width))
    elif self.direction == LEFT:
      self.location.x = max(0, min(self.location.x - step, self.map_width))
    elif self.direction == UP:
      self.location.y = max(0, min(self.location.y + step, self.map_height))
    elif self.direction == DOWN:
      self.location.y = max(0, min(self.location.y - step, self.map_height))

  def get_best_action(self):
    actions = [FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP]
    return max([self.q_table[(self.current_state, action)] for action in actions])


