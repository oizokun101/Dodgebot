import numpy as np
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
import math # use of pi.
import random
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped 


MAP_TOPIC = 'moving_map'
USE_SIM_TIME = True
TF_ODOM_LINK = 'map'

RESOLUTION = 1.0  # meters
GRID_HEIGHT = 10
GRID_WIDTH = 10
FREQUENCY = 1  # Hz

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

# Q-learning parameters 
LEARNING_RATE = 0.1 
DISCOUNT_FACTOR = 0.9
EPSILON = 0.3
EPISODES = 1000
MAX = 100 
# Actions 
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4
# REWARDS
GOAL = 100
COLLISION = -100
PENALTY = -1

FREQUENCY = 10

# Velocities that will be used 
LINEAR_VELOCITY = 0.1 # m/s
ANGULAR_VELOCITY = math.pi/8 # rad/s


class QLearningAlgortihm(Node):
    def __init__(self):
       super().__init__('q_learning_node')
       

       self.q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, 5))
       self.grid = None
       self.start = (1,1)
       self.goal = (7,8)
       self.actions = [UP, DOWN, LEFT, RIGHT, STAY]
       self.current_episode = 0
       
       self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), FREE)

       self.grid[0, :] = OCCUPIED
       self.grid[-1, :] = OCCUPIED
       self.grid[:, 0] = OCCUPIED
       self.grid[:, -1] = OCCUPIED
       
       self.resolution = RESOLUTION
       self.height = GRID_HEIGHT
       self.width = GRID_WIDTH
        
       self.origin_x = - (self.width / 2) * self.resolution
       self.origin_y = - (self.height / 2) * self.resolution
   
       
       self.path_pub = self.create_publisher(Path, '/q_learning/path', 10)
       self.obstacle_sub = self.create_subscription(PoseWithCovarianceStamped, '/obstacle_pose', self.obstacle_pose_callback, 10)
       
       self.obstacle_positions = {'obstacle_one': [3.0 ,3.0], 'obstacle_two': [0.0, 0.0], 'obstacle_three': [-3.0 ,-3.0]}
       
       self._map_pub = self.create_publisher(OccupancyGrid, MAP_TOPIC, 1)
    
       self.timer = self.create_timer(1.0 / FREQUENCY, self.qlearn_function)
    
    def obstacle_pose_callback(self, msg):
        """Get the obstacle positions"""
        obs_name = msg.header.frame_id 
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.obstacle_positions[obs_name] = [x, y]
        self.update_grid() 
    
    def update_grid(self): 
        """Update the grid with the obstacles"""
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH), FREE)
        self.grid[0, :] = OCCUPIED
        self.grid[-1, :] = OCCUPIED
        self.grid[:, 0] = OCCUPIED
        self.grid[:, -1] = OCCUPIED
        for pose in self.obstacle_positions.values():
            col = int(pose[0] + 5.0)
            row = int(pose[1] + 5.0)
            if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
                self.grid[row,col] = OCCUPIED
        self.publish_map()

        
    def index_of_state(self, state):
        """Convert state to index for q table"""
        row, col = state 
        return row * GRID_WIDTH + col
    
    def is_state_valid(self, state):
        """Check if state is valid if it's in bounds and not an obstacle"""
        row, col = state
        if row < 0 or row >= GRID_HEIGHT or col < 0 or col >= GRID_WIDTH or self.grid[row, col] == OCCUPIED:
            return False 
        return True
    
    def get_new_state(self, state, action):
        """Get the new state"""
        row, col = state
        if action == UP:
            new_state = (row - 1, col) 
        elif action == DOWN:
            new_state = (row + 1, col) 
        elif action == LEFT:
            new_state = (row, col - 1) 
        elif action == RIGHT:
            new_state = (row, col + 1) 
        else: 
            new_state = (row, col) 
        return new_state
    
    def reward_calc(self, state):
        """Get the reward"""
        if state == self.goal: 
            return GOAL
        if not self.is_state_valid(state):
            return COLLISION
        return PENALTY
    
    def get_action(self, index):
        """Get the best action  based on q table"""
        if random.random() < EPSILON:
            return random.randint(UP,STAY) 
        return np.argmax(self.q_table[index])
    
    def qlearn_function(self):
        """Function to run the q learning alg""" 
        
        self.update_grid()
        
        if self.current_episode >= EPISODES:
            self.timer.cancel()
            print("Done training")
            path = self.get_best_path()
            if path:
                self.publish_path(path)
            else:
                print("No valid path")
            return 
     
        state = self.start 
        state_index = self.index_of_state(state) 
        for iteration in range(MAX):
            action = self.get_action(state_index)
            new_state = self.get_new_state(state, action)
            new_state_index = self.index_of_state(new_state)
            reward = self.reward_calc(new_state)
            if self.is_state_valid(new_state):
                self.q_table[state_index, action] = (1 - LEARNING_RATE) * self.q_table[state_index, action] +  LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(self.q_table[new_state_index]))
                state = new_state
                state_index = new_state_index
            else: 
                self.q_table[state_index, action] = (1 - LEARNING_RATE) * self.q_table[state_index, action] + LEARNING_RATE * reward
                break 
            if state == self.goal: 
                break 
            
        self.current_episode += 1
        self.publish_map()
                
    def get_best_path(self): 
        """Function to get the best q-table path""" 
        self.update_grid()
        best_path = [self.start]
        state = self.start 
        state_index = self.index_of_state(state) 
        count = 0
        while state != self.goal and count < MAX:
            action = np.argmax(self.q_table[state_index])
            new_state = self.get_new_state(state, action)
            if not self.is_state_valid(new_state):
                print("No path found")
                return []
            best_path.append(new_state) 
            state = new_state
            state_index = self.index_of_state(state) 
            count += 1
        if state == self.goal:
            return best_path
        else: 
            print("No path found")
            return []
    
    def publish_path(self, path_list): 
        print(path_list)
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path_list:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(point[1] - 5.0 + 0.5)
            pose.pose.position.y = float(point[0] - 5.0 + 0.5) 
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)
        self.publish_map()

    def publish_map(self):
  
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info.resolution = self.resolution
        msg.info.height = self.height
        msg.info.width = self.width
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = self.grid.flatten().tolist()
        self._map_pub.publish(msg)

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    
    

    q_node = QLearningAlgortihm()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(q_node)
    try:
        executor.spin()
    except KeyboardInterrupt: 
        pass
    finally:
        q_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

  