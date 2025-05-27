#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np

MAP_TOPIC = 'moving_map'
USE_SIM_TIME = True
TF_ODOM_LINK = 'odom'

RESOLUTION = 0.05  # meters
GRID_HEIGHT = 10
GRID_WIDTH = 10
FREQUENCY = 10  # Hz

UNKNOWN = -1
FREE = 0
OCCUPIED = 100

class DynamicRect:
    """Minimal moving 2x2 rectangle."""
    def __init__(self, row, col, d_row, d_col):
        self.row, self.col = row, col        # current integer cell
        self.d_row, self.d_col = d_row, d_col  # ±1 step per tick

    def step(self, height, width):
        """Advance one step then bounce when hitting a wall."""
        self.row += self.d_row
        self.col += self.d_col
        # Bounce on vertical walls
        if self.col <= 0 or self.col >= width-2:
            self.d_col *= -1
            self.col  += self.d_col          # move back inside
        # Bounce on horizontal walls
        if self.row <= 0 or self.row >= height-2:
            self.d_row *= -1
            self.row  += self.d_row

class MovingMap(Node):
    def __init__(self, node_name='moving_obstacle_map', context=None):
        super().__init__(node_name, context=context)

        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])

        # Setting up the publisher to send map messages.
        self._map_pub = self.create_publisher(OccupancyGrid, MAP_TOPIC, 1)

        # Initializing grid with default values
        self.resolution = RESOLUTION
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH

        # Rate at which to operate the while loop.
        self.rate = self.create_rate(FREQUENCY)

        self._init_map()

    def _init_map(self):
        self.grid: np.ndarray = np.full((self.height, self.width), UNKNOWN)
        self.origin_x = - (self.width // 2) * self.resolution
        self.origin_y = - (self.height // 2) * self.resolution
        self.obstacles = [
            DynamicRect(self.height-2, self.width-2,  0, -1),   # right → left
            DynamicRect(self.height//2, self.width//2, 1,  0),   # middle ↓ up
            DynamicRect(2, 2,  0,  1)                   # left → right
        ]
    
    def make_header(clock):
        hdr = Header()
        hdr.stamp = clock.now().to_msg()
        hdr.frame_id = TF_ODOM_LINK
        return hdr

    def draw_obstacles(self):
      self.grid.fill(0)
      for obs in self.obstacles:
          r, c = obs.row, obs.col            # top-left
          self.grid[r  : r+2, c  : c+2] = 100   # 2×2 slice

    def publish_grid(self):
        # move every rectangle one cell
        for obs in self.obstacles:
            obs.step(self.height, self.width)
        
        self.draw_obstacles()
        msg = OccupancyGrid()
        msg.header = self.make_header(self.get_clock())

        msg.info.resolution = self.resolution
        msg.info.height = self.height
        msg.info.width = self.width
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = self.grid.flatten().tolist()
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MovingMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
