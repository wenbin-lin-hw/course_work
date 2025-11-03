from controller import Robot
from datetime import datetime
import math
import numpy as np


class Controller:
    def __init__(self, robot):
        # Robot Parameters
        self.robot = robot
        self.time_step = 32  # ms
        self.max_speed = 1  # m/s

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

        # Data
        self.inputs = []
        self.inputsPrevious = []

        # Flag
        self.flag_turn = 0

        # ========== NEW: Obstacle Avoidance State Machine ==========
        # States: 'LINE_FOLLOWING', 'OBSTACLE_DETECTED', 'AVOIDING_LEFT',
        #         'AVOIDING_RIGHT', 'SEARCHING_LINE'
        self.state = 'LINE_FOLLOWING'
        self.obstacle_side = None  # 'left' or 'right'
        self.avoidance_counter = 0  # Counter for avoidance maneuvers
        self.search_counter = 0  # Counter for searching the line

        # Obstacle detection thresholds
        self.OBSTACLE_THRESHOLD = 0.15  # Distance sensor threshold for obstacle
        self.OBSTACLE_CLOSE_THRESHOLD = 0.3  # Very close obstacle

        # Avoidance parameters
        self.AVOIDANCE_DURATION = 50  # Steps to move around obstacle
        self.SEARCH_DURATION = 100  # Steps to search for line
        self.PARALLEL_DURATION = 30  # Steps to move parallel to obstacle

        print("Controller initialized with obstacle avoidance capability")

    def clip_value(self, value, min_max):
        if (value > min_max):
            return min_max;
        elif (value < -min_max):
            return -min_max;
        return value;

    # ========== NEW: Obstacle Detection Function ==========
    def detect_obstacle(self):
        """
        Detect obstacles using proximity sensors

        Returns:
            tuple: (has_obstacle, obstacle_side, obstacle_distance)
                - has_obstacle: Boolean indicating if obstacle is detected
                - obstacle_side: 'front', 'left', 'right', or None
                - obstacle_distance: normalized distance value [0, 1]

        Sensor layout:
        ps0, ps1: front-right
        ps2: right side
        ps5: left side
        ps6, ps7: front-left
        """
        # Front sensors (ps0, ps1, ps6, ps7)
        front_left = max(self.inputs[6], self.inputs[7])  # ps6, ps7
        front_right = max(self.inputs[3], self.inputs[4])  # ps0, ps1
        front_max = max(front_left, front_right)

        # Side sensors
        right_side = self.inputs[5]  # ps2
        left_side = self.inputs[8]  # ps5

        # Check for obstacles
        has_obstacle = False
        obstacle_side = None
        obstacle_distance = 0

        # Front obstacle (highest priority)
        if front_max > self.OBSTACLE_THRESHOLD:
            has_obstacle = True
            obstacle_distance = front_max
            # Determine which side to avoid
            if front_left > front_right:
                obstacle_side = 'front_left'
            else:
                obstacle_side = 'front_right'
        # Left side obstacle
        elif left_side > self.OBSTACLE_THRESHOLD:
            has_obstacle = True
            obstacle_side = 'left'
            obstacle_distance = left_side
        # Right side obstacle
        elif right_side > self.OBSTACLE_THRESHOLD:
            has_obstacle = True
            obstacle_side = 'right'
            obstacle_distance = right_side

        return has_obstacle, obstacle_side, obstacle_distance

    # ========== NEW: Line Detection Function ==========
    def is_on_line(self):
        """
        Check if robot is on the black line

        Returns:
            Boolean: True if any ground sensor detects the line
        """
        # If any ground sensor value is low (dark), we're on the line
        line_threshold = 0.5
        return (self.inputs[0] < line_threshold or
                self.inputs[1] < line_threshold or
                self.inputs[2] < line_threshold)

    # ========== NEW: Obstacle Avoidance State Machine ==========
    def obstacle_avoidance_behavior(self):
        """
        Main obstacle avoidance state machine

        States:
        1. LINE_FOLLOWING: Normal line following
        2. OBSTACLE_DETECTED: Obstacle detected, decide avoidance strategy
        3. AVOIDING_LEFT: Turn left to avoid obstacle
        4. AVOIDING_RIGHT: Turn right to avoid obstacle
        5. SEARCHING_LINE: Search for the line after avoiding obstacle

        Returns:
            tuple: (left_velocity, right_velocity)
        """
        # Detect obstacles
        has_obstacle, obstacle_side, obstacle_distance = self.detect_obstacle()

        # State machine
        if self.state == 'LINE_FOLLOWING':
            # ===== State 1: Normal Line Following =====
            if has_obstacle:
                # Obstacle detected! Switch to avoidance mode
                self.state = 'OBSTACLE_DETECTED'
                self.obstacle_side = obstacle_side
                self.avoidance_counter = 0
                print(f"[{self.state}] Obstacle detected on {obstacle_side}!")
                return 0, 0  # Stop briefly
            else:
                # Continue line following (use original logic)
                return self.line_following_behavior()

        elif self.state == 'OBSTACLE_DETECTED':
            # ===== State 2: Decide Avoidance Strategy =====
            # Decide which direction to avoid based on obstacle position
            if obstacle_side in ['front_left', 'left']:
                # Obstacle on left, avoid by turning right
                self.state = 'AVOIDING_RIGHT'
                print(f"[{self.state}] Avoiding right")
            else:
                # Obstacle on right or front-right, avoid by turning left
                self.state = 'AVOIDING_LEFT'
                print(f"[{self.state}] Avoiding left")

            self.avoidance_counter = 0
            return 0, 0

        elif self.state == 'AVOIDING_LEFT':
            # ===== State 3: Turn Left to Avoid Obstacle =====
            self.avoidance_counter += 1

            # Phase 1: Turn left sharply (first 1/3 of duration)
            if self.avoidance_counter < self.AVOIDANCE_DURATION // 3:
                left_vel = 0.2
                right_vel = 0.8
            # Phase 2: Move forward while slightly left (middle 1/3)
            elif self.avoidance_counter < 2 * self.AVOIDANCE_DURATION // 3:
                left_vel = 0.6
                right_vel = 0.8
            # Phase 3: Turn right to get back parallel (last 1/3)
            else:
                left_vel = 0.8
                right_vel = 0.4

            # Check if we've completed avoidance
            if self.avoidance_counter >= self.AVOIDANCE_DURATION:
                self.state = 'SEARCHING_LINE'
                self.search_counter = 0
                print(f"[{self.state}] Searching for line")

            # Emergency: if still detecting obstacle, keep avoiding
            if has_obstacle and obstacle_distance > self.OBSTACLE_CLOSE_THRESHOLD:
                self.avoidance_counter = 0  # Reset counter

            return left_vel, right_vel

        elif self.state == 'AVOIDING_RIGHT':
            # ===== State 4: Turn Right to Avoid Obstacle =====
            self.avoidance_counter += 1

            # Phase 1: Turn right sharply (first 1/3 of duration)
            if self.avoidance_counter < self.AVOIDANCE_DURATION // 3:
                left_vel = 0.8
                right_vel = 0.2
            # Phase 2: Move forward while slightly right (middle 1/3)
            elif self.avoidance_counter < 2 * self.AVOIDANCE_DURATION // 3:
                left_vel = 0.8
                right_vel = 0.6
            # Phase 3: Turn left to get back parallel (last 1/3)
            else:
                left_vel = 0.4
                right_vel = 0.8

            # Check if we've completed avoidance
            if self.avoidance_counter >= self.AVOIDANCE_DURATION:
                self.state = 'SEARCHING_LINE'
                self.search_counter = 0
                print(f"[{self.state}] Searching for line")

            # Emergency: if still detecting obstacle, keep avoiding
            if has_obstacle and obstacle_distance > self.OBSTACLE_CLOSE_THRESHOLD:
                self.avoidance_counter = 0  # Reset counter

            return left_vel, right_vel

        elif self.state == 'SEARCHING_LINE':
            # ===== State 5: Search for Line After Avoidance =====
            self.search_counter += 1

            # Check if we found the line
            if self.is_on_line():
                # Found the line! Return to line following
                self.state = 'LINE_FOLLOWING'
                print(f"[{self.state}] Line found! Resuming line following")
                return self.line_following_behavior()

            # If not found, continue searching
            # Strategy: Move forward while slightly turning towards expected line position
            if self.obstacle_side in ['front_left', 'left']:
                # We avoided left, so line should be on our right
                # Turn slightly right while moving forward
                left_vel = 0.7
                right_vel = 0.5
            else:
                # We avoided right, so line should be on our left
                # Turn slightly left while moving forward
                left_vel = 0.5
                right_vel = 0.7

            # Timeout: if we can't find line after SEARCH_DURATION steps
            if self.search_counter >= self.SEARCH_DURATION:
                # Give up searching, return to line following mode
                # (will use default behavior)
                self.state = 'LINE_FOLLOWING'
                print(f"[{self.state}] Search timeout, resuming line following")

            return left_vel, right_vel

        # Default: return to line following
        return self.line_following_behavior()

    # ========== NEW: Extracted Line Following Behavior ==========
    def line_following_behavior(self):
        """
        Original line following logic extracted as a separate function

        Returns:
            tuple: (left_velocity, right_velocity)
        """
        # Turn at end of line
        if self.flag_turn:
            left_vel = -0.3
            right_vel = 0.3
            if np.min(self.inputs[0:3]) < 0.35:
                self.flag_turn = 0
            return left_vel, right_vel

        # Check end of line
        if (np.min(self.inputs[0:3]) - np.min(self.inputsPrevious[0:3])) > 0.2:
            self.flag_turn = 1
            return -0.3, 0.3

        # Follow the line based on ground sensors
        if self.inputs[0] < self.inputs[1] and self.inputs[0] < self.inputs[2]:
            # Line is on the left, turn left
            left_vel = 0.5
            right_vel = 1.0
        elif self.inputs[1] < self.inputs[0] and self.inputs[1] < self.inputs[2]:
            # Line is in the center, go straight
            left_vel = 1.0
            right_vel = 1.0
        elif self.inputs[2] < self.inputs[0] and self.inputs[2] < self.inputs[1]:
            # Line is on the right, turn right
            left_vel = 1.0
            right_vel = 0.5
        else:
            # Default: go straight
            left_vel = 0.8
            right_vel = 0.8

        return left_vel, right_vel

    def sense_compute_and_actuate(self):
        """
        Main control function - now uses obstacle avoidance state machine
        """
        if len(self.inputs) > 0 and len(self.inputsPrevious) > 0:
            # ========== NEW: Use obstacle avoidance behavior ==========
            self.velocity_left, self.velocity_right = self.obstacle_avoidance_behavior()

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

    def run_robot(self):
        # Main Loop
        count = 0;
        inputs_avg = []
        while self.robot.step(self.time_step) != -1:
            # Read Ground Sensors
            self.inputs = []
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()

            # Adjust Values
            min_gs = 0
            max_gs = 1000
            if (left > max_gs): left = max_gs
            if (center > max_gs): center = max_gs
            if (right > max_gs): right = max_gs
            if (left < min_gs): left = min_gs
            if (center < min_gs): center = min_gs
            if (right < min_gs): right = min_gs

            # Save Data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                if (i == 0 or i == 1 or i == 2 or i == 5 or i == 6 or i == 7):
                    temp = self.proximity_sensors[i].getValue()
                    # Adjust Values
                    min_ds = 0
                    max_ds = 2400
                    if (temp > max_ds): temp = max_ds
                    if (temp < min_ds): temp = min_ds
                    # Save Data
                    self.inputs.append((temp - min_ds) / (max_ds - min_ds))
                    # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            # Smooth filter (Average)
            smooth = 30
            if (count == smooth):
                inputs_avg = [sum(x) for x in zip(*inputs_avg)]
                self.inputs = [x / smooth for x in inputs_avg]
                # Compute and actuate
                self.sense_compute_and_actuate()
                # Reset
                count = 0
                inputs_avg = []
                self.inputsPrevious = self.inputs
            else:
                inputs_avg.append(self.inputs)
                count = count + 1


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()