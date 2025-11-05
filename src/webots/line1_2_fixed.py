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
        # 'AVOIDING_RIGHT', 'SEARCHING_LINE'
        self.state = 'LINE_FOLLOWING'
        self.obstacle_side = None  # 'left' or 'right'
        self.avoidance_counter = 0  # Counter for avoidance maneuvers
        self.search_counter = 0  # Counter for searching the line

        # Obstacle detection thresholds - ADJUSTED FOR SMALL ARENA (2m x 1.8m)
        self.OBSTACLE_THRESHOLD = 0.05  # Detect obstacle at ~4-5cm (more sensitive)
        self.OBSTACLE_CLOSE_THRESHOLD = 0.20  # Emergency threshold at ~1.5-2cm

        # Avoidance parameters - ADJUSTED FOR SMALL ARENA
        self.AVOIDANCE_DURATION = 30  # Reduced: steps to move around obstacle
        self.SEARCH_DURATION = 60  # Reduced: steps to search for line

        print("Controller initialized with obstacle avoidance capability (Small Arena Mode)")

    def clip_value(self, value, min_max):
        if (value > min_max):
            return min_max
        elif (value < -min_max):
            return -min_max
        return value

    # ========== NEW: Obstacle Detection Function ==========
    def detect_obstacle(self):
        """
        Detect obstacles using proximity sensors
        Returns: tuple: (has_obstacle, obstacle_side, obstacle_distance)
        """
        # Check if we have enough sensor data
        if len(self.inputs) < 9:
            return False, None, 0

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
        """
        # Check if we have ground sensor data
        if len(self.inputs) < 3:
            return False

        # If any ground sensor value is low (dark), we're on the line
        line_threshold = 0.5
        return (self.inputs[0] < line_threshold or
                self.inputs[1] < line_threshold or
                self.inputs[2] < line_threshold)

    def sense_compute_and_actuate(self):
        if (len(self.inputs) > 0 and len(self.inputsPrevious) > 0):
            # ========== NEW: Check for obstacle first ==========
            has_obstacle, obstacle_side, obstacle_distance = self.detect_obstacle()

            # ========== State Machine Logic ==========
            if self.state == 'LINE_FOLLOWING':
                # Check for obstacle
                if has_obstacle:
                    # Obstacle detected! Switch to avoidance mode
                    self.state = 'OBSTACLE_DETECTED'
                    self.obstacle_side = obstacle_side
                    self.avoidance_counter = 0
                    time = datetime.now()
                    print("({} - {}) Obstacle detected on {}!".format(
                        time.second, time.microsecond, obstacle_side))
                    self.velocity_left = 0
                    self.velocity_right = 0
                else:
                    # ========== Original Line Following Logic (UNCHANGED) ==========
                    # Check for any possible collision (original code)
                    if (np.max(self.inputs[3:9]) > 0.4):
                        time = datetime.now()
                        print("({} - {}) Object or walls detected!".format(time.second, time.microsecond))

                    # Turn
                    if (self.flag_turn):
                        self.velocity_left = -0.3
                        self.velocity_right = 0.3
                        if (np.min(self.inputs[0:3]) < 0.35):
                            self.flag_turn = 0
                    else:
                        # Check end of line
                        if ((np.min(self.inputs[0:3]) - np.min(self.inputsPrevious[0:3])) > 0.2):
                            self.flag_turn = 1
                        else:
                            # Follow the line
                            if (self.inputs[0] < self.inputs[1] and self.inputs[0] < self.inputs[2]):
                                self.velocity_left = 0.5
                                self.velocity_right = 1
                            elif (self.inputs[1] < self.inputs[0] and self.inputs[1] < self.inputs[2]):
                                self.velocity_left = 1
                                self.velocity_right = 1
                            elif (self.inputs[2] < self.inputs[0] and self.inputs[2] < self.inputs[1]):
                                self.velocity_left = 1
                                self.velocity_right = 0.5

            elif self.state == 'OBSTACLE_DETECTED':
                # Decide which direction to avoid
                if obstacle_side in ['front_left', 'left']:
                    self.state = 'AVOIDING_RIGHT'
                    print("Starting right avoidance maneuver")
                else:
                    self.state = 'AVOIDING_LEFT'
                    print("Starting left avoidance maneuver")

                self.avoidance_counter = 0
                self.velocity_left = 0
                self.velocity_right = 0

            elif self.state == 'AVOIDING_LEFT':
                # Turn Left to Avoid Obstacle
                self.avoidance_counter += 1
                # Phase 1: Turn left sharply
                if self.avoidance_counter < self.AVOIDANCE_DURATION // 3:
                    self.velocity_left = 0.2
                    self.velocity_right = 0.8
                # Phase 2: Move forward while slightly left
                elif self.avoidance_counter < 2 * self.AVOIDANCE_DURATION // 3:
                    self.velocity_left = 0.6
                    self.velocity_right = 0.8
                # Phase 3: Turn right to get back parallel
                else:
                    self.velocity_left = 0.8
                    self.velocity_right = 0.4

                # Check if avoidance complete
                if self.avoidance_counter >= self.AVOIDANCE_DURATION:
                    self.state = 'SEARCHING_LINE'
                    self.search_counter = 0
                    print("Searching for line")
                # Emergency: if still detecting obstacle, keep avoiding
                if has_obstacle and obstacle_distance > self.OBSTACLE_CLOSE_THRESHOLD:
                    self.avoidance_counter = 0
            elif self.state == 'AVOIDING_RIGHT':
                # Turn Right to Avoid Obstacle
                self.avoidance_counter += 1

                # Phase 1: Turn right sharply
                if self.avoidance_counter < self.AVOIDANCE_DURATION // 3:
                    self.velocity_left = 0.8
                    self.velocity_right = 0.2
                # Phase 2: Move forward while slightly right
                elif self.avoidance_counter < 2 * self.AVOIDANCE_DURATION // 3:
                    self.velocity_left = 0.8
                    self.velocity_right = 0.6
                # Phase 3: Turn left to get back parallel
                else:
                    self.velocity_left = 0.4
                    self.velocity_right = 0.8

                # Check if avoidance complete
                if self.avoidance_counter >= self.AVOIDANCE_DURATION:
                    self.state = 'SEARCHING_LINE'
                    self.search_counter = 0
                    print("Searching for line")

                # Emergency: if still detecting obstacle, keep avoiding
                if has_obstacle and obstacle_distance > self.OBSTACLE_CLOSE_THRESHOLD:
                    self.avoidance_counter = 0

            elif self.state == 'SEARCHING_LINE':
                # Search for Line After Avoidance
                self.search_counter += 1

                # Check if we found the line
                if self.is_on_line():
                    self.state = 'LINE_FOLLOWING'
                    print("Line found! Resuming line following")
                    # Reset to line following behavior
                    if (self.inputs[0] < self.inputs[1] and self.inputs[0] < self.inputs[2]):
                        self.velocity_left = 0.5
                        self.velocity_right = 1
                    elif (self.inputs[1] < self.inputs[0] and self.inputs[1] < self.inputs[2]):
                        self.velocity_left = 1
                        self.velocity_right = 1
                    elif (self.inputs[2] < self.inputs[0] and self.inputs[2] < self.inputs[1]):
                        self.velocity_left = 1
                        self.velocity_right = 0.5
                else:
                    # Continue searching
                    if self.obstacle_side in ['front_left', 'left']:
                        # Turn slightly right while moving forward
                        self.velocity_left = 0.7
                        self.velocity_right = 0.5
                    else:
                        # Turn slightly left while moving forward
                        self.velocity_left = 0.5
                        self.velocity_right = 0.7

                    # Timeout: if can't find line
                    if self.search_counter >= self.SEARCH_DURATION:
                        self.state = 'LINE_FOLLOWING'
                        print("Search timeout, resuming line following")

            # Set motor velocities
            self.left_motor.setVelocity(self.velocity_left)
            self.right_motor.setVelocity(self.velocity_right)

    def run_robot(self):
        # Main Loop
        count = 0
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