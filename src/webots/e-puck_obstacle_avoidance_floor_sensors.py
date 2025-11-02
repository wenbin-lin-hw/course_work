"""e-puck_obstacle_avoidance_floor_sensors.py
   Controller that combines floor sensor following with obstacle avoidance
   using a subsumption-like architecture.
"""

from controller import Robot, DistanceSensor, Motor
import sys

# Create robot instance
robot = Robot()

# Get simulation timestep
TIME_STEP = int(robot.getBasicTimeStep())

# Constants
MAX_SPEED = 6.28  # Maximum speed in rad/s
OBSTACLE_THRESHOLD = 80.0  # Threshold for obstacle detection
FLOOR_THRESHOLD = 500  # Threshold for floor sensors

# Initialize distance sensors (obstacle avoidance)
ps = []
ps_names = ['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7']

for name in ps_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    ps.append(sensor)

# Initialize ground sensors (floor following)
gs = []
gs_names = ['gs0', 'gs1', 'gs2']

for name in gs_names:
    sensor = robot.getDevice(name)
    sensor.enable(TIME_STEP)
    gs.append(sensor)

# Initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


def detect_obstacle():
    """
    Check if obstacle is detected using proximity sensors.
    Returns: (obstacle_detected, turn_direction)
    """
    # Read front and side sensors
    front_left = ps[7].getValue()
    front_right = ps[0].getValue()
    side_left = ps[6].getValue()
    side_right = ps[1].getValue()

    # Check for obstacles
    if front_right > OBSTACLE_THRESHOLD or side_right > OBSTACLE_THRESHOLD:
        return (True, 'left')  # Obstacle on right, turn left
    elif front_left > OBSTACLE_THRESHOLD or side_left > OBSTACLE_THRESHOLD:
        return (True, 'right')  # Obstacle on left, turn right
    elif front_left > OBSTACLE_THRESHOLD and front_right > OBSTACLE_THRESHOLD:
        return (True, 'left')  # Obstacle in front, turn left

    return (False, None)


def obstacle_avoidance_behavior():
    """
    Higher priority behavior: Obstacle avoidance
    Returns: (left_speed, right_speed) or None if not active
    """
    obstacle_detected, direction = detect_obstacle()

    if obstacle_detected:
        if direction == 'left':
            # Turn left
            return (-MAX_SPEED * 0.5, MAX_SPEED * 0.5)
        elif direction == 'right':
            # Turn right
            return (MAX_SPEED * 0.5, -MAX_SPEED * 0.5)

    return None  # Behavior not active


def floor_following_behavior():
    """
    Lower priority behavior: Follow floor sensors (line following)
    Returns: (left_speed, right_speed)
    """
    # Read ground sensors
    gs_values = [sensor.getValue() for sensor in gs]

    # Simple line following logic
    # gs0: left, gs1: center, gs2: right

    left_speed = MAX_SPEED
    right_speed = MAX_SPEED

    # If left sensor detects line (dark), turn right
    if gs_values[0] < FLOOR_THRESHOLD:
        left_speed = MAX_SPEED * 0.75
        right_speed = MAX_SPEED * 0.25
    # If right sensor detects line (dark), turn left
    elif gs_values[2] < FLOOR_THRESHOLD:
        left_speed = MAX_SPEED * 0.25
        right_speed = MAX_SPEED * 0.75
    # If center sensor detects line, go straight
    elif gs_values[1] < FLOOR_THRESHOLD:
        left_speed = MAX_SPEED
        right_speed = MAX_SPEED
    # If no line detected, search for it
    else:
        left_speed = MAX_SPEED * 0.5
        right_speed = -MAX_SPEED * 0.5

    return (left_speed, right_speed)


def default_behavior():
    """
    Lowest priority behavior: Move forward
    Returns: (left_speed, right_speed)
    """
    return (MAX_SPEED, MAX_SPEED)


# Main control loop
print("E-puck with layered behavior architecture")
print("- Layer 1 (Highest): Obstacle Avoidance")
print("- Layer 2: Floor Following")
print("- Layer 3 (Lowest): Forward Motion")

while robot.step(TIME_STEP) != -1:
    # Subsumption architecture: Higher priority behaviors subsume lower ones
    # Try obstacle avoidance (highest priority)
    speeds = obstacle_avoidance_behavior()
    if speeds is None:
        # Try floor following (medium priority)
        speeds = floor_following_behavior()

    if speeds is None:
        # Default behavior (lowest priority)
        speeds = default_behavior()

    # Apply motor speeds
    left_motor.setVelocity(speeds[0])
    right_motor.setVelocity(speeds[1])