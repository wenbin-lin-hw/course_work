"""e-puck_behavior_tree.py

   Controller using behavior tree for more complex decision making
"""

from controller import Robot
import sys


class BehaviorNode:
    """Base class for behavior tree nodes"""
    SUCCESS = 'success'
    FAILURE = 'failure'
    RUNNING = 'running'

    def tick(self):
        raise NotImplementedError


class Sequence(BehaviorNode):
    """Executes children in sequence until one fails"""

    def __init__(self, children):
        self.children = children
        self.current = 0

    def tick(self):
        while self.current < len(self.children):
            status = self.children[self.current].tick()
            if status != self.SUCCESS:
                return status
            self.current += 1
        self.current = 0
        return self.SUCCESS


class Fallback(BehaviorNode):
    """Tries children until one succeeds"""

    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status != self.FAILURE:
                return status
        return self.FAILURE


class ObstacleAvoidance(BehaviorNode):
    """Obstacle avoidance action"""

    def __init__(self, robot, ps, motors):
        self.robot = robot
        self.ps = ps
        self.left_motor, self.right_motor = motors
        self.threshold = 80.0

    def tick(self):
        # Check for obstacles
        front_left = self.ps[7].getValue()
        front_right = self.ps[0].getValue()

        if front_right > self.threshold:
            # Turn left
            self.left_motor.setVelocity(-3.0)
            self.right_motor.setVelocity(3.0)
            return self.RUNNING
        elif front_left > self.threshold:
            # Turn right
            self.left_motor.setVelocity(3.0)
            self.right_motor.setVelocity(-3.0)
            return self.RUNNING

        return self.FAILURE  # No obstacle


class FloorFollowing(BehaviorNode):
    """Floor sensor following action"""

    def __init__(self, robot, gs, motors):
        self.robot = robot
        self.gs = gs
        self.left_motor, self.right_motor = motors

    def tick(self):
        gs_values = [sensor.getValue() for sensor in self.gs]

        # Implement floor following logic
        if gs_values[0] < 500:  # Left sensor
            self.left_motor.setVelocity(4.5)
            self.right_motor.setVelocity(1.5)
        elif gs_values[2] < 500:  # Right sensor
            self.left_motor.setVelocity(1.5)
            self.right_motor.setVelocity(4.5)
        else:
            self.left_motor.setVelocity(6.28)
            self.right_motor.setVelocity(6.28)

        return self.SUCCESS


# Initialize robot (similar to previous example)
robot = Robot()
TIME_STEP = int(robot.getBasicTimeStep())

# Initialize sensors and motors...
# (same initialization code as before)

# Create behavior tree
root = Fallback([
    ObstacleAvoidance(robot, ps, (left_motor, right_motor)),
    FloorFollowing(robot, gs, (left_motor, right_motor))
])

# Main loop
while robot.step(TIME_STEP) != -1:
    root.tick()