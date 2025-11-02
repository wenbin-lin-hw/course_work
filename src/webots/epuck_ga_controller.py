"""
E-puck GA Controller for Race Circuit Navigation
This controller uses evolved neural network weights to navigate a race circuit
by following a line and avoiding obstacles intelligently.
"""

from controller import Robot, Emitter, Receiver
import numpy as np
import struct


class NeuralNetwork:
    """Simple feedforward neural network for robot control"""

    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Calculate total number of weights
        self.num_weights = (num_inputs * num_hidden) + num_hidden + \
                           (num_hidden * num_outputs) + num_outputs

        # Initialize weights
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

    def set_weights(self, weights):
        """Set network weights from a flat array (genotype)"""
        if len(weights) != self.num_weights:
            raise ValueError(f"Expected {self.num_weights} weights, got {len(weights)}")

        idx = 0

        # Weights from input to hidden layer
        size = self.num_inputs * self.num_hidden
        self.weights_input_hidden = np.array(weights[idx:idx + size]).reshape(
            self.num_inputs, self.num_hidden)
        idx += size

        # Bias for hidden layer
        self.bias_hidden = np.array(weights[idx:idx + self.num_hidden])
        idx += self.num_hidden

        # Weights from hidden to output layer
        size = self.num_hidden * self.num_outputs
        self.weights_hidden_output = np.array(weights[idx:idx + size]).reshape(
            self.num_hidden, self.num_outputs)
        idx += size

        # Bias for output layer
        self.bias_output = np.array(weights[idx:idx + self.num_outputs])

    def activate(self, inputs):
        """Forward pass through the network"""
        # Hidden layer with tanh activation
        hidden = np.tanh(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)

        # Output layer with tanh activation
        output = np.tanh(np.dot(hidden, self.weights_hidden_output) + self.bias_output)

        return output


class EPuckGAController:
    def __init__(self):
        # Initialize robot
        self.robot = Robot()
        self.time_step = 32  # ms
        self.max_speed = 6.28  # rad/s

        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Initialize proximity sensors (8 sensors)
        self.proximity_sensors = []
        for i in range(8):
            sensor = self.robot.getDevice(f'ps{i}')
            sensor.enable(self.time_step)
            self.proximity_sensors.append(sensor)

        # Initialize ground sensors (3 sensors for line following)
        self.ground_sensors = []
        for i in range(3):
            sensor = self.robot.getDevice(f'gs{i}')
            sensor.enable(self.time_step)
            self.ground_sensors.append(sensor)

        # Communication with supervisor
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)

        # Neural network configuration
        # Inputs: 3 ground sensors + 8 proximity sensors = 11 inputs
        # Hidden layer: 8 neurons
        # Outputs: 2 (left and right motor speeds)
        self.num_inputs = 11
        self.num_hidden = 8
        self.num_outputs = 2

        self.neural_network = NeuralNetwork(
            self.num_inputs,
            self.num_hidden,
            self.num_outputs
        )

        # Fitness tracking
        self.fitness = 0.0
        self.start_position = None
        self.max_distance = 0.0
        self.time_on_line = 0
        self.time_steps = 0
        self.collision_penalty = 0
        self.speed_reward = 0.0

        # Previous sensor values for derivative calculation
        self.prev_ground_sensors = [0, 0, 0]

        # Send number of weights to supervisor
        self.send_weights_size()

    def send_weights_size(self):
        """Send the number of weights to the supervisor"""
        message = f"weights: {self.neural_network.num_weights}"
        self.emitter.send(message.encode('utf-8'))

    def receive_genotype(self):
        """Receive genotype from supervisor"""
        if self.receiver.getQueueLength() > 0:
            message = self.receiver.getString()
            self.receiver.nextPacket()

            # Parse the genotype
            try:
                # Remove brackets and split
                genotype_str = message.strip('[]')
                weights = [float(x) for x in genotype_str.split()]
                self.neural_network.set_weights(weights)
                return True
            except Exception as e:
                print(f"Error parsing genotype: {e}")
                return False
        return False

    def send_fitness(self):
        """Send fitness value to supervisor"""
        message = f"fitness: {self.fitness}"
        self.emitter.send(message.encode('utf-8'))

    def get_sensor_values(self):
        """Read and normalize all sensor values"""
        inputs = []

        # Ground sensors (normalized to 0-1, where 0 is dark/line, 1 is bright)
        for sensor in self.ground_sensors:
            value = sensor.getValue()
            # Normalize: typical range is 0-1000
            normalized = np.clip(value / 1000.0, 0.0, 1.0)
            inputs.append(normalized)

        # Proximity sensors (normalized to 0-1, where 1 is close obstacle)
        for sensor in self.proximity_sensors:
            value = sensor.getValue()
            # Normalize: typical range is 0-4096
            normalized = np.clip(value / 4096.0, 0.0, 1.0)
            inputs.append(normalized)

        return np.array(inputs)

    def calculate_fitness(self):
        """
        Calculate fitness based on:
        1. Distance traveled (primary objective - speed)
        2. Time spent on the line (line following accuracy)
        3. Collision avoidance (penalty for hitting obstacles)
        4. Average speed (reward for fast movement)
        """

        # Get current position
        node = self.robot.getSelf()
        if node is not None:
            position = node.getPosition()

            if self.start_position is None:
                self.start_position = position

            # Calculate distance from start
            distance = np.sqrt(
                (position[0] - self.start_position[0]) ** 2 +
                (position[2] - self.start_position[2]) ** 2
            )

            # Update max distance
            if distance > self.max_distance:
                self.max_distance = distance

        # Check if robot is on the line (ground sensors detect dark)
        ground_values = [s.getValue() for s in self.ground_sensors]
        on_line = any(v < 500 for v in ground_values)  # Dark threshold

        if on_line:
            self.time_on_line += 1

        # Check for collisions (high proximity sensor values)
        proximity_values = [s.getValue() for s in self.proximity_sensors]
        if max(proximity_values) > 100:  # Collision threshold
            self.collision_penalty += 0.1

        # Calculate average speed
        current_speed = (abs(self.left_motor.getVelocity()) +
                         abs(self.right_motor.getVelocity())) / 2.0
        self.speed_reward += current_speed / self.max_speed

        self.time_steps += 1

        # Fitness function:
        # - Maximize distance traveled (most important for race)
        # - Reward staying on line
        # - Penalize collisions
        # - Reward high average speed

        line_following_bonus = (self.time_on_line / max(self.time_steps, 1)) * 10.0
        speed_bonus = (self.speed_reward / max(self.time_steps, 1)) * 5.0

        self.fitness = (
                self.max_distance * 10.0 +  # Distance is most important
                line_following_bonus +  # Bonus for staying on line
                speed_bonus -  # Bonus for speed
                self.collision_penalty * 5.0  # Penalty for collisions
        )

        return self.fitness

    def run(self):
        """Main control loop"""
        print("E-puck GA Controller started")
        print(f"Waiting for genotype ({self.neural_network.num_weights} weights)...")

        # Wait for genotype from supervisor
        genotype_received = False
        while self.robot.step(self.time_step) != -1:
            if not genotype_received:
                if self.receive_genotype():
                    print("Genotype received, starting evaluation...")
                    genotype_received = True
                    break

        # Main control loop
        while self.robot.step(self.time_step) != -1:
            # Get sensor inputs
            sensor_inputs = self.get_sensor_values()

            # Get motor commands from neural network
            outputs = self.neural_network.activate(sensor_inputs)

            # Map neural network outputs to motor speeds
            # outputs are in range [-1, 1], map to [0, max_speed]
            left_speed = (outputs[0] + 1.0) / 2.0 * self.max_speed
            right_speed = (outputs[1] + 1.0) / 2.0 * self.max_speed

            # Set motor speeds
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)

            # Calculate fitness
            self.calculate_fitness()

        # Send final fitness to supervisor
        self.send_fitness()
        print(f"Evaluation complete. Fitness: {self.fitness:.2f}")


if __name__ == "__main__":
    controller = EPuckGAController()
    controller.run()