from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw

class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 1  # m/s
 
        # MLP Parameters and Variables   
        ### Define below the architecture of your MLP network. 
        ### Add the number of neurons for each layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.
        self.number_input_layer = 11 #8 proximity + 3 ground sensors
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        self.number_hidden_layer = [8,6]
        self.number_output_layer = 2 
        
        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        
        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []
        
        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1,len(self.number_neuros_per_layer)):
            if(n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n-1]+1)*self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n]

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
        
        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        self.receivedData = "" 
        self.receivedDataPrevious = "" 
        self.flagMessage = False
        # Time tracking
        self.step_count = 0
        self.time_on_line = 0
        self.time_off_line = 0
        self.total_distance = 0
        self.distance_on_line = 0
        # Speed tracking
        self.speed_history = []
        self.avg_speed_on_line = 0
        # Collision tracking
        self.collision_count = 0
        self.near_collision_count = 0
        self.time_near_obstacle = 0
        # Line following tracking
        self.consecutive_on_line = 0
        self.consecutive_off_line = 0
        
        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0

    # def forwardFitness(self):
    #     """
    #     Fitness function for forward movement behavior
    #     Objective: Move as fast as possible in forward direction
    #     Rewards: High speed, straight movement, continuous forward motion
    #     Penalties: Stopping, turning, backward movement, collisions
    #     """
    #
    #     # 1. SPEED COMPONENT - Reward high forward speed
    #     avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)
    #     speed_fitness = avg_speed * 100  # 0-100 points
    #
    #     # 2. FORWARD DIRECTION - Both wheels must move forward
    #     if self.velocity_left > 0 and self.velocity_right > 0:
    #         forward_bonus = 50
    #         # Extra bonus for high speed forward
    #         if avg_speed > 0.8:
    #             forward_bonus += 25
    #     elif self.velocity_left < 0 or self.velocity_right < 0:
    #         forward_bonus = -75  # Heavy penalty for backward
    #     else:
    #         forward_bonus = -25  # Penalty for stopping
    #
    #     # 3. STRAIGHT MOVEMENT - Minimize turning
    #     speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
    #     straight_bonus = (1 - speed_diff) * 50  # 0-50 points
    #
    #     # 4. COLLISION AVOIDANCE - Penalty for obstacles
    #     front_sensors = [
    #         self.proximity_sensors[0].getValue(),  # Front-right
    #         self.proximity_sensors[1].getValue(),  # Front-right-side
    #         self.proximity_sensors[6].getValue(),  # Front-left-side
    #         self.proximity_sensors[7].getValue()  # Front-left
    #     ]
    #     max_front_proximity = max(front_sensors)
    #
    #     if max_front_proximity > 3000:  # Very close to obstacle
    #         collision_penalty = 100
    #     elif max_front_proximity > 2000:
    #         collision_penalty = 50
    #     else:
    #         collision_penalty = (max_front_proximity / 4096) * 30
    #
    #     # 5. ACTIVITY PENALTY - Discourage inactivity
    #     if avg_speed < 0.05:
    #         activity_penalty = 50
    #     else:
    #         activity_penalty = 0
    #
    #     # Calculate total fitness
    #     fitness = (
    #             speed_fitness +
    #             forward_bonus +
    #             straight_bonus -
    #             collision_penalty -
    #             activity_penalty
    #     )
    #
    #     fitness = max(0, fitness)
    #
    #     return fitness
    #
    #
    # def followLineFitness(self):
    #     """
    #     Fitness function for line following behavior
    #     Objective: Stay on the black track line and follow it smoothly
    #     Rewards: Staying on line, moderate speed, smooth following
    #     Penalties: Going off track, stopping, erratic movement
    #     """
    #
    #     # 1. LINE DETECTION - Read ground sensors
    #     left_ground = self.left_ir.getValue()
    #     center_ground = self.center_ir.getValue()
    #     right_ground = self.right_ir.getValue()
    #
    #     # Normalize (0 = black/line, 1000 = white/off-line)
    #     # Assuming sensor range 0-1000
    #
    #     # 2. ON-LINE REWARD - Center sensor should detect line
    #     if center_ground < 400:  # On black line
    #         center_on_line = 100
    #     elif center_ground < 600:  # Partially on line
    #         center_on_line = 50
    #     else:  # Off line
    #         center_on_line = 0
    #
    #     # 3. CENTERING BONUS - Robot centered on line
    #     # Best case: center is dark, sides are light (or all dark for wide lines)
    #     if center_ground < 400:
    #         if left_ground < 500 and right_ground < 500:
    #             # All sensors on line (good for wide lines)
    #             centering_bonus = 50
    #         elif left_ground > 600 and right_ground > 600:
    #             # Only center on line (perfectly centered on narrow line)
    #             centering_bonus = 75
    #         else:
    #             # Partially centered
    #             centering_bonus = 25
    #     else:
    #         centering_bonus = 0
    #
    #     # 4. LINE TRACKING QUALITY - Penalize being off track
    #     ground_avg = (left_ground + center_ground + right_ground) / 3
    #     if ground_avg < 500:  # Mostly on line
    #         tracking_quality = 50
    #     elif ground_avg < 700:  # Partially on line
    #         tracking_quality = 25
    #     else:  # Completely off line
    #         tracking_quality = -50
    #
    #     # 5. SPEED COMPONENT - Moderate speed is good for line following
    #     avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)
    #
    #     # Optimal speed for line following: 0.4-0.7 of max_speed
    #     if 0.4 <= avg_speed <= 0.7:
    #         speed_fitness = 50
    #     elif 0.2 <= avg_speed < 0.4:
    #         speed_fitness = 30
    #     elif avg_speed > 0.7:
    #         speed_fitness = 20  # Too fast may lose line
    #     else:
    #         speed_fitness = -25  # Too slow or stopped
    #
    #     # 6. SMOOTH MOVEMENT - Reward smooth steering
    #     speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
    #
    #     # Some turning is expected for line following
    #     if speed_diff < 0.3:  # Smooth turning
    #         smooth_bonus = 30
    #     elif speed_diff < 0.5:
    #         smooth_bonus = 15
    #     else:  # Too much turning (erratic)
    #         smooth_bonus = -20
    #
    #     # 7. FORWARD DIRECTION - Should move forward
    #     if self.velocity_left > 0 and self.velocity_right > 0:
    #         forward_bonus = 25
    #     else:
    #         forward_bonus = -25
    #
    #     # Calculate total fitness
    #     fitness = (
    #             center_on_line +
    #             centering_bonus +
    #             tracking_quality +
    #             speed_fitness +
    #             smooth_bonus +
    #             forward_bonus
    #     )
    #
    #     fitness = max(0, fitness)
    #
    #     return fitness
    #
    #
    # def avoidCollisionFitness(self):
    #     """
    #     Fitness function for collision avoidance behavior
    #     Objective: Navigate environment while avoiding obstacles
    #     Rewards: Movement with clear space, successful obstacle avoidance
    #     Penalties: Getting close to obstacles, collisions, stopping
    #     """
    #
    #     # 1. READ ALL PROXIMITY SENSORS
    #     proximity_values = [self.proximity_sensors[i].getValue() for i in range(8)]
    #
    #     # Sensor layout (e-puck):
    #     # 0: front-right, 1: right-front, 2: right, 3: right-back
    #     # 4: back, 5: left-back, 6: left, 7: left-front
    #
    #     front_sensors = [proximity_values[0], proximity_values[7]]  # Front
    #     side_sensors = [proximity_values[1], proximity_values[2],
    #                     proximity_values[5], proximity_values[6]]  # Sides
    #     back_sensors = [proximity_values[3], proximity_values[4]]  # Back
    #
    #     max_proximity = max(proximity_values)
    #     avg_proximity = sum(proximity_values) / len(proximity_values)
    #     max_front = max(front_sensors)
    #
    #     # 2. CLEARANCE REWARD - Reward for maintaining distance from obstacles
    #     if max_proximity < 500:  # Very clear space
    #         clearance_reward = 100
    #     elif max_proximity < 1000:  # Good clearance
    #         clearance_reward = 75
    #     elif max_proximity < 2000:  # Moderate clearance
    #         clearance_reward = 50
    #     elif max_proximity < 3000:  # Close to obstacle
    #         clearance_reward = 20
    #     else:  # Very close - danger!
    #         clearance_reward = -50
    #
    #     # 3. COLLISION PENALTY - Heavy penalty for being too close
    #     if max_proximity > 3500:  # Imminent collision
    #         collision_penalty = 150
    #     elif max_proximity > 3000:  # Very close
    #         collision_penalty = 100
    #     elif max_proximity > 2500:  # Close
    #         collision_penalty = 50
    #     else:
    #         collision_penalty = 0
    #
    #     # 4. MOVEMENT REWARD - Should keep moving
    #     avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)
    #
    #     if avg_speed > 0.5:
    #         movement_reward = 75
    #     elif avg_speed > 0.3:
    #         movement_reward = 50
    #     elif avg_speed > 0.1:
    #         movement_reward = 25
    #     else:
    #         movement_reward = -25  # Penalty for stopping
    #
    #     # 5. AVOIDANCE BEHAVIOR - Reward appropriate reactions to obstacles
    #     # If obstacle in front, robot should turn (differential speed)
    #     if max_front > 2000:  # Obstacle detected in front
    #         speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
    #         if speed_diff > 0.3:  # Robot is turning to avoid
    #             avoidance_bonus = 50
    #         else:  # Not turning enough
    #             avoidance_bonus = -25
    #     else:  # No front obstacle
    #         avoidance_bonus = 25  # Bonus for clear navigation
    #
    #     # 6. EXPLORATION BONUS - Reward for moving in open space
    #     if avg_proximity < 1000 and avg_speed > 0.4:
    #         exploration_bonus = 50
    #     else:
    #         exploration_bonus = 0
    #
    #     # 7. FORWARD PREFERENCE - Prefer forward movement
    #     if self.velocity_left > 0 and self.velocity_right > 0:
    #         forward_bonus = 25
    #     else:
    #         forward_bonus = 0
    #
    #     # Calculate total fitness
    #     fitness = (
    #             clearance_reward +
    #             movement_reward +
    #             avoidance_bonus +
    #             exploration_bonus +
    #             forward_bonus -
    #             collision_penalty
    #     )
    #
    #     fitness = max(0, fitness)
    #
    #     return fitness
    #
    #
    # def spinningFitness(self):
    #     """
    #     Fitness function for spinning/rotation behavior
    #     Objective: Rotate in place or perform circular motion
    #     Rewards: Rotational movement, consistent spinning speed
    #     Penalties: Forward movement, stopping, inconsistent rotation
    #     """
    #
    #     # 1. DIFFERENTIAL SPEED - Wheels should move in opposite or highly differential speeds
    #     speed_diff = abs(self.velocity_left - self.velocity_right)
    #
    #     # For spinning in place: wheels should move in opposite directions
    #     opposite_direction = (self.velocity_left * self.velocity_right) < 0
    #
    #     if opposite_direction:
    #         # Perfect spin in place
    #         spin_quality = 100
    #         # Reward higher differential speed
    #         spin_speed_bonus = (speed_diff / (2 * self.max_speed)) * 50
    #     else:
    #         # Circular motion (both forward but different speeds)
    #         if speed_diff > 0.3 * self.max_speed:
    #             spin_quality = 60
    #             spin_speed_bonus = (speed_diff / (2 * self.max_speed)) * 30
    #         else:
    #             spin_quality = 20
    #             spin_speed_bonus = 0
    #
    #     # 2. ROTATION SPEED - Reward active rotation
    #     avg_abs_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
    #
    #     if avg_abs_speed > 0.5 * self.max_speed:
    #         rotation_speed_reward = 50
    #     elif avg_abs_speed > 0.3 * self.max_speed:
    #         rotation_speed_reward = 30
    #     elif avg_abs_speed > 0.1 * self.max_speed:
    #         rotation_speed_reward = 15
    #     else:
    #         rotation_speed_reward = -25  # Penalty for not moving
    #
    #     # 3. CONSISTENCY - Reward consistent spinning behavior
    #     # Track speed history for consistency check
    #     if not hasattr(self, 'spin_history'):
    #         self.spin_history = []
    #
    #     current_spin_rate = self.velocity_left - self.velocity_right
    #     self.spin_history.append(current_spin_rate)
    #
    #     # Keep only recent history
    #     if len(self.spin_history) > 10:
    #         self.spin_history.pop(0)
    #
    #     # Check consistency
    #     if len(self.spin_history) >= 5:
    #         spin_variance = sum([(x - current_spin_rate) ** 2 for x in self.spin_history[-5:]]) / 5
    #         if spin_variance < 0.1:  # Consistent spinning
    #             consistency_bonus = 40
    #         elif spin_variance < 0.3:
    #             consistency_bonus = 20
    #         else:
    #             consistency_bonus = 0
    #     else:
    #         consistency_bonus = 0
    #
    #     # 4. ANTI-FORWARD PENALTY - Penalize moving straight forward
    #     if abs(self.velocity_left - self.velocity_right) < 0.2 * self.max_speed:
    #         # Moving too straight
    #         straight_penalty = 50
    #     else:
    #         straight_penalty = 0
    #
    #     # 5. DIRECTION CONSISTENCY - Reward spinning in same direction
    #     if not hasattr(self, 'spin_direction'):
    #         self.spin_direction = None
    #
    #     current_direction = 1 if (self.velocity_left - self.velocity_right) > 0 else -1
    #
    #     if self.spin_direction is None:
    #         self.spin_direction = current_direction
    #         direction_bonus = 0
    #     elif self.spin_direction == current_direction:
    #         direction_bonus = 30  # Consistent direction
    #     else:
    #         direction_bonus = -20  # Changed direction
    #         self.spin_direction = current_direction
    #
    #     # 6. ACTIVITY REWARD - Must be actively spinning
    #     if avg_abs_speed < 0.05:
    #         activity_penalty = 50
    #     else:
    #         activity_penalty = 0
    #
    #     # Calculate total fitness
    #     fitness = (
    #             spin_quality +
    #             spin_speed_bonus +
    #             rotation_speed_reward +
    #             consistency_bonus +
    #             direction_bonus -
    #             straight_penalty -
    #             activity_penalty
    #     )
    #
    #     fitness = max(0, fitness)
    #
    #     return fitness
    # ============================================================================
    # FITNESS FUNCTION 1: FORWARD FITNESS
    # ============================================================================

    def forwardFitness(self):
        """
        Fitness for forward movement and circuit completion
        Primary Goal: Complete circuit lap as fast as possible

        Rewards:
        - High forward speed
        - Distance traveled
        - Consistent forward motion
        - Progress around circuit

        Penalties:
        - Stopping or slow movement
        - Backward movement
        - Excessive turning (not following circuit)
        """

        fitness = 0

        # 1. SPEED REWARD - Emphasize high speed (most important for lap time)
        avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

        # Reward high speeds more (quadratic reward for speed)
        if avg_speed > 0.8:
            speed_reward = 150 * (avg_speed ** 1.5)  # Extra reward for high speed
        elif avg_speed > 0.6:
            speed_reward = 100 * avg_speed
        elif avg_speed > 0.3:
            speed_reward = 50 * avg_speed
        elif avg_speed > 0.1:
            speed_reward = 20 * avg_speed
        else:
            speed_reward = -30  # Penalty for being too slow

        fitness += speed_reward

        # 2. FORWARD DIRECTION - Must move forward
        if self.velocity_left > 0 and self.velocity_right > 0:
            forward_reward = 50

            # Bonus for both wheels at high speed
            if self.velocity_left > 0.7 * self.max_speed and self.velocity_right > 0.7 * self.max_speed:
                forward_reward += 50
        elif self.velocity_left < 0 or self.velocity_right < 0:
            forward_reward = -100  # Heavy penalty for backward
        else:
            forward_reward = -50  # Penalty for stopping

        fitness += forward_reward

        # 3. STRAIGHT MOVEMENT BONUS - Reward moving straight (faster lap times)
        speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed

        if speed_diff < 0.15:  # Very straight
            straight_reward = 40
        elif speed_diff < 0.3:  # Moderately straight
            straight_reward = 20
        elif speed_diff < 0.5:  # Some turning (acceptable for circuit)
            straight_reward = 5
        else:  # Too much turning
            straight_reward = -10

        fitness += straight_reward

        # 4. DISTANCE PROGRESS - Reward covering distance
        # Estimate distance traveled this step
        distance_this_step = avg_speed * self.max_speed * (self.time_step / 1000.0)
        self.total_distance += distance_this_step

        # Reward for distance (encourages continuous movement)
        distance_reward = distance_this_step * 100  # Scale appropriately
        fitness += distance_reward

        # 5. CONSISTENCY REWARD - Maintain consistent speed
        self.speed_history.append(avg_speed)
        if len(self.speed_history) > 20:
            self.speed_history.pop(0)

        if len(self.speed_history) >= 10:
            speed_variance = sum([(s - avg_speed) ** 2 for s in self.speed_history[-10:]]) / 10
            if speed_variance < 0.05:  # Very consistent
                consistency_reward = 30
            elif speed_variance < 0.15:
                consistency_reward = 15
            else:
                consistency_reward = 0
            fitness += consistency_reward

        # 6. ANTI-SPINNING PENALTY - Penalize spinning behavior
        if self.velocity_left * self.velocity_right < 0:  # Opposite directions
            fitness -= 50


        self.step_count += 1

        return max(0, fitness)

    # ============================================================================
    # FITNESS FUNCTION 2: FOLLOW LINE FITNESS
    # ============================================================================

    def followLineFitness(self):
        """
        Fitness for following the circuit line accurately
        Primary Goal: Stay on track line while moving fast

        Rewards:
        - Staying on black line
        - Centered on line
        - High speed while on line
        - Continuous line following

        Penalties:
        - Going off line
        - Staying off line
        - Slow speed on line
        """

        fitness = 0

        # 1. READ GROUND SENSORS
        left_ground = self.left_ir.getValue()
        center_ground = self.center_ir.getValue()
        right_ground = self.right_ir.getValue()

        # Normalize (assuming 0-1000 range: 0=black, 1000=white)
        # Adjust threshold based on your sensor calibration
        BLACK_THRESHOLD = 500  # Values below this are "on line"

        # 2. ON-LINE DETECTION
        left_on_line = left_ground < BLACK_THRESHOLD
        center_on_line = center_ground < BLACK_THRESHOLD
        right_on_line = right_ground < BLACK_THRESHOLD

        sensors_on_line = sum([left_on_line, center_on_line, right_on_line])

        # 3. LINE FOLLOWING REWARD (Critical for circuit completion)
        if sensors_on_line == 3:
            # All sensors on line - perfectly on wide line or centered
            line_reward = 150
            self.consecutive_on_line += 1
            self.consecutive_off_line = 0
            self.time_on_line += 1
        elif sensors_on_line == 2:
            # Two sensors on line - good tracking
            line_reward = 100
            self.consecutive_on_line += 1
            self.consecutive_off_line = 0
            self.time_on_line += 1
        elif sensors_on_line == 1:
            # One sensor on line - marginal, needs correction
            if center_on_line:
                line_reward = 50  # Center sensor is most important
            else:
                line_reward = 30  # Side sensor only
            self.consecutive_on_line += 1
            self.consecutive_off_line = 0
            self.time_on_line += 1
        else:
            # Completely off line - major penalty
            line_reward = -100
            self.consecutive_on_line = 0
            self.consecutive_off_line += 1
            self.time_off_line += 1

        fitness += line_reward

        # 4. CENTERING BONUS - Reward being centered on line
        if center_on_line:
            if left_on_line and right_on_line:
                # All three on line - excellent centering
                centering_bonus = 50
            elif not left_on_line and not right_on_line:
                # Only center on narrow line - perfect centering
                centering_bonus = 75
            else:
                # Center + one side
                centering_bonus = 30
            fitness += centering_bonus

        # 5. SPEED ON LINE REWARD - Encourage fast movement while on line
        avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

        if sensors_on_line >= 1:  # On line
            if avg_speed > 0.7:
                speed_on_line_reward = 80  # High speed on line is excellent
            elif avg_speed > 0.5:
                speed_on_line_reward = 50
            elif avg_speed > 0.3:
                speed_on_line_reward = 25
            else:
                speed_on_line_reward = -20  # Too slow even though on line

            fitness += speed_on_line_reward

            # Track average speed on line
            if self.time_on_line > 0:
                self.avg_speed_on_line = (self.avg_speed_on_line * (
                            self.time_on_line - 1) + avg_speed) / self.time_on_line

        # 6. CONTINUOUS LINE FOLLOWING BONUS
        if self.consecutive_on_line > 20:
            fitness += 50  # Bonus for staying on line continuously
        elif self.consecutive_on_line > 10:
            fitness += 25

        # 7. OFF-LINE PENALTY - Escalating penalty for staying off line
        if self.consecutive_off_line > 10:
            fitness -= 100  # Severe penalty - robot is lost
        elif self.consecutive_off_line > 5:
            fitness -= 50

        # 8. LINE TRACKING QUALITY - Overall quality metric
        ground_avg = (left_ground + center_ground + right_ground) / 3

        if ground_avg < 400:  # Strongly on line
            tracking_quality = 40
        elif ground_avg < 600:  # Mostly on line
            tracking_quality = 20
        elif ground_avg < 800:  # Partially on line
            tracking_quality = 0
        else:  # Off line
            tracking_quality = -30

        fitness += tracking_quality

        # 9. APPROPRIATE STEERING - Reward correct steering based on line position
        if not center_on_line:
            # Need to steer back to line
            if left_on_line and not right_on_line:
                # Line is on left, should turn left
                if self.velocity_left < self.velocity_right:
                    fitness += 20  # Correct steering
            elif right_on_line and not left_on_line:
                # Line is on right, should turn right
                if self.velocity_right < self.velocity_left:
                    fitness += 20  # Correct steering

        # 10. DISTANCE ON LINE - Track progress on line
        if sensors_on_line >= 1:
            distance_this_step = avg_speed * self.max_speed * (self.time_step / 1000.0)
            self.distance_on_line += distance_this_step

            # Reward for covering distance while on line
            fitness += distance_this_step * 50



        return max(0, fitness)

    # ============================================================================
    # FITNESS FUNCTION 3: AVOID COLLISION FITNESS
    # ============================================================================

    def avoidCollisionFitness(self):
        """
        Fitness for collision avoidance with obstacles
        Primary Goal: Navigate circuit without hitting obstacles

        Rewards:
        - Maintaining clearance from obstacles
        - Successful avoidance maneuvers
        - Quick recovery after avoidance

        Penalties:
        - Getting close to obstacles
        - Collisions
        - Stopping due to obstacles
        """

        fitness = 0

        # 1. READ ALL PROXIMITY SENSORS
        proximity_values = [self.proximity_sensors[i].getValue() for i in range(8)]

        # E-puck sensor layout:
        # 0: front-right (45°), 1: right-front (90°), 2: right-side (90°), 3: right-back (180°)
        # 4: back (180°), 5: left-back (180°), 6: left-side (90°), 7: front-left (45°)

        front_sensors = [proximity_values[0], proximity_values[7]]  # Front sensors
        front_side_sensors = [proximity_values[1], proximity_values[6]]  # Front-side
        side_sensors = [proximity_values[2], proximity_values[5]]  # Pure side
        back_sensors = [proximity_values[3], proximity_values[4]]  # Back

        max_proximity = max(proximity_values)
        max_front = max(front_sensors)
        max_front_side = max(front_side_sensors)
        avg_proximity = sum(proximity_values) / 8

        # 2. CLEARANCE REWARD - Reward for free space (essential for fast circuit)
        if max_proximity < 300:  # Very clear
            clearance_reward = 100
        elif max_proximity < 800:  # Clear
            clearance_reward = 75
        elif max_proximity < 1500:  # Moderate clearance
            clearance_reward = 50
        elif max_proximity < 2500:  # Getting close
            clearance_reward = 20
        elif max_proximity < 3500:  # Very close
            clearance_reward = -50
        else:  # Danger zone
            clearance_reward = -150
            self.near_collision_count += 1
            self.time_near_obstacle += 1

        fitness += clearance_reward

        # 3. COLLISION PENALTY - Severe penalty for collisions
        if max_proximity > 3800:  # Collision threshold
            collision_penalty = 300
            self.collision_count += 1
            fitness -= collision_penalty
        elif max_proximity > 3500:  # Near collision
            fitness -= 150
        elif max_proximity > 3000:  # Close call
            fitness -= 75

        # 4. MOVEMENT DESPITE OBSTACLES - Should keep moving even with obstacles
        avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

        if max_proximity > 2000:  # Obstacle detected
            if avg_speed > 0.4:
                # Good - still moving despite obstacle
                movement_reward = 50
            elif avg_speed > 0.2:
                movement_reward = 20
            else:
                # Bad - stopped due to obstacle
                movement_reward = -40
        else:  # No obstacle
            if avg_speed > 0.6:
                movement_reward = 40  # Reward fast movement in clear space
            else:
                movement_reward = 10

        fitness += movement_reward

        # 5. AVOIDANCE BEHAVIOR - Reward appropriate reactions
        if max_front > 2500:  # Obstacle in front
            # Should be turning to avoid
            speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed

            if speed_diff > 0.4:
                # Good avoidance - turning significantly
                avoidance_reward = 60
            elif speed_diff > 0.2:
                # Moderate avoidance
                avoidance_reward = 30
            else:
                # Not avoiding enough - dangerous
                avoidance_reward = -40

            fitness += avoidance_reward

            # Should also be slowing down appropriately
            if max_front > 3500:  # Very close
                if avg_speed < 0.3:
                    fitness += 30  # Good - slowed down
                else:
                    fitness -= 30  # Bad - too fast near obstacle

        # 6. SIDE CLEARANCE - Reward maintaining side clearance (important for circuit)
        max_side = max(side_sensors)
        if max_side < 1000:
            fitness += 30  # Good side clearance
        elif max_side > 2500:
            fitness -= 20  # Too close to side obstacles

        # 7. FRONT CLEARANCE PRIORITY - Front clearance is most critical
        if max_front < 1500:
            fitness += 50  # Excellent front clearance
        elif max_front < 2500:
            fitness += 25  # Good front clearance

        # 8. SMOOTH AVOIDANCE - Reward smooth avoidance maneuvers
        if max_proximity > 2000 and max_proximity < 3500:
            # In avoidance zone
            if 0.2 < avg_speed < 0.7:  # Moderate speed
                speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
                if 0.2 < speed_diff < 0.6:  # Smooth turning
                    fitness += 40  # Smooth avoidance maneuver

        # 9. RECOVERY BONUS - Reward getting back to clear space after obstacle
        if not hasattr(self, 'previous_max_proximity'):
            self.previous_max_proximity = 0

        if self.previous_max_proximity > 3000 and max_proximity < 2000:
            # Successfully avoided obstacle and cleared it
            fitness += 80

        self.previous_max_proximity = max_proximity

        # 10. EXPLORATION REWARD - Reward moving in open space at high speed
        if avg_proximity < 1000 and avg_speed > 0.7:
            fitness += 60  # Excellent - fast in clear space



        return max(0, fitness)

    # ============================================================================
    # FITNESS FUNCTION 4: SPINNING FITNESS (PENALTY)
    # ============================================================================

    def spinningFitness(self):
        """
        Fitness to PENALIZE spinning behavior
        For circuit racing, spinning is undesirable and wastes time

        This function returns NEGATIVE fitness for spinning
        and POSITIVE fitness for non-spinning behavior

        Goal: Discourage spinning, encourage forward motion
        """

        fitness = 0

        # 1. DETECT SPINNING - Wheels moving in opposite directions
        opposite_direction = (self.velocity_left * self.velocity_right) < 0

        if opposite_direction:
            # SPINNING DETECTED - Major penalty
            spin_magnitude = abs(self.velocity_left - self.velocity_right) / self.max_speed

            # Stronger penalty for faster spinning
            spinning_penalty = 150 * spin_magnitude
            fitness -= spinning_penalty

            # Additional penalty for spinning in place
            avg_abs_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2
            if avg_abs_speed > 0.5 * self.max_speed:
                fitness -= 100  # Heavy penalty for active spinning

        else:
            # NOT SPINNING - Reward
            fitness += 50

            # 2. REWARD COORDINATED FORWARD MOVEMENT
            if self.velocity_left > 0 and self.velocity_right > 0:
                # Both wheels forward - excellent
                fitness += 50

                # Extra reward for similar speeds (straight movement)
                speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
                if speed_diff < 0.2:
                    fitness += 40  # Moving straight
                elif speed_diff < 0.4:
                    fitness += 20  # Gentle turning (acceptable)

        # 3. PENALIZE EXCESSIVE DIFFERENTIAL
        speed_diff = abs(self.velocity_left - self.velocity_right)

        if speed_diff > 0.7 * self.max_speed and not opposite_direction:
            # Very high differential (sharp turn) - minor penalty
            fitness -= 30

        # 4. TRACK SPINNING HISTORY
        if not hasattr(self, 'spin_history'):
            self.spin_history = []

        is_spinning = 1 if opposite_direction else 0
        self.spin_history.append(is_spinning)

        if len(self.spin_history) > 20:
            self.spin_history.pop(0)

        # 5. PENALIZE FREQUENT SPINNING
        if len(self.spin_history) >= 10:
            spin_frequency = sum(self.spin_history[-10:]) / 10

            if spin_frequency > 0.5:  # Spinning more than 50% of time
                fitness -= 100  # Severe penalty
            elif spin_frequency > 0.3:
                fitness -= 50

        # 6. REWARD CONSISTENCY IN NON-SPINNING
        if len(self.spin_history) >= 20:
            if sum(self.spin_history[-20:]) == 0:
                # No spinning for 20 steps - excellent
                fitness += 50



        return max(0, fitness)

    def check_for_new_genes(self):
        if(self.flagMessage == True):
                # Split the list based on the number of layers of your network
                part = []
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        part.append((self.number_neuros_per_layer[n-1]+1)*(self.number_neuros_per_layer[n]))
                    else:   
                        part.append(self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n])
                
                # Set the weights of the network
                data = []
                weightsPart = []
                sum = 0
                for n in range(1,len(self.number_neuros_per_layer)):
                    if(n == 1):
                        weightsPart.append(self.receivedData[n-1:part[n-1]])
                    elif(n == (len(self.number_neuros_per_layer)-1)):
                        weightsPart.append(self.receivedData[sum:])
                    else:
                        weightsPart.append(self.receivedData[sum:sum+part[n-1]])
                    sum += part[n-1]
                for n in range(1,len(self.number_neuros_per_layer)):  
                    if(n == 1):
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1]+1,self.number_neuros_per_layer[n]])    
                    else:
                        weightsPart[n-1] = weightsPart[n-1].reshape([self.number_neuros_per_layer[n-1],self.number_neuros_per_layer[n]])    
                    data.append(weightsPart[n-1])                
                self.network.weights = data
                
                #Reset fitness list
                self.fitness_values = []
        
    def clip_value(self,value,min_max):
        if (value > min_max):
            return min_max
        elif (value < -min_max):
            return -min_max
        return value

    def sense_compute_and_actuate(self):
        # MLP: 
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]
        
        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left*3)
        self.right_motor.setVelocity(self.velocity_right*3)

    def calculate_fitness(self):
        
        ### Define the fitness function to increase the speed of the robot and 
        ### to encourage the robot to move forward only
        forwardFitness = self.forwardFitness()
        
        ### Define the fitness function to encourage the robot to follow the line
        followLineFitness = self.followLineFitness()
                
        ### Define the fitness function to avoid collision
        avoidCollisionFitness = self.avoidCollisionFitness()
        
        ### Define the fitness function to avoid spining behaviour
        spinningFitness = self.spinningFitness()
         
        ### Define the fitness function of this iteration which should be a combination of the previous functions         
        combinedFitness = forwardFitness + followLineFitness + avoidCollisionFitness + spinningFitness
        
        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values) 

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        #print("Robot send fitness:", string_message)
        self.emitter.send(string_message)
            
    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while(self.receiver.getQueueLength() > 0):
                # Adjust the Data to our model                
                # Webots 2022: 
                # self.receivedData = self.receiver.getData().decode("utf-8")
                # Webots 2023: 
                self.receivedData = self.receiver.getString()
                
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                #print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()
                
            # Is it a new Genotype?
            if(np.array_equal(self.receivedDataPrevious,self.receivedData) == False):
                self.flagMessage = True
                
            else:
                self.flagMessage = False
                
            self.receivedDataPrevious = self.receivedData 
        else:
            #print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):        
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []
            
            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()
            
            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            #print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))
                        
            ### Please adjust the ground sensors values to facilitate learning 
            min_gs = 0
            max_gs = 100
            
            if(left > max_gs): left = max_gs
            if(center > max_gs): center = max_gs
            if(right > max_gs): right = max_gs
            if(left < min_gs): left = min_gs
            if(center < min_gs): center = min_gs
            if(right < min_gs): right = min_gs
            
            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left-min_gs)/(max_gs-min_gs))
            self.inputs.append((center-min_gs)/(max_gs-min_gs))
            self.inputs.append((right-min_gs)/(max_gs-min_gs))
            #print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))
            
            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if(i==0 or i==1 or i==2 or i==3 or i==4 or i==5 or i==6 or i==7):        
                    temp = self.proximity_sensors[i].getValue()
                    
                    ### Please adjust the distance sensors values to facilitate learning 
                    min_ds = 0
                    max_ds = 100
                    
                    if(temp > max_ds): temp = max_ds
                    if(temp < min_ds): temp = min_ds
                    
                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp-min_ds)/(max_ds-min_ds))
                    #print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))
    
            # GA Iteration       
            # Verify if there is a new genotype to be used that was sent from Supervisor  
            self.check_for_new_genes()
            # Define the robot's actuation (motor values) based on the output of the MLP 
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()
            
            # End of the iteration

            
if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
    