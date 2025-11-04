# ============================================================================
# ADDITIONAL INITIALIZATION PARAMETERS (Add to __init__ method)
# ============================================================================

# Add these to your __init__ method after existing initializations:

# Position and distance tracking
self.previous_position = None
self.current_position = None
self.total_distance = 0
self.distance_on_line = 0

# Circuit completion tracking
self.lap_progress = 0  # Estimated progress around circuit (0-1)
self.checkpoints_passed = 0
self.circuit_perimeter = 1.5 * 4  # 6 meters for square track

# Time tracking
self.step_count = 0
self.time_on_line = 0
self.time_off_line = 0

# Line following tracking
self.consecutive_on_line = 0
self.consecutive_off_line = 0

# Collision tracking
self.collision_count = 0
self.near_collision_count = 0
self.time_near_obstacle = 0

# Speed tracking
self.speed_history = []
self.avg_speed_on_line = 0

# Fitness components
self.fitness_forward = 0
self.fitness_follow_line = 0
self.fitness_avoid_collision = 0
self.fitness_spinning = 0


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

    # Update cumulative fitness
    self.fitness_forward += max(0, fitness)
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
            self.avg_speed_on_line = (self.avg_speed_on_line * (self.time_on_line - 1) + avg_speed) / self.time_on_line

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

    # Update cumulative fitness
    self.fitness_follow_line += max(0, fitness)

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

    # Update cumulative fitness
    self.fitness_avoid_collision += max(0, fitness)

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

    # Update cumulative fitness
    self.fitness_spinning += max(0, fitness)

    return max(0, fitness)


# ============================================================================
# COMBINED FITNESS FUNCTION FOR CIRCUIT RACING
# ============================================================================

def calculate_circuit_fitness(self):
    """
    Combined fitness function optimized for circuit lap completion

    Weights are tuned for:
    - Fast lap times (forward + follow line)
    - Staying on track (follow line)
    - Avoiding obstacles (avoid collision)
    - Preventing spinning (spinning penalty)
    """

    # Calculate individual fitness components
    f_forward = self.forwardFitness()
    f_follow = self.followLineFitness()
    f_avoid = self.avoidCollisionFitness()
    f_spin = self.spinningFitness()

    # Optimized weights for circuit racing
    weights = {
        'forward': 0.30,  # 30% - Speed is important
        'follow_line': 0.45,  # 45% - Most important - must stay on track
        'avoid_collision': 0.20,  # 20% - Important but secondary to line following
        'spinning': 0.05  # 5% - Small weight to penalize spinning
    }

    # Calculate weighted combination
    combined_fitness = (
            weights['forward'] * f_forward +
            weights['follow_line'] * f_follow +
            weights['avoid_collision'] * f_avoid +
            weights['spinning'] * f_spin
    )

    # BONUS: Lap completion estimation
    # Estimate progress based on distance on line
    if self.distance_on_line > 0:
        estimated_laps = self.distance_on_line / self.circuit_perimeter

        # Bonus for completing laps
        if estimated_laps >= 1.0:
            combined_fitness += 500 * estimated_laps  # Major bonus for lap completion
        elif estimated_laps >= 0.75:
            combined_fitness += 200
        elif estimated_laps >= 0.5:
            combined_fitness += 100
        elif estimated_laps >= 0.25:
            combined_fitness += 50

    # PENALTY: Excessive collisions or off-track time
    if self.collision_count > 3:
        combined_fitness -= 500  # Severe penalty for multiple collisions

    if self.time_off_line > self.time_on_line and self.step_count > 50:
        combined_fitness -= 300  # Penalty for being off track most of the time

    # Update total fitness
    self.fitness += combined_fitness
    self.fitness_values.append(combined_fitness)

    return combined_fitness


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_fitness_summary(self):
    """
    Return comprehensive fitness summary
    """
    total_time = self.step_count * self.time_step / 1000.0  # seconds

    return {
        'total_fitness': self.fitness,
        'forward_fitness': self.fitness_forward,
        'follow_line_fitness': self.fitness_follow_line,
        'avoid_collision_fitness': self.fitness_avoid_collision,
        'spinning_fitness': self.fitness_spinning,
        'total_distance': self.total_distance,
        'distance_on_line': self.distance_on_line,
        'time_on_line': self.time_on_line,
        'time_off_line': self.time_off_line,
        'collision_count': self.collision_count,
        'near_collision_count': self.near_collision_count,
        'avg_speed_on_line': self.avg_speed_on_line,
        'estimated_laps': self.distance_on_line / self.circuit_perimeter,
        'total_time': total_time,
        'avg_fitness_per_step': self.fitness / max(1, self.step_count)
    }


def reset_fitness(self):
    """
    Reset all fitness and tracking variables for new evaluation
    """
    # Fitness values
    self.fitness_values = []
    self.fitness = 0
    self.fitness_forward = 0
    self.fitness_follow_line = 0
    self.fitness_avoid_collision = 0
    self.fitness_spinning = 0

    # Distance tracking
    self.previous_position = None
    self.current_position = None
    self.total_distance = 0
    self.distance_on_line = 0

    # Circuit tracking
    self.lap_progress = 0
    self.checkpoints_passed = 0

    # Time tracking
    self.step_count = 0
    self.time_on_line = 0
    self.time_off_line = 0

    # Line following
    self.consecutive_on_line = 0
    self.consecutive_off_line = 0

    # Collision tracking
    self.collision_count = 0
    self.near_collision_count = 0
    self.time_near_obstacle = 0

    # Speed tracking
    self.speed_history = []
    self.avg_speed_on_line = 0

    # Spinning tracking
    if hasattr(self, 'spin_history'):
        self.spin_history = []
    if hasattr(self, 'previous_max_proximity'):
        self.previous_max_proximity = 0


def get_performance_metrics(self):
    """
    Calculate performance metrics for circuit racing
    """
    if self.step_count == 0:
        return {}

    total_time = self.step_count * self.time_step / 1000.0  # seconds

    metrics = {
        'lap_time': total_time,
        'distance_traveled': self.total_distance,
        'distance_on_line': self.distance_on_line,
        'line_following_ratio': self.time_on_line / max(1, self.step_count),
        'collision_rate': self.collision_count / total_time,
        'avg_speed': self.total_distance / total_time if total_time > 0 else 0,
        'avg_speed_on_line': self.avg_speed_on_line,
        'estimated_laps_completed': self.distance_on_line / self.circuit_perimeter,
        'fitness_per_meter': self.fitness / max(0.001, self.total_distance),
        'efficiency_score': (self.distance_on_line / max(0.001, total_time)) * (1 - self.collision_count * 0.1)
    }

    return metrics


# ============================================================================
# MAIN CONTROL LOOP INTEGRATION
# ============================================================================

def run_step(self):
    """
    Example integration into main control loop
    Call this each simulation step
    """

    # ... Your MLP network forward pass here ...
    # self.inputs = [sensor readings]
    # outputs = self.network.forward(self.inputs)
    # self.velocity_left = outputs[0] * self.max_speed
    # self.velocity_right = outputs[1] * self.max_speed
    # self.left_motor.setVelocity(self.velocity_left)
    # self.right_motor.setVelocity(self.velocity_right)

    # Calculate fitness for this step
    fitness = self.calculate_circuit_fitness()

    return fitness


def get_final_fitness(self):
    """
    Get final fitness score for GA evaluation
    This is what the GA will use to rank individuals
    """

    # Primary metric: total accumulated fitness
    base_fitness = self.fitness

    # Bonus for lap completion
    estimated_laps = self.distance_on_line / self.circuit_perimeter
    lap_bonus = estimated_laps * 1000  # 1000 points per lap

    # Time penalty (encourage faster laps)
    total_time = self.step_count * self.time_step / 1000.0
    if estimated_laps >= 1.0:
        # Completed at least one lap - reward faster times
        time_penalty = total_time * 10  # Penalty for slow laps
    else:
        time_penalty = 0

    # Collision penalty
    collision_penalty = self.collision_count * 200

    # Final fitness
    final_fitness = base_fitness + lap_bonus - time_penalty - collision_penalty

    return max(0, final_fitness)