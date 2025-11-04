# ============================================================================
# ADDITIONAL INITIALIZATION PARAMETERS (Add to __init__ method)
# ============================================================================

# Add these to your __init__ method after existing initializations:

# Position tracking for distance calculation
self.previous_position = None
self.total_distance = 0

# Rotation tracking for spinning behavior
self.previous_orientation = None
self.total_rotation = 0

# Time tracking
self.step_count = 0

# Sensor history for behavior analysis
self.sensor_history = []

# Fitness tracking for each behavior
self.fitness_forward = 0
self.fitness_follow_line = 0
self.fitness_avoid_collision = 0
self.fitness_spinning = 0


# ============================================================================
# FITNESS FUNCTION 1: FORWARD FITNESS
# ============================================================================

def forwardFitness(self):
    """
    Fitness function for forward movement behavior
    Objective: Move as fast as possible in forward direction
    Rewards: High speed, straight movement, continuous forward motion
    Penalties: Stopping, turning, backward movement, collisions
    """

    # 1. SPEED COMPONENT - Reward high forward speed
    avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)
    speed_fitness = avg_speed * 100  # 0-100 points

    # 2. FORWARD DIRECTION - Both wheels must move forward
    if self.velocity_left > 0 and self.velocity_right > 0:
        forward_bonus = 50
        # Extra bonus for high speed forward
        if avg_speed > 0.8:
            forward_bonus += 25
    elif self.velocity_left < 0 or self.velocity_right < 0:
        forward_bonus = -75  # Heavy penalty for backward
    else:
        forward_bonus = -25  # Penalty for stopping

    # 3. STRAIGHT MOVEMENT - Minimize turning
    speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
    straight_bonus = (1 - speed_diff) * 50  # 0-50 points

    # 4. COLLISION AVOIDANCE - Penalty for obstacles
    front_sensors = [
        self.proximity_sensors[0].getValue(),  # Front-right
        self.proximity_sensors[1].getValue(),  # Front-right-side
        self.proximity_sensors[6].getValue(),  # Front-left-side
        self.proximity_sensors[7].getValue()  # Front-left
    ]
    max_front_proximity = max(front_sensors)

    if max_front_proximity > 3000:  # Very close to obstacle
        collision_penalty = 100
    elif max_front_proximity > 2000:
        collision_penalty = 50
    else:
        collision_penalty = (max_front_proximity / 4096) * 30

    # 5. ACTIVITY PENALTY - Discourage inactivity
    if avg_speed < 0.05:
        activity_penalty = 50
    else:
        activity_penalty = 0

    # Calculate total fitness
    fitness = (
            speed_fitness +
            forward_bonus +
            straight_bonus -
            collision_penalty -
            activity_penalty
    )

    fitness = max(0, fitness)


    return fitness


# ============================================================================
# FITNESS FUNCTION 2: FOLLOW LINE FITNESS
# ============================================================================

def followLineFitness(self):
    """
    Fitness function for line following behavior
    Objective: Stay on the black track line and follow it smoothly
    Rewards: Staying on line, moderate speed, smooth following
    Penalties: Going off track, stopping, erratic movement
    """

    # 1. LINE DETECTION - Read ground sensors
    left_ground = self.left_ir.getValue()
    center_ground = self.center_ir.getValue()
    right_ground = self.right_ir.getValue()

    # Normalize (0 = black/line, 1000 = white/off-line)
    # Assuming sensor range 0-1000

    # 2. ON-LINE REWARD - Center sensor should detect line
    if center_ground < 400:  # On black line
        center_on_line = 100
    elif center_ground < 600:  # Partially on line
        center_on_line = 50
    else:  # Off line
        center_on_line = 0

    # 3. CENTERING BONUS - Robot centered on line
    # Best case: center is dark, sides are light (or all dark for wide lines)
    if center_ground < 400:
        if left_ground < 500 and right_ground < 500:
            # All sensors on line (good for wide lines)
            centering_bonus = 50
        elif left_ground > 600 and right_ground > 600:
            # Only center on line (perfectly centered on narrow line)
            centering_bonus = 75
        else:
            # Partially centered
            centering_bonus = 25
    else:
        centering_bonus = 0

    # 4. LINE TRACKING QUALITY - Penalize being off track
    ground_avg = (left_ground + center_ground + right_ground) / 3
    if ground_avg < 500:  # Mostly on line
        tracking_quality = 50
    elif ground_avg < 700:  # Partially on line
        tracking_quality = 25
    else:  # Completely off line
        tracking_quality = -50

    # 5. SPEED COMPONENT - Moderate speed is good for line following
    avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

    # Optimal speed for line following: 0.4-0.7 of max_speed
    if 0.4 <= avg_speed <= 0.7:
        speed_fitness = 50
    elif 0.2 <= avg_speed < 0.4:
        speed_fitness = 30
    elif avg_speed > 0.7:
        speed_fitness = 20  # Too fast may lose line
    else:
        speed_fitness = -25  # Too slow or stopped

    # 6. SMOOTH MOVEMENT - Reward smooth steering
    speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed

    # Some turning is expected for line following
    if speed_diff < 0.3:  # Smooth turning
        smooth_bonus = 30
    elif speed_diff < 0.5:
        smooth_bonus = 15
    else:  # Too much turning (erratic)
        smooth_bonus = -20

    # 7. FORWARD DIRECTION - Should move forward
    if self.velocity_left > 0 and self.velocity_right > 0:
        forward_bonus = 25
    else:
        forward_bonus = -25

    # Calculate total fitness
    fitness = (
            center_on_line +
            centering_bonus +
            tracking_quality +
            speed_fitness +
            smooth_bonus +
            forward_bonus
    )

    fitness = max(0, fitness)

    return fitness


# ============================================================================
# FITNESS FUNCTION 3: AVOID COLLISION FITNESS
# ============================================================================

def avoidCollisionFitness(self):
    """
    Fitness function for collision avoidance behavior
    Objective: Navigate environment while avoiding obstacles
    Rewards: Movement with clear space, successful obstacle avoidance
    Penalties: Getting close to obstacles, collisions, stopping
    """

    # 1. READ ALL PROXIMITY SENSORS
    proximity_values = [self.proximity_sensors[i].getValue() for i in range(8)]

    # Sensor layout (e-puck):
    # 0: front-right, 1: right-front, 2: right, 3: right-back
    # 4: back, 5: left-back, 6: left, 7: left-front

    front_sensors = [proximity_values[0], proximity_values[7]]  # Front
    side_sensors = [proximity_values[1], proximity_values[2],
                    proximity_values[5], proximity_values[6]]  # Sides
    back_sensors = [proximity_values[3], proximity_values[4]]  # Back

    max_proximity = max(proximity_values)
    avg_proximity = sum(proximity_values) / len(proximity_values)
    max_front = max(front_sensors)

    # 2. CLEARANCE REWARD - Reward for maintaining distance from obstacles
    if max_proximity < 500:  # Very clear space
        clearance_reward = 100
    elif max_proximity < 1000:  # Good clearance
        clearance_reward = 75
    elif max_proximity < 2000:  # Moderate clearance
        clearance_reward = 50
    elif max_proximity < 3000:  # Close to obstacle
        clearance_reward = 20
    else:  # Very close - danger!
        clearance_reward = -50

    # 3. COLLISION PENALTY - Heavy penalty for being too close
    if max_proximity > 3500:  # Imminent collision
        collision_penalty = 150
    elif max_proximity > 3000:  # Very close
        collision_penalty = 100
    elif max_proximity > 2500:  # Close
        collision_penalty = 50
    else:
        collision_penalty = 0

    # 4. MOVEMENT REWARD - Should keep moving
    avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

    if avg_speed > 0.5:
        movement_reward = 75
    elif avg_speed > 0.3:
        movement_reward = 50
    elif avg_speed > 0.1:
        movement_reward = 25
    else:
        movement_reward = -25  # Penalty for stopping

    # 5. AVOIDANCE BEHAVIOR - Reward appropriate reactions to obstacles
    # If obstacle in front, robot should turn (differential speed)
    if max_front > 2000:  # Obstacle detected in front
        speed_diff = abs(self.velocity_left - self.velocity_right) / self.max_speed
        if speed_diff > 0.3:  # Robot is turning to avoid
            avoidance_bonus = 50
        else:  # Not turning enough
            avoidance_bonus = -25
    else:  # No front obstacle
        avoidance_bonus = 25  # Bonus for clear navigation

    # 6. EXPLORATION BONUS - Reward for moving in open space
    if avg_proximity < 1000 and avg_speed > 0.4:
        exploration_bonus = 50
    else:
        exploration_bonus = 0

    # 7. FORWARD PREFERENCE - Prefer forward movement
    if self.velocity_left > 0 and self.velocity_right > 0:
        forward_bonus = 25
    else:
        forward_bonus = 0

    # Calculate total fitness
    fitness = (
            clearance_reward +
            movement_reward +
            avoidance_bonus +
            exploration_bonus +
            forward_bonus -
            collision_penalty
    )

    fitness = max(0, fitness)


    return fitness


# ============================================================================
# FITNESS FUNCTION 4: SPINNING FITNESS
# ============================================================================

def spinningFitness(self):
    """
    Fitness function for spinning/rotation behavior
    Objective: Rotate in place or perform circular motion
    Rewards: Rotational movement, consistent spinning speed
    Penalties: Forward movement, stopping, inconsistent rotation
    """

    # 1. DIFFERENTIAL SPEED - Wheels should move in opposite or highly differential speeds
    speed_diff = abs(self.velocity_left - self.velocity_right)

    # For spinning in place: wheels should move in opposite directions
    opposite_direction = (self.velocity_left * self.velocity_right) < 0

    if opposite_direction:
        # Perfect spin in place
        spin_quality = 100
        # Reward higher differential speed
        spin_speed_bonus = (speed_diff / (2 * self.max_speed)) * 50
    else:
        # Circular motion (both forward but different speeds)
        if speed_diff > 0.3 * self.max_speed:
            spin_quality = 60
            spin_speed_bonus = (speed_diff / (2 * self.max_speed)) * 30
        else:
            spin_quality = 20
            spin_speed_bonus = 0

    # 2. ROTATION SPEED - Reward active rotation
    avg_abs_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / 2

    if avg_abs_speed > 0.5 * self.max_speed:
        rotation_speed_reward = 50
    elif avg_abs_speed > 0.3 * self.max_speed:
        rotation_speed_reward = 30
    elif avg_abs_speed > 0.1 * self.max_speed:
        rotation_speed_reward = 15
    else:
        rotation_speed_reward = -25  # Penalty for not moving

    # 3. CONSISTENCY - Reward consistent spinning behavior
    # Track speed history for consistency check
    if not hasattr(self, 'spin_history'):
        self.spin_history = []

    current_spin_rate = self.velocity_left - self.velocity_right
    self.spin_history.append(current_spin_rate)

    # Keep only recent history
    if len(self.spin_history) > 10:
        self.spin_history.pop(0)

    # Check consistency
    if len(self.spin_history) >= 5:
        spin_variance = sum([(x - current_spin_rate) ** 2 for x in self.spin_history[-5:]]) / 5
        if spin_variance < 0.1:  # Consistent spinning
            consistency_bonus = 40
        elif spin_variance < 0.3:
            consistency_bonus = 20
        else:
            consistency_bonus = 0
    else:
        consistency_bonus = 0

    # 4. ANTI-FORWARD PENALTY - Penalize moving straight forward
    if abs(self.velocity_left - self.velocity_right) < 0.2 * self.max_speed:
        # Moving too straight
        straight_penalty = 50
    else:
        straight_penalty = 0

    # 5. DIRECTION CONSISTENCY - Reward spinning in same direction
    if not hasattr(self, 'spin_direction'):
        self.spin_direction = None

    current_direction = 1 if (self.velocity_left - self.velocity_right) > 0 else -1

    if self.spin_direction is None:
        self.spin_direction = current_direction
        direction_bonus = 0
    elif self.spin_direction == current_direction:
        direction_bonus = 30  # Consistent direction
    else:
        direction_bonus = -20  # Changed direction
        self.spin_direction = current_direction

    # 6. ACTIVITY REWARD - Must be actively spinning
    if avg_abs_speed < 0.05:
        activity_penalty = 50
    else:
        activity_penalty = 0

    # Calculate total fitness
    fitness = (
            spin_quality +
            spin_speed_bonus +
            rotation_speed_reward +
            consistency_bonus +
            direction_bonus -
            straight_penalty -
            activity_penalty
    )

    fitness = max(0, fitness)


    return fitness


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_fitness_values(self):
    """
    Return dictionary of all fitness values
    """
    return {
        'forward': self.fitness_forward,
        'follow_line': self.fitness_follow_line,
        'avoid_collision': self.fitness_avoid_collision,
        'spinning': self.fitness_spinning,
        'total': self.fitness
    }


def reset_fitness(self):
    """
    Reset all fitness values for new evaluation
    """
    self.fitness_values = []
    self.fitness = 0
    self.fitness_forward = 0
    self.fitness_follow_line = 0
    self.fitness_avoid_collision = 0
    self.fitness_spinning = 0

    # Reset tracking variables
    self.previous_position = None
    self.total_distance = 0
    self.previous_orientation = None
    self.total_rotation = 0
    self.step_count = 0
    self.sensor_history = []

    # Reset spinning-specific tracking
    if hasattr(self, 'spin_history'):
        self.spin_history = []
    if hasattr(self, 'spin_direction'):
        self.spin_direction = None


def calculate_combined_fitness(self, weights=None):
    """
    Calculate weighted combination of all fitness functions

    Args:
        weights: dict with keys 'forward', 'follow_line', 'avoid_collision', 'spinning'
                 If None, uses equal weights
    """
    if weights is None:
        weights = {
            'forward': 0.25,
            'follow_line': 0.25,
            'avoid_collision': 0.25,
            'spinning': 0.25
        }

    # Calculate individual fitness values
    f_forward = self.forwardFitness()
    f_follow = self.followLineFitness()
    f_avoid = self.avoidCollisionFitness()
    f_spin = self.spinningFitness()

    # Weighted combination
    combined = (
            weights['forward'] * f_forward +
            weights['follow_line'] * f_follow +
            weights['avoid_collision'] * f_avoid +
            weights['spinning'] * f_spin
    )

    self.fitness += combined
    self.fitness_values.append(combined)

    return combined


def get_average_fitness(self, fitness_type='total'):
    """
    Get average fitness over episode

    Args:
        fitness_type: 'total', 'forward', 'follow_line', 'avoid_collision', or 'spinning'
    """
    if fitness_type == 'total':
        if len(self.fitness_values) > 0:
            return self.fitness / len(self.fitness_values)
    elif fitness_type == 'forward':
        return self.fitness_forward / max(1, self.step_count)
    elif fitness_type == 'follow_line':
        return self.fitness_follow_line / max(1, self.step_count)
    elif fitness_type == 'avoid_collision':
        return self.fitness_avoid_collision / max(1, self.step_count)
    elif fitness_type == 'spinning':
        return self.fitness_spinning / max(1, self.step_count)

    return 0