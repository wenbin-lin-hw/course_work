def forwardFitness(self):
    """
    Calculate fitness for forward movement on athletics track (1.5m x 1.5m outer square)
    Rewards: forward speed, staying on track, smooth movement, avoiding obstacles
    Penalties: stopping, going off track, collisions, erratic movement
    """

    # 1. SPEED COMPONENT - Reward forward movement
    # Average of left and right wheel velocities (normalized to max_speed)
    avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)
    speed_fitness = avg_speed * 100  # Scale to 0-100

    # 2. STRAIGHT MOVEMENT COMPONENT - Reward moving straight (both wheels similar speed)
    # Penalize differential steering (turning)
    speed_difference = abs(self.velocity_left - self.velocity_right) / self.max_speed
    straight_fitness = (1 - speed_difference) * 50  # Scale to 0-50

    # 3. TRACK ADHERENCE COMPONENT - Reward staying on the black track
    # Ground sensors: lower values = darker surface (on track)
    # Typical values: ~0 for black, ~1000 for white
    left_ground = self.left_ir.getValue()
    center_ground = self.center_ir.getValue()
    right_ground = self.right_ir.getValue()

    # Normalize ground sensor values (assuming 0-1000 range)
    # Lower is better (darker = on track)
    ground_avg = (left_ground + center_ground + right_ground) / 3
    on_track_fitness = max(0, (1000 - ground_avg) / 1000) * 150  # Scale to 0-150

    # Bonus if all three sensors detect track (centered on track)
    if left_ground < 500 and center_ground < 500 and right_ground < 500:
        on_track_fitness += 50  # Bonus for being well-centered

    # 4. OBSTACLE AVOIDANCE COMPONENT - Penalize proximity to obstacles
    # Read all 8 proximity sensors
    proximity_values = []
    for i in range(8):
        proximity_values.append(self.proximity_sensors[i].getValue())

    # Normalize proximity values (typical range 0-4096, higher = closer)
    max_proximity = max(proximity_values)
    collision_penalty = (max_proximity / 4096) * 100  # Scale to 0-100

    # Heavy penalty if very close to obstacle (potential collision)
    if max_proximity > 3000:
        collision_penalty += 100  # Additional penalty for near collision

    # 5. FORWARD DIRECTION COMPONENT - Ensure robot moves forward, not backward
    # Both wheels should be positive (forward)
    if self.velocity_left > 0 and self.velocity_right > 0:
        forward_bonus = 50
    elif self.velocity_left < 0 or self.velocity_right < 0:
        forward_bonus = -50  # Penalty for backward movement
    else:
        forward_bonus = 0  # Stopped

    # 6. ACTIVITY COMPONENT - Penalize staying still
    if avg_speed < 0.1:  # Nearly stopped
        activity_penalty = 50
    else:
        activity_penalty = 0

    # 7. CALCULATE TOTAL FITNESS
    fitness = (
            speed_fitness +  # 0-100: reward speed
            straight_fitness +  # 0-50: reward straight movement
            on_track_fitness +  # 0-200: reward staying on track (most important)
            forward_bonus +  # -50 to +50: ensure forward direction
            - collision_penalty +  # 0 to -200: penalize obstacles/collisions
            - activity_penalty  # 0 to -50: penalize inactivity
    )

    # Ensure fitness is non-negative
    fitness = max(0, fitness)

    # Store fitness value
    self.fitness_values.append(fitness)
    self.fitness += fitness

    return fitness


# ALTERNATIVE: Simplified version focusing on key metrics
def forwardFitness_simple(self):
    """
    Simplified fitness function focusing on core objectives
    """
    # Speed component (0-100)
    avg_speed = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)
    speed_fitness = avg_speed * 100

    # On-track component (0-100)
    ground_avg = (self.left_ir.getValue() + self.center_ir.getValue() + self.right_ir.getValue()) / 3
    on_track_fitness = max(0, (1000 - ground_avg) / 10)

    # Obstacle avoidance (0 to -50)
    max_proximity = max([self.proximity_sensors[i].getValue() for i in range(8)])
    obstacle_penalty = -(max_proximity / 4096) * 50

    # Total fitness
    fitness = speed_fitness + on_track_fitness + obstacle_penalty
    fitness = max(0, fitness)

    self.fitness_values.append(fitness)
    self.fitness += fitness

    return fitness


# HELPER: Get average fitness over episode
def get_average_fitness(self):
    """
    Calculate average fitness over the entire episode
    """
    if len(self.fitness_values) > 0:
        return self.fitness / len(self.fitness_values)
    return 0


# HELPER: Reset fitness for new episode
def reset_fitness(self):
    """
    Reset fitness values for new evaluation
    """
    self.fitness_values = []
    self.fitness = 0