import math

# Assumed ball diameter
BALL_DIAMETER = 23 * 2


# Calculate where the white ball should hit (ghost ball) based on the desired target path
def get_ghost(y):
    target, hole = y

    # Compute the hit vector
    dx = hole[0] - target[0]
    dy = hole[1] - target[1]

    # Compute vector magnitude (distance)
    distance = math.sqrt(dx ** 2 + dy ** 2)
    unit_x = dx / distance
    unit_y = dy / distance

    # Ghost ball position is 1 ball diameter behind the target along the line
    return {
        "x": target[0] - unit_x * BALL_DIAMETER,
        "y": target[1] - unit_y * BALL_DIAMETER
    }


# Calculate cut angle for the shot, useful for evaluating difficulty and feasibility
def get_cut_angle(white_dir, target_dir):
    if len(white_dir) > 1:
        white_x, white_y = white_dir["y2"][0]
        ghost_x, ghost_y = white_dir["y2"][1]
    else:
        white_x, white_y = white_dir["y1"][0]
        ghost_x, ghost_y = white_dir["y1"][1]

    target_x, target_y = target_dir["y1"][0]
    hole_x, hole_y = target_dir["y1"][1]

    # Vector from white ball to target (ghost ball)
    v1x = white_x - ghost_x
    v1y = white_y - ghost_y

    # Vector from target ball to hole
    v2x = hole_x - target_x
    v2y = hole_y - target_y

    # Vector magnitudes
    mag1 = math.sqrt(v1x ** 2 + v1y ** 2)
    mag2 = math.sqrt(v2x ** 2 + v2y ** 2)

    # Dot product
    dot_product = v1x * v2x + v1y * v2y

    # Calculate angle in radians
    cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
    angle_rad = math.acos(cos_angle)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return round(angle_deg, 1)
