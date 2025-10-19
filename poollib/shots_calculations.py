import math
from .Shot import Shot

HOLE_DIAMETER = 40
HOLE_DIAMETER2 = HOLE_DIAMETER + 10
RECOMMENDATIONS = 3

# Create all possible shots for a given target ball
def get_shots(white, target):
    holes = [
        {"x": 0 + HOLE_DIAMETER, "y": 0 + HOLE_DIAMETER, "code": "UL"},
        {"x": 1000 - HOLE_DIAMETER, "y": 0 + HOLE_DIAMETER, "code": "UR"},
        {"x": 0 + HOLE_DIAMETER, "y": 1000, "code": "CL"},
        {"x": 1000 - HOLE_DIAMETER, "y": 1000, "code": "CR"},
        {"x": 0 + HOLE_DIAMETER, "y": 2000 - HOLE_DIAMETER, "code": "LL"},
        {"x": 1000 - HOLE_DIAMETER, "y": 2000 - HOLE_DIAMETER, "code": "LR"},
    ]

    # Edges in order: top, right, bottom, left
    edges = {"U": (holes[0], holes[1]),
             "R": (holes[1], holes[5]),
             "D": (holes[4], holes[5]),
             "L": (holes[0], holes[4])}

    shots = []

    for hole in holes:
        # Direct shot
        new_shot = Shot(white, target, hole)
        shots.append(new_shot)

        # Shots with one edge reflection before pocket
        match hole["code"]:
            case "UL":
                edge_target1 = find_edge_point(edges["D"], target, hole)
                edge_target2 = find_edge_point(edges["R"], target, hole)
            case "UR":
                edge_target1 = find_edge_point(edges["L"], target, hole)
                edge_target2 = find_edge_point(edges["D"], target, hole)
            case "LL":
                edge_target1 = find_edge_point(edges["U"], target, hole)
                edge_target2 = find_edge_point(edges["R"], target, hole)
            case "LR":
                edge_target1 = find_edge_point(edges["L"], target, hole)
                edge_target2 = find_edge_point(edges["U"], target, hole)
            case "CL":
                edge_target1 = find_edge_point(edges["D"], target, hole)
                edge_target2 = find_edge_point(edges["U"], target, hole)
            case "CR":
                edge_target1 = find_edge_point(edges["D"], target, hole)
                edge_target2 = find_edge_point(edges["U"], target, hole)

        new_shot_edge1 = Shot(white, target, hole, edge_target1)
        new_shot_edge2 = Shot(white, target, hole, edge_target2)
        shots.append(new_shot_edge1)
        shots.append(new_shot_edge2)

        ghost_balls = [
            [new_shot.get_ghost(), None],
            [new_shot_edge1.get_ghost(), edge_target1],
            [new_shot_edge2.get_ghost(), edge_target2]
        ]

        # Shots where white ball bounces off a wall before hitting the target
        for ghost in ghost_balls:
            for edge in edges.values():
                edge_white = find_edge_point(edge, white, ghost[0])
                shots.append(Shot(white, target, hole, ghost[1], edge_white))

    return shots

def find_edge_point(edge, start, end):
    x1, y1 = start.get_coordinates()

    if len(end) > 2:
        x2, y2 = end["x"], end["y"]
    else:
        x2, y2 = end

    (edge_start, edge_end) = edge
    x3, y3 = edge_start["x"], edge_start["y"]
    x4, y4 = edge_end["x"], edge_end["y"]

    # Determine if the edge is horizontal or vertical
    if y3 == y4:
        mirrored_x, mirrored_y = x2, 2 * y3 - y2
    elif x3 == x4:
        mirrored_x, mirrored_y = 2 * x3 - x2, y2

    dx = mirrored_x - x1
    dy = mirrored_y - y1

    if dx == 0:
        x_int = x1
        if y3 == y4:
            y_int = y3
        else:
            return None
    elif dy == 0:
        y_int = y1
        if x3 == x4:
            x_int = x3
        else:
            return None
    else:
        m = dy / dx
        b = y1 - m * x1
        if y3 == y4:
            y_int = y3
            x_int = (y_int - b) / m
        elif x3 == x4:
            x_int = x3
            y_int = m * x_int + b
        else:
            return None

    return {"x": x_int, "y": y_int}

def ball_shot_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))

    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy

    return math.hypot(px - nearest_x, py - nearest_y)

def is_shot_blocked(shot, balls, ball_radius):
    (x1, y1), (x2, y2) = shot

    for ball in balls:
        bx, by = ball.get_coordinates()
        if (bx, by) == (x1, y1) or (bx, by) == (x2, y2):
            continue

        dist = ball_shot_dist(bx, by, x1, y1, x2, y2)
        if dist < 2 * ball_radius - 2:
            return True

    return False

def is_edge_possible(line):
    x, y = line[0]

    if x < HOLE_DIAMETER2 and y < HOLE_DIAMETER2:
        return False
    elif x > 1000 - HOLE_DIAMETER2 and y < HOLE_DIAMETER2:
        return False
    elif x < HOLE_DIAMETER2 and 1000 - HOLE_DIAMETER2 < y < 1000 + HOLE_DIAMETER2:
        return False
    elif x > 1000 - HOLE_DIAMETER2 and 1000 - HOLE_DIAMETER2 < y < 1000 + HOLE_DIAMETER2:
        return False
    elif x < HOLE_DIAMETER2 and y > 2000 - HOLE_DIAMETER2:
        return False
    elif x > 1000 - HOLE_DIAMETER2 and y > 2000 - HOLE_DIAMETER2:
        return False
    else:
        return True

def get_best_shots(shots):
    factors = []
    best_shots = []

    if len(shots) <= RECOMMENDATIONS:
        return shots

    for shot in shots:
        print(".", end='')

        t1, w1 = shot.get_lines()
        if len(t1) > 1 and len(w1) > 1:
            edge_shot = 8.0
        elif len(t1) > 1 or len(w1) > 1:
            edge_shot = 4.0
        else:
            edge_shot = 1.0

        factor = ((180 - shot.get_angle()) * 300 + shot.get_length()) * edge_shot
        factors.append(factor)

    for i in range(RECOMMENDATIONS):
        index = factors.index(min(factors))
        best_shots.append(shots[index])
        del shots[index]
        del factors[index]

    return best_shots
