import cv2 as cv
import numpy as np


def visualize(img, balls, shots):
    """
    Draws the table, balls, and calculated shots.
    img   - input image (used only for size)
    balls - list of Ball objects
    shots - list of Shot objects
    returns an image with visualization
    """
    ball_radius = 23

    # Create base table visualization
    table = create_table(img)

    # Draw shots
    for shot in shots:
        target_dir, white_dir = shot.get_lines()

        # Draw lines for rebounds if exist
        if len(target_dir) > 1:
            cv.line(table, target_dir["y2"][0], target_dir["y2"][1], (0, 0, 255), 3)
        if len(white_dir) > 1:
            cv.line(table, white_dir["y2"][0], white_dir["y2"][1], (0, 0, 255), 3)

        # Draw main shot lines
        cv.line(table, white_dir["y1"][0], white_dir["y1"][1], (0, 0, 255), 3)
        cv.line(table, target_dir["y1"][0], target_dir["y1"][1], (0, 0, 255), 3)

    # Draw balls
    for ball in balls:
        center = ball.get_coordinates()
        color = ball.get_color()

        if ball.get_type() == "stripe":
            # Draw full white circle first
            cv.circle(table, center, ball_radius, (255, 255, 255), -1)
            # Draw colored inner circle
            cv.circle(table, center, ball_radius - 8, color, -1)
        else:
            if ball.get_type() == "white":
                cv.circle(table, center, ball_radius, (255, 255, 255), -1)
            elif ball.get_type() == "black":
                cv.circle(table, center, ball_radius, (0, 0, 0), -1)
            else:
                cv.circle(table, center, ball_radius, color, -1)

    # Text parameters for angles
    font = cv.FONT_HERSHEY_DUPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    font_thickness = 2

    # Draw ghost balls and angle text
    for shot in shots:
        ghost = shot.get_ghost()
        cv.circle(table, ghost, ball_radius, (0, 255, 0), 5)
        text = f"{round(180 - shot.get_angle(), 1)}deg"
        cv.putText(table, text, (ghost[0] - 30, ghost[1] - 35),
                   font, font_scale, font_color, font_thickness)

    return table


def create_table(img):
    """
    Creates a blank table visualization.
    img - input image (used only for dimensions)
    returns a BGR image representing the table
    """
    edge_width = 25
    corner_radius = 70
    center_radius = 50
    height, width = img.shape[:2]

    table_color = (200, 150, 69)  # main table color
    edge_color = (184, 107, 69)  # table edges

    # Create blank image
    output = np.zeros_like(img)
    output[:] = table_color

    # Draw table edges
    cv.rectangle(output, (0, 0), (width, edge_width), edge_color, -1)
    cv.rectangle(output, (0, 0), (edge_width, height), edge_color, -1)
    cv.rectangle(output, (width - edge_width, 0), (width, height), edge_color, -1)
    cv.rectangle(output, (0, height - edge_width), (width, height), edge_color, -1)

    # Draw pockets (corners and sides)
    cv.circle(output, (0, 0), corner_radius, 0, -1)  # top-left
    cv.circle(output, (width, 0), corner_radius, 0, -1)  # top-right
    cv.circle(output, (0, height), corner_radius, 0, -1)  # bottom-left
    cv.circle(output, (width, height), corner_radius, 0, -1)  # bottom-right
    cv.circle(output, (0, width), center_radius, 0, -1)  # left-middle
    cv.circle(output, (width, width), center_radius, 0, -1)  # right-middle

    return output
