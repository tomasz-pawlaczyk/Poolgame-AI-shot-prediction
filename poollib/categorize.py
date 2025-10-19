import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans


def categorize(x, y, img):
    ball_radius = 23

    # Create a circular mask for the detected ball
    ball_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.circle(ball_mask, (x, y), ball_radius, 255, -1)  # white mask over the ball area
    isolated_ball = cv.bitwise_and(img, img, mask=ball_mask)

    # Calculate the dominant color and its ratio
    color_hsv, color_bgr, ratio = calculate_color(isolated_ball)

    # Determine the ball type based on color and ratio
    ball_type = calculate_type(color_hsv, ratio)

    # Return the ball type and its color (in BGR)
    return ball_type, color_bgr[0, 0].tolist()


def calculate_color(img):
    # Convert from BGR to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Filter out dark pixels (ignore background)
    # Condition: brightness (V) > 5 and saturation (S) > 5
    mask = (pixels[:, 1] > 5) & (pixels[:, 2] > 5)
    filtered_pixels = pixels[mask]

    if len(filtered_pixels) == 0:
        return [[0, 0, 0]]  # fallback: no meaningful pixels found

    # Use KMeans to find dominant color clusters
    kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)
    kmeans.fit(filtered_pixels)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    centers = kmeans.cluster_centers_

    # Identify the cluster with the highest saturation (S)
    saturations = centers[:, 1]
    values = centers[:, 2]
    color_idx = np.argmax(saturations)

    # Identify the whitest cluster: high V and low S
    whiteness_score = values - saturations
    white_idx = np.argmax(whiteness_score)

    # Count how many pixels belong to each cluster
    color_count = counts[color_idx]
    white_count = counts[white_idx]

    # Compute color-to-white ratio (how much of the ball is colored vs. white)
    ratio = color_count / white_count if white_count > 0 else float('inf')

    # Adjust ratio if the “white” cluster isn’t truly white
    if centers[white_idx][2] < 200 and centers[white_idx][1] > 80:
        ratio = 10.0

    # Get the dominant HSV color and convert it back to BGR
    sorted_idx = np.argsort(-counts)
    dominant_hsv = kmeans.cluster_centers_[sorted_idx].astype(int)
    color_hsv = max(dominant_hsv, key=lambda hsv: hsv[1])
    hsv_array = np.array(color_hsv, dtype=np.uint8).reshape(1, 1, 3)
    color_bgr = cv.cvtColor(hsv_array, cv.COLOR_HSV2BGR)

    return color_hsv, color_bgr, ratio


def calculate_type(color, ratio):
    BLACK_THRESHOLD_V = 80
    WHITE_THRESHOLD_S = 80
    WHITE_THRESHOLD_RATIO = 1.1
    FULL_THRESHOLD_RATIO = 2.0

    h, s, v = color

    # Check if the ball is black (low brightness)
    black = v < BLACK_THRESHOLD_V

    # Check if the ball is white (low saturation)
    white = s < WHITE_THRESHOLD_S

    # Classify the ball based on color and ratio
    if ratio < WHITE_THRESHOLD_RATIO and white:
        return "white"
    elif black:
        return "black"
    elif ratio < FULL_THRESHOLD_RATIO:
        return "stripe"
    else:
        return "solid"
