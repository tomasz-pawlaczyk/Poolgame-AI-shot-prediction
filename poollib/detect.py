import cv2 as cv
import numpy as np


def detect_balls(img):
    print("\033[1m-----[Detect Balls]-----\033[0m")

    # Load image
    if img is None:
        raise Exception(f"\033[31m[DB 1/4!!!]\033[0m Invalid image")

    print(f"\033[32m[DB 1/4]\033[0m Image loaded successfully")

    # Detect balls using the Hough Transform
    table_mask = create_mask(img)

    print(f"\033[32m[DB 3/4]\033[0m Detecting balls...")
    circles = cv.HoughCircles(
        table_mask,             # input image (grayscale)
        cv.HOUGH_GRADIENT,      # detection method
        dp=1.3,                 # accumulator resolution (1.0 = same as input, >1 = smaller)
        minDist=45,             # minimum distance between detected circle centers
        param1=150,             # upper threshold for Canny edge detector
        param2=15,              # detection threshold (lower = more noise)
        minRadius=18,           # minimum ball radius
        maxRadius=35            # maximum ball radius
    )

    balls_only = cv.bitwise_and(img, img, mask=cv.bitwise_not(table_mask))
    return circles, balls_only


def create_mask(img):
    print(f"\033[32m[DB 2/4]\033[0m Creating table mask")
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Sample the table color (average of 4 points)
    pixel_color1 = np.uint8([[hsv[2, 500]]])
    pixel_color2 = np.uint8([[hsv[1998, 500]]])
    pixel_color3 = np.uint8([[hsv[4, 540]]])
    pixel_color4 = np.uint8([[hsv[1996, 540]]])

    pixel_color = np.mean([pixel_color1, pixel_color2, pixel_color3, pixel_color4], axis=0)

    # Define HSV range for the table color
    lower_color = np.array([pixel_color[0][0][0] - 7, 60, 60])
    upper_color = np.array([pixel_color[0][0][0] + 7, 255, 255])

    # Create and clean the mask
    table_mask = cv.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    table_mask = cv.morphologyEx(table_mask, cv.MORPH_CLOSE, kernel)
    table_mask = cv.morphologyEx(table_mask, cv.MORPH_OPEN, kernel)

    # Remove pockets from the mask
    table_mask = cv.bitwise_or(table_mask, pocket_mask(hsv))
    return table_mask


def calculate_white_percent(img):
    # Calculates what percentage of the image is white vs. black
    if len(img.shape) > 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Threshold to binary (if not already)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Count pixels
    total_pixels = binary.shape[0] * binary.shape[1]
    white_pixels = cv.countNonZero(binary)
    percent_white = (white_pixels / total_pixels) * 100

    return percent_white


def pocket_mask(img):
    height, width = img.shape[:2]
    final_mask = np.zeros((height, width), dtype=np.uint8)

    # List of pockets to detect
    config = ['UL', 'UR', 'LL', 'LR', 'CL', 'CR']

    for corner in config:
        white_black_ratio = 66
        running_time = 0

        while white_black_ratio > 65:
            scale = 0.10
            if running_time >= 1:
                scale += 1 * 0.2

            sw = int(scale * width)

            match corner:
                case 'UL':
                    cropped_img = img[0:sw, 0:sw]
                case 'UR':
                    cropped_img = img[0:sw, width - sw:width]
                case 'LL':
                    cropped_img = img[height - sw:height, 0:sw]
                case 'LR':
                    cropped_img = img[height - sw:height, width - sw:width]
                case 'CL':
                    cropped_img = img[int(height / 2 - sw / 2):int(height / 2 + sw / 2), 0:sw]
                case 'CR':
                    cropped_img = img[int(height / 2 - sw / 2):int(height / 2 + sw / 2), width - sw:width]
                case _:
                    pass

            # Ensure image type is uint8
            if cropped_img.dtype != np.uint8:
                if np.max(cropped_img) <= 1.0:
                    cropped_img = (cropped_img * 255).astype(np.uint8)
                else:
                    cropped_img = cropped_img.astype(np.uint8)

            # Ensure the image has 3 channels
            if len(cropped_img.shape) == 2:
                cropped_img = cv.cvtColor(cropped_img, cv.COLOR_GRAY2BGR)

            # Convert to grayscale
            gray = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)

            # Edge detection
            edges = cv.Canny(gray, 30, 150)

            # Find contours
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Draw contours
            contour_img = np.zeros_like(gray)
            cv.drawContours(contour_img, contours, -1, 255, 2)

            # Morphological closing to merge nearby contours
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            closed = cv.morphologyEx(contour_img, cv.MORPH_CLOSE, kernel)

            # Fill holes
            floodfill = closed.copy()
            h, w = floodfill.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            cv.floodFill(floodfill, mask, (0, 0), 255)
            floodfill_inv = cv.bitwise_not(floodfill)
            filled = closed | floodfill_inv

            # Build density map
            density_map = cv.GaussianBlur(contour_img, (kernel_size * 2 + 1, kernel_size * 2 + 1), 0)

            # Threshold density map
            threshold_value = 90
            _, density_mask = cv.threshold(density_map, threshold_value, 255, cv.THRESH_BINARY)

            # Combine all masks
            corner_mask = cv.bitwise_or(filled, density_mask)

            white_black_ratio = calculate_white_percent(density_mask)
            if white_black_ratio > 65:
                running_time += 1
            elif white_black_ratio < 15:
                break

            cv.bitwise_and(cropped_img, cropped_img, mask=cv.bitwise_not(density_mask))

            match corner:
                case 'UL':
                    final_mask[0:sw, 0:sw] = cv.bitwise_or(final_mask[0:sw, 0:sw], corner_mask)
                case 'UR':
                    final_mask[0:sw, width - sw:width] = cv.bitwise_or(
                        final_mask[0:sw, width - sw:width], corner_mask)
                case 'LL':
                    final_mask[height - sw:height, 0:sw] = cv.bitwise_or(
                        final_mask[height - sw:height, 0:sw], corner_mask)
                case 'LR':
                    final_mask[height - sw:height, width - sw:width] = cv.bitwise_or(
                        final_mask[height - sw:height, width - sw:width], corner_mask)
                case 'CL':
                    final_mask[int(height / 2 - sw / 2):int(height / 2 + sw / 2), 0:sw] = cv.bitwise_or(
                        final_mask[int(height / 2 - sw / 2):int(height / 2 + sw / 2), 0:sw], corner_mask)
                case 'CR':
                    final_mask[int(height / 2 - sw / 2):int(height / 2 + sw / 2), width - sw:width] = cv.bitwise_or(
                        final_mask[int(height / 2 - sw / 2):int(height / 2 + sw / 2), width - sw:width], corner_mask)

    return final_mask
