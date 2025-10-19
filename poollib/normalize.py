import cv2 as cv
import numpy as np


#### NORMALIZE ####
# This function normalizes the colors in an image based on
# a detected white object. Returns a normalized image.
# img - image to normalize
# max_gain_factor - limits channel amplification to prevent overexposure
# percentile_ref - reference white color is selected based on this percentile

def normalize(img, max_gain_factor=2.0, percentile_ref=95):
    print("\033[1m-----[Normalize Photo]-----\033[0m")

    if img is None:
        raise Exception(f"\033[31m[NP 1/3!!!]\033[0m Invalid image")

    print(f"\033[32m[NP 1/3]\033[0m Image loaded successfully")

    center, radius = find_white(img)

    if center and radius and radius > 0:
        # Properly loaded image
        img_data = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width = img_data.shape[:2]

        # Coordinates of rectangular ROI around the white ball
        x1 = max(0, center[0] - radius)
        y1 = max(0, center[1] - radius)
        x2 = min(width, center[0] + radius + 1)
        y2 = min(height, center[1] + radius + 1)

        if x1 >= x2 or y1 >= y2:
            raise Exception(f"\033[31m[NP 3/3!!!]\033[0m Normalization failed")

        # Crop the ROI from image
        roi = img_data[y1:y2, x1:x2]
        roi_h, roi_w, _ = roi.shape

        # Center coordinates relative to ROI
        roi_center_x = center[0] - x1
        roi_center_y = center[1] - y1

        # Select pixels inside the circular area of the ball
        Y, X = np.ogrid[:roi_h, :roi_w]
        dist_from_center = np.sqrt((X - roi_center_x) ** 2 + (Y - roi_center_y) ** 2)
        mask_circle = dist_from_center <= radius
        white_pixels = roi[mask_circle]

        if white_pixels.size == 0:
            raise Exception(f"\033[31m[NP 3/3!!!]\033[0m Normalization failed")

        # Reference RGB color from the selected percentile of ball pixels
        ref_r = np.percentile(white_pixels[:, 0], percentile_ref)
        ref_g = np.percentile(white_pixels[:, 1], percentile_ref)
        ref_b = np.percentile(white_pixels[:, 2], percentile_ref)

        # Minimum reference threshold
        min_ref_threshold = 10
        if ref_r < min_ref_threshold or ref_g < min_ref_threshold or ref_b < min_ref_threshold:
            raise Exception(f"\033[31m[NP 3/3!!!]\033[0m Normalization failed")

        # Target white value
        target_white = 255.0

        # Raw gain factors for each channel
        r_factor_raw = target_white / ref_r
        g_factor_raw = target_white / ref_g
        b_factor_raw = target_white / ref_b

        # Limit the gain to prevent overexposure
        r_factor = min(r_factor_raw, max_gain_factor)
        g_factor = min(g_factor_raw, max_gain_factor)
        b_factor = min(b_factor_raw, max_gain_factor)

        # Apply gain factors to each channel
        normalized_data = img_data.astype(np.float32)
        normalized_data[:, :, 0] *= r_factor
        normalized_data[:, :, 1] *= g_factor
        normalized_data[:, :, 2] *= b_factor

        # Clip values to valid color range
        np.clip(normalized_data, 0, 255, out=normalized_data)
        normalized_data = normalized_data.astype(np.uint8)

        # Convert back from RGB to BGR (OpenCV default)
        normalized_bgr = cv.cvtColor(normalized_data, cv.COLOR_RGB2BGR)

        print(f"\033[32m[NP 3/3]\033[0m Image colors normalized")
        return normalized_bgr


#### FIND WHITE ####
# This function detects a white object to be used as reference for normalization
# img - input image

def find_white(img):
    # Convert to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # White color range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 150, 255])

    # Create mask
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Morphological operations to clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Search for white object (usually the white ball)
    center = None
    radius = None
    best_contour = None
    max_radius = 0

    min_radius_threshold = 5
    max_radius_threshold = 100

    for c in contours:
        (x, y), r = cv.minEnclosingCircle(c)
        if min_radius_threshold < r < max_radius_threshold:
            if r > max_radius:
                max_radius = r
                best_contour = c

    if best_contour is not None:
        (x, y), r = cv.minEnclosingCircle(best_contour)
        center = (int(x), int(y))
        radius = int(r)

    # Check if a valid white object was found
    if center is None:
        raise Exception(f"\033[31m[NP 2/3!!!]\033[0m White object not detected")

    print(f"\033[32m[NP 2/3]\033[0m White object detected")
    return center, radius
