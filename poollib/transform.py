import cv2 as cv
import numpy as np
import math
from PIL import Image
import pillow_heif

#### Transform ####
# img      - input image (required)
# res_w    - output width (default 1000)
# res_h    - output height (default 2000)
# returns the transformed image in BGR format ready for analysis
def transform(img, res_w=1000, res_h=2000):
    print("-----[Transform Photo]-----")

    # Check if image is valid
    if img is None:
        raise Exception("[TRANSFORM] Invalid image")
    print("[TRANSFORM] Image loaded")

    height, width = img.shape[:2]

    # Rotate image if it is landscape
    if width > height:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        height, width = width, height

    # Convert to grayscale and extract edges
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    edges = cv.Canny(blurred, 170, 200)

    # Detect circles (pockets) using Hough Transform
    circles = cv.HoughCircles(edges,
                              cv.HOUGH_GRADIENT,
                              dp=1.6,
                              minDist=500,
                              param1=30,
                              param2=30,
                              minRadius=70,
                              maxRadius=100)

    if circles is None:
        raise Exception("[TRANSFORM] No pockets detected, check image")

    print("[TRANSFORM] Pockets detected")
    circles = np.uint16(np.around(circles))[0]

    # Initialize variables to store circles closest to corners
    tlCirc = {"circle": [0, 0, 0], "dist": 1e6}  # top-left
    trCirc = {"circle": [0, 0, 0], "dist": 1e6}  # top-right
    blCirc = {"circle": [0, 0, 0], "dist": 1e6}  # bottom-left
    brCirc = {"circle": [0, 0, 0], "dist": 1e6}  # bottom-right

    # Find circles closest to each corner
    for (x, y, r) in circles:
        # Top-left
        dist = math.hypot(x, y)
        if dist < tlCirc["dist"]:
            tlCirc["dist"] = dist
            tlCirc["circle"] = [x, y, r]
        # Top-right
        dist = math.hypot(width - x, y)
        if dist < trCirc["dist"]:
            trCirc["dist"] = dist
            trCirc["circle"] = [x, y, r]
        # Bottom-left
        dist = math.hypot(x, height - y)
        if dist < blCirc["dist"]:
            blCirc["dist"] = dist
            blCirc["circle"] = [x, y, r]
        # Bottom-right
        dist = math.hypot(width - x, height - y)
        if dist < brCirc["dist"]:
            brCirc["dist"] = dist
            brCirc["circle"] = [x, y, r]

    # Prepare points for perspective transform
    pts1 = np.float32([
        [tlCirc["circle"][0], tlCirc["circle"][1]],
        [trCirc["circle"][0], trCirc["circle"][1]],
        [blCirc["circle"][0], blCirc["circle"][1]],
        [brCirc["circle"][0], brCirc["circle"][1]]
    ])
    pts2 = np.float32([
        [0, 0],
        [res_w, 0],
        [0, res_h],
        [res_w, res_h]
    ])

    # Compute perspective transform and apply
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (res_w, res_h))

    print("[TRANSFORM] Image transformed successfully")
    return dst


#### HEIC to OpenCV ####
# Converts a HEIC image to a format usable by OpenCV
def heic2opencv(input_path):
    # Open HEIC file
    heif_image = pillow_heif.open_heif(input_path)

    # Convert to PIL image
    image = Image.frombytes(
        heif_image.mode,
        heif_image.size,
        heif_image.data,
        "raw"
    )

    # Convert to NumPy array (RGB)
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

    return image_bgr
