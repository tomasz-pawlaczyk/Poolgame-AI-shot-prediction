import cv2 as cv
import numpy as np
import tkinter as tk

from .transform import transform, heic2opencv
from .normalize import normalize
from .detect import detect_balls
from .visualize import visualize
from .Ball import Ball
from .shots_calculations import get_shots, is_shot_blocked, get_best_shots, is_edge_possible

# Table class loads and stores an image of the table
# Methods allow processing for analysis of balls and possible shots
class Table:
    def __init__(self, input_file):
        # Load image (HEIC or JPG)
        ext = input_file.split(".")[-1].lower()
        if ext == "heic":
            self.__img = heic2opencv(input_file)
        elif ext == "jpg":
            self.__img = cv.imread(input_file)
        else:
            raise Exception(f"[TABLE INIT] Unsupported image format")

        if self.__img is None:
            raise Exception(f"[TABLE INIT] Failed to load image")

        self.__name = input_file.split("/")[-1]
        self.__height, self.__width = self.__img.shape[:2]

        # Detected balls
        self.__balls = []

        # Calculated shots
        self.__shots = []

    def show(self):
        if self.__img is not None:
            root = tk.Tk()
            scale = int(0.75 * root.winfo_screenheight())
            root.destroy()
            self.__img = cv.resize(self.__img, [scale // 2, scale])
            cv.imshow("Table", self.__img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def save(self, path="./default.jpg"):
        if self.__img is not None:
            cv.imwrite(path, self.__img)

    def transform(self):
        # Transform table image (perspective correction)
        self.__img = transform(self.__img)
        self.__height, self.__width = self.__img.shape[:2]

    def normalize(self):
        # Normalize image (lighting, colors)
        self.__img = normalize(self.__img)

    def detect(self):
        # Detect balls on the table
        circles, self.__img = detect_balls(self.__img)
        if circles is None:
            raise Exception("[DETECT] No balls detected")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            self.__balls.append(Ball(x, y))
        print("[DETECT] Balls added")

    def print_balls(self):
        print("-----[Balls]-----")
        for i, ball in enumerate(self.__balls, start=1):
            print(f"Ball {i}: {ball.get_coordinates()}, color: {ball.get_color()}, type: {ball.get_type()}")

    def categorize_balls(self):
        # Determine type/color of each ball
        print("-----[Categorize Balls]-----")
        if self.__balls:
            print("[CATEGORIZE] Categorizing balls", end='')
            for ball in self.__balls:
                ball.categorize(self.__img)
                print(".", end='')
            print("\n[CATEGORIZE] Balls categorized")
        else:
            print("[CATEGORIZE] No balls to categorize")

    def visualize(self):
        # Visualize table with balls and shots
        self.__img = visualize(self.__img, self.__balls, self.__shots)

    def calculate_shots(self, type="all"):
        # Calculate all possible shots for balls of given type
        print("-----[Calculate Shots]-----")
        if not self.__balls:
            print("[SHOTS] No balls detected")
            return

        white = next((b for b in self.__balls if b.get_type() == "white"), None)
        if not white:
            print("[SHOTS] White ball not found")
            return
        print("[SHOTS] White ball found")

        print(f"[SHOTS] Calculating shots for {type}", end='')
        for ball in self.__balls:
            if ball.get_type() != "white" and (type == "all" or ball.get_type() == type):
                self.__shots.extend(get_shots(white, ball))
                print(".", end='')
        print("\n[SHOTS] Shots calculated")

    def validate_shots(self):
        # Filter valid shots based on angles and obstacles
        print("-----[Validate Shots]-----")
        valid_shots = []

        if not self.__shots:
            print("[VALIDATE] No shots detected")
            return

        print("[VALIDATE] Validating shots", end='')
        for shot in self.__shots:
            print(".", end='')
            t_dir, w_dir = shot.get_lines()
            lines = [
                t_dir.get("y1"),
                w_dir.get("y1"),
                t_dir.get("y2") if len(t_dir) > 1 else None,
                w_dir.get("y2") if len(w_dir) > 1 else None
            ]
            angle = shot.get_angle()
            blocked = [is_shot_blocked(line, self.__balls, 23) for line in lines if line]
            edges_ok = [is_edge_possible(line) for line in lines if line]

            if not any(blocked) and angle > 120 and all(edges_ok):
                valid_shots.append(shot)

        self.__shots = valid_shots
        print("\n[VALIDATE] Valid shots selected")

    def calculate_best_shots(self):
        # Select best recommended shots
        print("-----[Best Shots]-----")
        if not self.__shots:
            print("[BEST] No shots detected")
            return

        print("[BEST] Calculating best recommendations", end='')
        self.__shots = get_best_shots(self.__shots)
        print("\n[BEST] Recommended shots selected")

    def print_shots(self):
        print("-----[Shots]-----")
        for i, shot in enumerate(self.__shots, start=1):
            print(f"Shot {i}: length: {round(shot.get_length(),2)}, angle: {shot.get_angle()}")

    def get_balls_count(self):
        return len(self.__balls)
