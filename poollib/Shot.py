from .shot_init import get_ghost, get_cut_angle
import math


class Shot:
    def __init__(self, white, target, hole, edge_target=None, edge_white=None):
        # white, target coordinates, hole position, and optional cushions for reflections

        # Path of target ball to hole: one line if direct, two lines if reflecting off cushion
        if edge_target:
            self.__target_dir = {
                "y1": (target.get_coordinates(), (int(edge_target["x"]), int(edge_target["y"]))),
                "y2": ((int(edge_target["x"]), int(edge_target["y"])), (int(hole["x"]), int(hole["y"])))
            }
            self.__y1length = (math.dist(self.__target_dir["y1"][0], self.__target_dir["y1"][1]) +
                               math.dist(self.__target_dir["y2"][0], self.__target_dir["y2"][1]))
        else:
            self.__target_dir = {"y1": (target.get_coordinates(), (int(hole["x"]), int(hole["y"])))}
            self.__y1length = math.dist(self.__target_dir["y1"][0], self.__target_dir["y1"][1])

        # Ghost ball position
        self.__ghost = get_ghost(self.__target_dir["y1"])

        # Path from white ball to ghost ball: one line if direct, two lines if reflecting off cushion
        if edge_white:
            self.__white_dir = {
                "y1": (white.get_coordinates(), (int(edge_white["x"]), int(edge_white["y"]))),
                "y2": ((int(edge_white["x"]), int(edge_white["y"])), (int(self.__ghost["x"]), int(self.__ghost["y"])))
            }
            self.__y2length = (math.dist(self.__white_dir["y1"][0], self.__white_dir["y1"][1]) +
                               math.dist(self.__white_dir["y2"][0], self.__white_dir["y2"][1]))
        else:
            self.__white_dir = {"y1": (white.get_coordinates(), (int(self.__ghost["x"]), int(self.__ghost["y"])))}
            self.__y2length = math.dist(self.__white_dir["y1"][0], self.__white_dir["y1"][1])

        # Angle between reflection vectors
        self.__angle = get_cut_angle(self.__white_dir, self.__target_dir)

        # Total shot length
        self.__length = self.__y1length + self.__y2length

    def get_lines(self):
        return self.__target_dir, self.__white_dir

    def get_ghost(self):
        return int(self.__ghost["x"]), int(self.__ghost["y"])

    def get_angle(self):
        return self.__angle

    def get_length(self):
        return self.__length
