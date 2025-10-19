from .categorize import categorize

# The Ball class stores data for a single billiard ball as an object

class Ball:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y
        self.__type = None
        self.__color = None
        self.__pocketed = None
    
    def get_coordinates(self):
        return (int(self.__x),int(self.__y))
    
    def get_color(self):
        return self.__color
    
    def get_type(self):
        return self.__type
    
    def categorize(self, img):
        self.__type, self.__color = categorize(self.__x, self.__y, img)