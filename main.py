from poollib.Table import Table


def main():
    ### ------------ INIT ------------ ###
    # Initialize Table object with path to image
    t1 = Table("./samples/photo04.jpg")

    ### ------------ PREPROCESSING ------------ ###
    # Transform image to top-down view (only for raw photos)
    # t1.transform()
    # Normalize colors (only for raw photos)
    # t1.normalize()

    ### ------------ BALL DETECTION ------------ ###
    # Detect balls and store them in Table object
    t1.detect()
    # Categorize balls by color/type (currently only colors work well)
    t1.categorize_balls()
    # Optionally, print detected balls
    # t1.print_balls()

    ### ------------ SHOT CALCULATION ------------ ###
    # Calculate all shots (type argument: "all", "stripe", "solid", "black")
    t1.calculate_shots()
    t1.visualize()
    t1.save("./shots/all_shots.jpg")

    # Validate and remove impossible shots
    t1.validate_shots()
    t1.visualize()
    t1.save("./shots/valid_shots.jpg")

    # Determine the best shot(s)
    t1.calculate_best_shots()
    t1.visualize()
    t1.save("./shots/best_shots.jpg")

    # Optionally, print detected shots
    # t1.print_shots()

    ### ------------ DISPLAY / SAVE RESULT ------------ ###
    # Display the current state of the table
    # t1.show()
    # Save the current image to specified path
    # t1.save("after.jpg")


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(error)
