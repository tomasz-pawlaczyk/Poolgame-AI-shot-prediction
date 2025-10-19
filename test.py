PICTURES_DIR = "01 - policzone przed filtrami"
CSV_SHEET = "coole.csv"

import os
import csv
import shutil
from datetime import datetime
from poollib.Table import Table

ok = 0
output_folder_name = datetime.now().strftime("%m%d_%H%M%S")


def compare_output(expected_balls, photo_name):
    """
    Compare detected balls with expected count for a single photo.
    Copy failing photos to output folder.
    """
    global ok

    full_path = os.path.join(PICTURES_DIR, photo_name)
    t1 = Table(full_path)
    t1.detect()
    detected_count = t1.get_balls_count()

    if int(detected_count) != expected_balls:
        print(f"FAIL| {photo_name}")
        shutil.copy(full_path, os.path.join(output_folder_name, photo_name))
    else:
        print(f"OK  | {photo_name}")
        ok += 1

    return int(detected_count) == expected_balls


def main():
    # Read CSV with expected results
    with open(CSV_SHEET, newline='', encoding='utf-8') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=';', quotechar='"'))

    # Create output folder
    os.mkdir(output_folder_name)

    # Compare each row
    for i, row in enumerate(rows):
        if i == 0:
            # Add new header column
            row.append("Script output")
            continue
        expected_balls = int(row[1])
        photo_name = row[0]
        row.append(compare_output(expected_balls, photo_name))

    # Save updated CSV in output folder
    output_csv_path = os.path.join(output_folder_name, CSV_SHEET)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as new_csv:
        writer = csv.writer(new_csv, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)

    print(f"Passed: {ok}/{len(rows) - 1}")


if __name__ == '__main__':
    main()
