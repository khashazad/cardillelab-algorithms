import os

script_directory = os.path.dirname(os.path.realpath(__file__))

for folder in os.listdir(script_directory):
    total_count = 0
    if os.path.isdir(os.path.join(script_directory, folder)):
        for dir in os.listdir(os.path.join(script_directory, folder)):

            points_count = len(os.listdir(os.path.join(script_directory, folder, dir)))
            total_count += points_count

            prefix = dir.split(" - ")[0]
            os.rename(os.path.join(script_directory, folder, dir), os.path.join(script_directory, folder, f"{prefix} - {points_count} points"))

        prefix = folder.split(" - ")[0]
        os.rename(os.path.join(script_directory, folder), os.path.join(script_directory, f"{prefix} - {total_count} points"))