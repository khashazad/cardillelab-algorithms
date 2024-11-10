import os
import json

PREFIX = os.path.join(os.path.dirname(__file__), "point_groups", "groups")


def build_path(file_name):
    return os.path.join(PREFIX, file_name)


def rename_point_group_folder(point_set_directory_path):
    for folder in os.listdir(point_set_directory_path):
        total_count = 0
        if os.path.isdir(os.path.join(point_set_directory_path, folder)):
            for dir in os.listdir(os.path.join(point_set_directory_path, folder)):

                points_count = len(
                    os.listdir(os.path.join(point_set_directory_path, folder, dir))
                )
                total_count += points_count

                prefix = dir.split(" - ")[0]
                os.rename(
                    os.path.join(point_set_directory_path, folder, dir),
                    os.path.join(
                        point_set_directory_path,
                        folder,
                        f"{prefix} - {points_count} points",
                    ),
                )


def parse_point_coordinates(point_group):
    point_set_directory_path = build_path(point_group)
    point_coordinates = []

    if not os.path.exists(point_set_directory_path):
        raise ValueError(f"Invalid point group directory: {point_set_directory_path}")

    def process_json_file(content_path):
        with open(content_path, "r") as file:
            point_coordinates.extend(json.load(file).get("points", []))

    for folder in os.listdir(point_set_directory_path):
        content_path = os.path.join(point_set_directory_path, folder)

        if os.path.isdir(content_path):
            for file in os.listdir(content_path):
                file_path = os.path.join(content_path, file)
                if file_path.endswith(".json"):
                    process_json_file(file_path)
                else:
                    point_coordinates.append(
                        (
                            float(file.split(",")[0][1:]),
                            float(file.split(",")[1][:-5]),
                        )
                    )
        elif content_path.endswith(".json"):
            process_json_file(content_path)

    point_count = len(point_coordinates)
    with open("points_count.txt", "w") as file:
        file.write(f"Total points: {point_count}")

    return sorted(point_coordinates, key=lambda x: (x[0], x[1]))


# Paths to study point directories under "point_groups"

PNW_1 = build_path("pnw_1")
PNW_6 = build_path("pnw_6")
PNW_10 = build_path("pnw_10")
PNW_15 = build_path("pnw_15")
PNW_20 = build_path("pnw_20")
PNW_30 = build_path("pnw_30")
PNW_40 = build_path("pnw_40")
RANDONIA_4 = build_path("randonia_4")
RANDONIA_11 = build_path("randonia_11")
RANDONIA_MIX = build_path("randonia_mix")

STUDY_POINT_GROUPS = {
    "pnw_1": PNW_1,
    "pnw_6": PNW_6,
    "pnw_10": PNW_10,
    "pnw_15": PNW_15,
    "pnw_20": PNW_20,
    "pnw_30": PNW_30,
    "pnw_40": PNW_40,
    "randonia_4": RANDONIA_4,
    "randonia_11": RANDONIA_11,
    "randonia_mix": RANDONIA_MIX,
}
