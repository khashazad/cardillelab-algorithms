import os
from eeek.gather_collections import reduce_collection_to_points_and_write_to_file
from eeek.image_collections import COLLECTIONS
from eeek.prepare_optimization_run import (
    build_observations,
    fitted_coefficients_and_dates,
    parse_point_coordinates,
    create_points_file,
)

POINT_SET = 5
RUN_VERSION = 1

root = os.path.dirname(os.path.abspath(__file__))

run_directory = os.path.join(
    root, "torch runs", f"point set {POINT_SET}", f"v{RUN_VERSION}"
)

os.makedirs(run_directory, exist_ok=True)

if __name__ == "__main__":
    points = parse_point_coordinates(f"{root}/points/sets/{POINT_SET}")

    reduce_collection_to_points_and_write_to_file(
        COLLECTIONS["L8_L9_2022_2023"], points, f"{run_directory}/measurements.csv"
    )

    fitted_coefficiets_by_point = fitted_coefficients_and_dates(
        points, f"{run_directory}/fitted_coefficients.csv"
    )

    create_points_file(f"{run_directory}/points.csv", fitted_coefficiets_by_point)

    observations = build_observations(
        fitted_coefficiets_by_point, f"{run_directory}/observations.csv"
    )
