from lib.image_collections import build_collection
from lib.study_areas import PNW, RANDONIA
import ee
from lib.observations_points import parse_point_coordinates
from lib.constants import Index, Sensor

ee.Initialize()


def build_pnw_swir_2022_2023_5_point():
    index = Index.SWIR
    sensors = [Sensor.L8, Sensor.L9]
    years = [2022, 2023]
    point_group = "pnw_6"
    study_area = PNW
    day_step_size = 6
    start_doy = 150
    end_doy = 250
    cloud_cover_threshold = 20

    return {
        "tag": f"{study_area['name']}_{index.value}_{years[0]}_{years[1]}",
        "index": index,
        "points": parse_point_coordinates(point_group),
        "collection": build_collection(
            study_area["coords"],
            years,
            index,
            sensors,
            day_step_size,
            start_doy,
            end_doy,
            cloud_cover_threshold,
        ),
        "years": years,
    }


PNW_SWIR_2022_2023_5_POINT = build_pnw_swir_2022_2023_5_point()
