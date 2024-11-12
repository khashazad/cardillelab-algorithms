from lib.image_collections import build_collection
from lib.study_areas import PNW, RANDONIA
import ee
from lib.observations_points import parse_point_coordinates
from lib.constants import Index, Sensor

ee.Initialize()


def study_package(
    study_area,
    index,
    sensors,
    years,
    point_group,
    day_step_size,
    start_doy,
    end_doy,
    cloud_cover_threshold,
):
    return {
        "tag": f"{study_area['name']}_{index.value}_{years[0]}_{years[1]}",
        "index": index,
        "points": parse_point_coordinates(point_group),
        "study_area": study_area,
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


def pnw_swir_2022_2023_1_points():
    index = Index.SWIR
    sensors = [Sensor.L8, Sensor.L9]
    years = [2022, 2023]
    point_group = "pnw_1"
    study_area = PNW
    day_step_size = 6
    start_doy = 150
    end_doy = 250
    cloud_cover_threshold = 20

    return study_package(
        study_area,
        index,
        sensors,
        years,
        point_group,
        day_step_size,
        start_doy,
        end_doy,
        cloud_cover_threshold,
    )


def pnw_swir_2017_2018_1_points():
    index = Index.SWIR
    sensors = [Sensor.L7, Sensor.L8, Sensor.L9]
    years = [2017, 2018]
    point_group = "pnw_1"
    study_area = PNW
    day_step_size = 6
    start_doy = 150
    end_doy = 250
    cloud_cover_threshold = 20

    return study_package(
        study_area,
        index,
        sensors,
        years,
        point_group,
        day_step_size,
        start_doy,
        end_doy,
        cloud_cover_threshold,
    )


def pnw_nbr_2017_2019_1_point():
    index = Index.NBR
    sensors = [Sensor.L7, Sensor.L8, Sensor.L9]
    years = [2017, 2018, 2019]
    point_group = "pnw_1"
    study_area = PNW
    day_step_size = 10
    start_doy = 150
    end_doy = 250
    cloud_cover_threshold = 20

    return study_package(
        study_area,
        index,
        sensors,
        years,
        point_group,
        day_step_size,
        start_doy,
        end_doy,
        cloud_cover_threshold,
    )


def pnw_nbr_2017_2018_1_point():
    index = Index.NBR
    sensors = [Sensor.L7, Sensor.L8, Sensor.L9]
    years = [2017, 2018]
    point_group = "pnw_1"
    study_area = PNW
    day_step_size = 10
    start_doy = 150
    end_doy = 250
    cloud_cover_threshold = 20

    return study_package(
        study_area,
        index,
        sensors,
        years,
        point_group,
        day_step_size,
        start_doy,
        end_doy,
        cloud_cover_threshold,
    )
