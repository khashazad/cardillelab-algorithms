from lib.image_collections import build_collection
from lib.study_areas import PNW, RANDONIA
import ee
from lib.observations_points import parse_point_coordinates
from lib.constants import Index, Initialization, Sensor

ee.Initialize()


def get_tag(
    **kwargs,
):
    year_tag = (
        kwargs["years"][0]
        if len(kwargs["years"]) == 1
        else f"{kwargs['years'][0]}-{kwargs['years'][-1]}"
    )
    return f"{kwargs['study_area']['name']}_{'_'.join([s.value for s in kwargs['sensors']])}_{kwargs['index'].value}_{year_tag}_{kwargs['point_group']}"


def get_collection(**kwargs):
    return build_collection(
        kwargs["study_area"]["coords"],
        kwargs["years"],
        kwargs["index"],
        kwargs["sensors"],
        kwargs["day_step_size"],
        kwargs["start_doy"],
        kwargs["end_doy"],
        kwargs["cloud_cover_threshold"],
    )


def get_points(point_group):
    return parse_point_coordinates(point_group)


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
    initialization=Initialization.POSTHOC,
):
    tag = get_tag(
        study_area,
        index,
        sensors,
        years,
        point_group,
        day_step_size,
        start_doy,
        end_doy,
        cloud_cover_threshold,
        initialization,
    )

    return {
        "tag": tag,
        "study_area": study_area,
        "index": index,
        "sensors": sensors,
        "years": years,
        "point_group": point_group,
        "day_step_size": day_step_size,
        "start_doy": start_doy,
        "end_doy": end_doy,
        "cloud_cover_threshold": cloud_cover_threshold,
        "initialization": initialization,
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
        "initialization": initialization,
    }
