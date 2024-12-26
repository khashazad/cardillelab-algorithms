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
    return f"{'_'.join([s.value for s in kwargs['sensors']])}_{kwargs['index'].value}_{year_tag}_{kwargs['point_group']}"


def get_collection(**kwargs):
    return build_collection(
        kwargs["study_area"],
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
