import json
import os
from lib.constants import Harmonic
from lib.image_collections import build_collection
from lib.observations_points import parse_point_coordinates

class KalmanRunConfig:
    def __init__(self, index, sensors, years, point_group, study_area, day_step_size, start_doy, end_doy, cloud_cover_threshold, initialization,  harmonic_flags: dict, recording_flags: dict):
        self.index = index
        self.sensors = sensors
        self.years = years
        self.point_group = point_group
        self.study_area = study_area
        self.day_step_size = day_step_size
        self.start_doy = start_doy
        self.end_doy = end_doy
        self.cloud_cover_threshold = cloud_cover_threshold
        self.initialization = initialization
        self.harmonic_flags = harmonic_flags
        self.recording_flags = recording_flags

    def get_kalman_parameters(self):
        include_slope = self.harmonic_flags.get(Harmonic.SLOPE.value, False)
        bimodal = self.harmonic_flags.get(Harmonic.BIMODAL.value, False)
        trimodal = self.harmonic_flags.get(Harmonic.TRIMODAL.value, False)

        postfix = ""
        if include_slope:
            postfix += "_slope"
        if trimodal:
            postfix += "_trimodal"
        elif bimodal:
            postfix += "_bimodal"
        else:
            postfix += "_unimodal"

        return json.load(open(f"{os.path.join(os.path.dirname(os.path.realpath(__file__)), "parameters")}/kalman_parameters{postfix}.json"))
    
    def get_points(self):
        return parse_point_coordinates(self.point_group)
    
    def get_tag(self):
        year_tag = (
            self.years[0]
            if len(self.years) == 1
            else f"{self.years[0]}-{self.years[-1]}"
        )
        return "_".join(
            [
            self.study_area['name'],
            '_'.join([s.value for s in self.sensors]),
            self.index.value,
            str(year_tag),
            self.point_group
            ])

    def get_collection(self):
        return build_collection(
            self.study_area["coords"],
            self.years,
            self.index,
            self.sensors,
            self.day_step_size,
            self.start_doy,
            self.end_doy,
            self.cloud_cover_threshold,
        )
    
    def write_to_file(self, file_path):
        
        points = self.get_points()

        points = [{"index": i, "point": p} for i, p in enumerate(points)]

        with open(file_path, "w") as file:
            json.dump({
                "sensors": [s.value for s in self.sensors],
                "years": list(self.years) if isinstance(self.years, range) else self.years,
                "point_group": self.point_group,
                "points": points,
                "study_area": self.study_area,
                "day_step_size": self.day_step_size,
                "start_doy": self.start_doy,
                "end_doy": self.end_doy,
                "cloud_cover_threshold": self.cloud_cover_threshold,
                "initialization": self.initialization.value,
                "harmonic_flags": self.harmonic_flags,
                "recording_flags": {k.value: v for k, v in self.recording_flags.items()},
            }, file)



