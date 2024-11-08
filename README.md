# Earth Engine Algorithms

This repository contains two main algorithms for the Earth Engine platform:

1. EKF: Extended Kalman Filter
2. BULCD: Bayesian Updating of Land Cover Classification

### Installation and Prerequisites

1. Clone repository
2. Create new virtual environment: `python -m venv {path/to/environment}`
3. Activate environment: `source {path/to/environemnt}/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Earth Engine Extended Kalman Filter

This algorithm implements an Extended Kalman Filter for the purpose of keeping continual track of expected values of satellite imagery.

### Preparing New Test Run

#### 1. Create a new study group

This can be achieved by:

1. Create a new folder in the `lib` > `point_groups` > `groups` > `{new_group_name}` directory
2. Add a **json** file in the new directory with the following structure:

   ```json
   {
       "points": [
           [lon, lat],
           [lon, lat],
           ...
       ]
   }
   ```

   Where `lon` and `lat` are the coordinates of the point of interest.

3. Register the new group by adding the new path to the `STUDY_POINT_GROUPS` dictionary in `lib/observations_points.py`:

   ```python
   NEW_GROUP_PATH = build_path("new_group_name")

   STUDY_POINT_GROUPS = {
       ...
       "new_group_tag": NEW_GROUP_PATH,
   }
   ```

4. Register the study area by adding the boundary coordinates to a new variable in `lib/study_areas.py`:

   ```python
   NEW_STUDY_AREA_TAG = [
       (lon, lat),
       (lon, lat),
       (lon, lat),
       (lon, lat),
   ],
   ```

5. Register the image collection by adding the new collection to the `COLLECTIONS` dictionary in `lib/image_collections.py`:

   ```python
   IMAGE_COLLECTION = gather_collections_and_reduce(
        {
            "L8dictionary": {
                "years_list": [2017, 2018],
                "first_doy": 150,
                "last_doy": 250,
                "cloud_cover_threshold": 20,
            },
            "L9dictionary": {
                "years_list": [2017, 2018],
                "first_doy": 150,
                "last_doy": 250,
                "cloud_cover_threshold": 20,
            },
            "default_study_area": (ee.Geometry.Polygon(NEW_STUDY_AREA_TAG)), # study area tag created above
            "band_name_reduction": "swir",
            "which_reduction": "SWIR",
            "day_step_size": 6,
            "verbose": False,
            "dataset_selection": {
                "L5": False,
                "L7": False,
                "L8": True,
                "L9": True,
                "MO": False,
                "S2": False,
                "S1": False,
                "DW": False,
            },
            "first_expectation_year": 2017,
            "verbose": False,
        }
    )

   COLLECTIONS = {
       ...
       "new_collection_tag": IMAGE_COLLECTION,
   }
   ```

### Running the algorithm

Open the `kalman_caller.py` file and modify:

- `STUDY_GROUP_TAG` variable to the tag of the new study group created above.
- `COLLECTION_TAG` variable to the tag of the new image collection created above.

Then run the script.

### Viewing Results

The results will be saved in the `tests/kalman/{STUDY_GROUP_TAG}.../{datetime/` directory.
