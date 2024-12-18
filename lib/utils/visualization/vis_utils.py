import json
import ee
import ipywidgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd

from lib.constants import DATE_LABEL
from lib.utils.visualization.constant import COLOR_PALETTE_10

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


class CoefInspector(ipywidgets.VBox):
    def __init__(self, host_map, output, coef_layers=None):
        self._host_map = host_map
        self._output = output

        self.coef_layers = coef_layers
        if self.coef_layers is not None and not isinstance(
            self.coef_layers, (tuple, list)
        ):
            self.coef_layers = [self.coef_layers]

        if not host_map:
            raise ValueError("Must pass a valid map when creating coef inspector")

        host_map.default_style = {"cursor": "crosshair"}
        self._clear_inspector_output()

        host_map.on_interaction(self._on_map_interaction)

        super().__init__()

    def _clear_inspector_output(self, wait=False):
        self._output.clear_output(wait=wait)

    def _on_map_interaction(self, **kwargs):
        latlon = kwargs.get("coordinates")
        if kwargs.get("type") == "click":
            self._on_map_click(latlon)

    def _get_visible_map_layers(self):
        layers = self._host_map.ee_layers
        return {k: v for k, v in layers.items() if v["ee_layer"].visible}

    def _query_point(self, latlon, ee_object):
        point = ee.Geometry.Point(latlon[::-1])
        scale = self._host_map.get_scale()
        if isinstance(ee_object, ee.ImageCollection):
            ee_object = ee_object.mosaic()
        if isinstance(ee_object, ee.Image):
            return ee_object.reduceRegion(ee.Reducer.first(), point, scale).getInfo()
        return None

    def _build_func(self, coefs):
        def f(t):
            intercept = coefs[0]
            slope = coefs[1] * t
            mode1 = (coefs[2] * np.cos(2 * np.pi * t)) + (
                coefs[3] * np.sin(2 * np.pi * t)
            )
            mode2 = (coefs[4] * np.cos(4 * np.pi * t)) + (
                coefs[5] * np.sin(4 * np.pi * t)
            )
            mode3 = (coefs[6] * np.cos(6 * np.pi * t)) + (
                coefs[7] * np.sin(6 * np.pi * t)
            )
            return intercept + slope + mode1 + mode2 + mode3

        return f

    def _pixels_info(self, latlon):
        layers = self._get_visible_map_layers()
        functions = {}
        pixel_vals = {}
        for layer_name, layer in layers.items():
            if self.coef_layers is not None and layer_name not in self.coef_layers:
                continue
            ee_object = layer["ee_object"]
            pixel = self._query_point(latlon, ee_object)
            vals = list(pixel.values())
            functions[layer_name] = self._build_func(vals)
            pixel_vals[layer_name] = vals

        return functions, pixel_vals

    def plot(self, funcs, pixels, latlon):
        fig, ax = plt.subplots(1)
        x = np.linspace(0, 5, 5000)
        for i, (layer_name, func) in enumerate(funcs.items()):
            y = func(x)
            ax.plot(x, y, label=layer_name)
            if i == 0:
                min_val = np.min(y)
                if min_val < 0:
                    min_val = 1.4 * min_val
                else:
                    min_val = 0.6 * min_val
                ax.set_ylim(min_val, 1.4 * np.max(y))
        ax.legend()

        with self._output:
            print(latlon)
            display(fig.figure)
            for layer_name, pixel in pixels.items():
                print(layer_name, pixel)

    def _on_map_click(self, latlon):
        self._host_map.default_style = {"cursor": "wait"}
        self._clear_inspector_output(True)

        funcs, pixels = self._pixels_info(latlon)

        self.plot(funcs, pixels, latlon)
        self._host_map.default_style = {"cursor": "crosshair"}


def plot_ccdc_segments(axs, data, segments_file_path):

    def year_fraction_to_datetime(year_fraction):
        year = np.floor(year_fraction).astype(int)
        fraction = year_fraction - year
        days_in_year = (
            366 if pd.Timestamp(year=year, month=1, day=1).is_leap_year else 365
        )
        day_of_year = fraction * days_in_year

        return pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(day_of_year, unit="D")

    segments = json.load(open(segments_file_path))

    first_date = pd.to_datetime(data[DATE_LABEL].min())
    last_date = pd.to_datetime(data[DATE_LABEL].max())

    for idx, (start, end) in enumerate(segments):
        if start != 0 and end != 0:

            start_date = year_fraction_to_datetime(start)
            end_date = year_fraction_to_datetime(end)

            if end_date < first_date:
                continue

            if start_date > last_date:
                continue

            if start_date < first_date:
                start_date = first_date

            if end_date > last_date:
                end_date = last_date

            axs.fill_betweenx(
                axs.get_ylim(),
                start_date,
                end_date,
                color=COLOR_PALETTE_10[idx],
                alpha=0.3,
            )
