from pydantic import BaseModel
from typing import Tuple, Any, Dict, Optional
from .utils import Rosetta
from copy import deepcopy


class Style(BaseModel):
    """Miscellaneous properties; set defaults once, then use or inherit elsewhere for consistent style
    (note: some names are formatted to be congruent with matplotlib.pyplot vars & args, rather than isthmuslib)"""

    color: Optional[Any] = "DarkGreen"  # isthmuslib default
    facecolor: Optional[Any] = "w"
    title_fontsize: Optional[Any] = 18.0
    label_fontsize: Optional[Any] = 15.0
    legend_fontsize: Optional[Any] = 15.0
    tick_fontsize: Optional[Any] = 12.0
    figsize: Optional[Tuple[Any, Any]] = (12.0, 8.0)
    linewidth: Optional[Any] = 5.0
    linestyle: Optional[Any] = "-"
    grid: Optional[bool] = False
    tight_axes: Optional[bool] = True
    formatter: Optional[str] = "%Y-%m-%d %H:%M:%S"
    include_timezone: Optional[bool] = False
    markersize: Optional[Any] = 50
    transparent_alpha: Optional[float] = 0.5
    watermark_placement: Optional[Any] = (0.05, 0.05)
    watermark_fontsize: Optional[Any] = 20
    watermark_color: Optional[Any] = "dimgrey"
    watermark_text: Optional[str] = ""
    histogram_bins: Optional[Any] = 25
    multi_hist_alpha: Optional[float] = 0.5
    rosetta: Optional[Rosetta] = Rosetta()
    median_linestyle: Optional[Any] = "--"
    median_linewidth: Optional[Any] = 3.0
    median_linecolor: Optional[Any] = "k"
    mean_linestyle: Optional[Any] = "-"
    mean_linewidth: Optional[Any] = 3.0
    mean_linecolor: Optional[Any] = "k"
    cycler: Optional[Any] = None
    good_color: Optional[Any] = "navy"
    bad_color: Optional[Any] = "firebrick"
    cmap: Optional[Any] = "inferno"
    sequential_cmap: Optional[Any] = "Greens"
    log_formatter: Optional[
        str
    ] = "\n@@ {time:x} AT: {time} | LEVEL: {level} | IN: {name}.{function}\n\n{message} |\n"
    timeframe_prefix: Optional[str] = "from "
    timeframe_between: Optional[str] = " to "
    timeframe_suffix: Optional[str] = ""
    auto_locf_params: Optional[Dict[str, Any]] = {
        "window": 3,
        "impute_method": "median",
        "impute_direction": "forward",
        "add_noise": False,
    }
    x_axis_human_tick_labels: Optional[bool] = False
    x_axis_formatter: Optional[str] = None
    dict_pretty_max_length: Optional[int] = 32
    dict_pretty_previews: Optional[bool] = False
    axhline_color: Optional[str] = "gray"
    axhline_linestyle: Optional[str] = "--"
    axhline_linewidth: Optional[float] = 1.0
    axvline_color: Optional[str] = "gray"
    axvline_linestyle: Optional[str] = "--"
    axvline_linewidth: Optional[float] = 1.0

    def translate(self, key: str, **kwargs) -> str:
        """Helper function that allows Style objects to translate text according to the provided Rosetta mappings

        :param key: the string to be translated
        :param kwargs: keyword arguments for the rosetta translate() method
        :return: translated string
        """
        if self.rosetta:
            return self.rosetta.translate(key, **kwargs)
        else:
            return key

    def override(self, *args):
        """Helper function that overrides specific style features

        :return: returns a fresh copy of a Style object with the updated values
        """

        if not args:
            return deepcopy(self)

        elif len(args) == 1:
            if args[0] is None:
                return deepcopy(self)
            elif isinstance(args[0], dict):
                override_dict: Dict[str, any] = args[0]
            elif getattr(args[0], "dict", None):
                override_dict: Dict[str, any] = args[0].dict()
            else:
                raise ValueError(f"Unsure how to interpret override input of type {type(args[0])}")

        elif (len(args) == 2) and isinstance(args[0], str):
            override_dict: Dict[str, any] = {args[0]: args[1]}

        else:
            raise ValueError(f"Unsure how to interpret 3+ inputs to override")

        return self.__class__(**{**self.dict(), **override_dict})
