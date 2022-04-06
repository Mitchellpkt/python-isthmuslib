from pydantic import BaseModel
from typing import Tuple, Any, Dict
from .utils import Rosetta
from copy import deepcopy


class Style(BaseModel):
    """ Miscellaneous properties; set defaults once, then use or inherit elsewhere for consistent style
        (note: some names are formatted to be congruent with matplotlib.pyplot vars & args, rather than isthmuslib) """

    color: Any = 'DarkGreen'  # isthmuslib default
    facecolor: Any = 'w'
    title_fontsize: Any = 18.0
    label_fontsize: Any = 15.0
    legend_fontsize: Any = 15.0
    tick_fontsize: Any = 12.0
    figsize: Tuple[Any, Any] = (10.0, 8.0)
    linewidth: Any = 5.0
    linestyle: Any = '-'
    grid: bool = True
    tight_axes: bool = True
    formatter: str = '%Y-%m-%d %H:%M:%S'
    markersize: Any = 50
    transparent_alpha: float = 0.5
    watermark_placement: Any = (0.05, 0.05)
    watermark_fontsize: Any = 20
    watermark_color: Any = 'dimgrey'
    watermark_text: str = ''
    histogram_bins: Any = 25
    multi_hist_alpha: float = 0.5
    rosetta: Rosetta = Rosetta()
    median_linestyle: Any = '--'
    median_linewidth: Any = 3.0
    median_linecolor: Any = 'k'
    mean_linestyle: Any = '-'
    mean_linewidth: Any = 3.0
    mean_linecolor: Any = 'k'
    cycler: Any = None
    good_color: Any = 'navy'
    bad_color: Any = 'firebrick'
    cmap: Any = 'inferno'
    sequential_cmap: Any = 'Greens'
    log_formatter: str = "\n@@ {time:x} AT: {time} | LEVEL: {level} | IN: {name}.{function}\n\n{message} |\n"


    def translate(self, key: str, **kwargs) -> str:
        """ Helper function that allows Style objects to translate text according to the provided Rosetta mappings

        :param key: the string to be translated
        :param kwargs: keyword arguments for the rosetta translate() method
        :return: translated string
        """
        if self.rosetta:
            return self.rosetta.translate(key, **kwargs)
        else:
            return key

    def override(self, *args):
        """ Helper function that overrides specific style features

        :return: returns a fresh copy of a Style object with the updated values
        """

        if not args:
            return deepcopy(self)

        elif len(args) == 1:
            if args[0] is None:
                return deepcopy(self)
            elif isinstance(args[0], dict):
                override_dict: Dict[str, any] = args[0]
            elif getattr(args[0], 'dict', None):
                override_dict: Dict[str, any] = args[0].dict()
            else:
                raise ValueError(f"Unsure how to interpret override input of type {type(args[0])}")

        elif (len(args) == 2) and isinstance(args[0], str):
            override_dict: Dict[str, any] = {args[0]: args[1]}

        else:
            raise ValueError(f"Unsure how to interpret 3+ inputs to override")

        return Style(**{**self.dict(), **override_dict})
