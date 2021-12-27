from typing import Any
from pydantic import BaseModel
from typing import Tuple
from .utils import Rosetta


class Style(BaseModel):
    """ Miscellaneous properties; set defaults once, then use or inherit elsewhere for consistent style
        (note: some names are formatted to be congruent with matplotlib.pyplot vars & args, rather than isthmuslib) """

    color: str = 'DarkGreen'  # isthmuslib default
    facecolor: str = 'w'
    title_fontsize: float = 18.0
    label_fontsize: float = 15.0
    legend_fontsize: float = 15.0
    tick_fontsize: float = 12.0
    figsize: Tuple[float, float] = (10.0, 8.0)
    linewidth: float = 5.0
    grid: str = 'on'
    tight_axes: bool = True
    formatter: str = '%Y-%m-%d %H:%M:%S'
    markersize: float = 50
    transparent_alpha: float = 0.5
    watermark_placement: Any = (0.05, 0.05)
    watermark_fontsize: float = 20
    watermark_color: str = 'dimgrey'
    histogram_bins: int = 25
    multi_hist_alpha: float = 0.5
    rosetta: Rosetta = Rosetta()

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
