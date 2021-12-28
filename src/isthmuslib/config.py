from pydantic import BaseModel
from typing import Tuple, Dict, Any
from .utils import Rosetta


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

    def override(self, overrides: Dict[str, Any]):
        """ Helper function that overrides specific style features

        :param overrides: feature names and new values, e.g. ... = style.override({'color':'k', 'size':50})
        :return: returns a fresh copy of a Style object with the updated values
        """
        return Style(**{**self.dict(), **overrides})
