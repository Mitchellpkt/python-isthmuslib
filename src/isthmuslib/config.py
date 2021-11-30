from pydantic import BaseModel
from typing import Tuple


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
