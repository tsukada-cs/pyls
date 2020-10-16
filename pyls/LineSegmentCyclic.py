import copy

import numpy as np
import pandas as pd
import xarray as xr

class LineSegmentCyclic():
    """LineSegmentCyclic is a class for handling line segments on cyclic boundaries."""
    
    def __init__(self, x1, x2, y1, y2, x_cyclic=False, y_cyclic=False, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Parameters
        ----------
        x1: array_like
            x values of start points in pixels. This corresponds to the azimuth axis.
        x2: array_like
            x values of end points in pixels. This corresponds to the azimuth axis.
        y1: array_like
            y values of start points in pixels. This corresponds to the time axis.
        y2: array_like
            y values of end points in pixels. This corresponds to the time axis.
        
        """
        super().__init__(x1, x2, y1, y2)
        self.is_cyclic = {"x": x_cyclic, "y": y_cyclic}
        if xmin is None:
            xmin = min(x1, x2)
        if xmax is None:
            xmax = max(x1, x2)
        if ymin is None:
            ymin = min(y1, y2)
        if ymax is None:
            ymax = max(y1, y2)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    