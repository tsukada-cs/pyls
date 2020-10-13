import copy

import numpy as np
import pandas as pd
import xarray as xr

class LineSegment():
    """LineSegment is a class for handling line segments"""
    
    def __init__(self, x1, x2, y1, y2):
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
        x1, x2, y1, y2 = np.array(x1), np.array(x2), np.array(y1), np.array(y2)
        index = np.arange(x1.size)
        x1 = xr.DataArray(x1, name="x1", dims=('i'), coords={'i':index}, attrs={'units':"1"})
        x2 = xr.DataArray(x2, name="x2", dims=('i'), coords={'i':index}, attrs={'units':"1"})
        y1 = xr.DataArray(y1, name="y1", dims=('i'), coords={'i':index}, attrs={'units':"1"})
        y2 = xr.DataArray(y2, name="y2", dims=('i'), coords={'i':index}, attrs={'units':"1"})
        self.lines = xr.Dataset(data_vars={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, attrs=None)

    def __str__(self):
        return str(self.lines)
    
    def __repr__(self):
        return repr(self.lines)
    
    def __getitem__(self, key):
        return self.lines[key]
    
    def __setitem__(self, key, value):
        self.lines[key] = value

    def __len__(self):
        return len(self.lines)

    def sel(self, **kwargs):
        return self.lines.sel(kwargs)

    def isel(self, **kwargs):
        return self.lines.isel(kwargs)

    @property
    def length(self):
        return np.sqrt((self.lines["x2"]-self.lines["x1"])**2 + (self.lines["y2"]-self.lines["y1"])**2)
    @property
    def x_length(self):
        return np.abs(self.lines["x2"]-self.lines["x1"])
    @property
    def y_length(self):
        return np.abs(self.lines["y2"]-self.lines["y1"])
    @property
    def i(self):
        return self.lines.i
    @property
    def x1(self):
        return self.lines.x1
    @property
    def x2(self):
        return self.lines.x2
    @property
    def y1(self):
        return self.lines.y1
    @property
    def y2(self):
        return self.lines.y2
    @property
    def attrs(self):
        return self.lines.attrs

    @staticmethod
    def from_netcdf(nc_name):
        da = xr.open_dataset(nc_name)
        return LineSegment(x1=da.x1.values, y1=da.y1.values, x2=da.x2.values, y2=da.y2.values)

    def get_left(self, img, gap=1.0):
        """
        Returns the values at positions along the line segment that are only a gap pixel to the left.
        
        Parameters
        ----------
        img : numpy ndarray
            Image to get a value.
        gap : int or float, default 1.0
            Distance from line segment.
        """
        left = xr.DataArray(np.zeros(self.i.size), name="left", dims=('i'), coords={'i':self.i}, attrs={'units':"1"})
        for i, (x1, y1, x2, y2) in enumerate(zip(self.x1.values, self.y1.values, self.x2.values, self.y2.values)):
            x10, y10 = np.linspace(x1, x2, 10), np.linspace(y1, y2, 10)
            angle = np.arctan2(y2-y1, x2-x1)
            x10L = np.round(x10 - gap * np.cos(angle + np.pi/2)).astype(int) % img.shape[1]
            y10L = np.round(y10 - gap * np.sin(angle + np.pi/2)).astype(int)
            y10L_inboard = np.logical_and(y10L >= 0, y10L < img.shape[0])
            x10L, y10L = x10L[y10L_inboard], y10L[y10L_inboard]
            left.values[i] = img[y10L, x10L].mean()
        return left

    def get_right(self, img, gap=1.0):
        """
        Returns the values at positions along the line segment that are only a gap pixel to the right.
        
        Parameters
        ----------
        img : numpy ndarray
            Image to get a value.
        gap : int or float, default 1.0
            Distance from line segment.
        """
        right = xr.DataArray(np.zeros(self.i.size), name="right", dims=('i'), coords={'i':self.i}, attrs={'units':"1"})
        for i, (x1, y1, x2, y2) in enumerate(zip(self.x1.values, self.y1.values, self.x2.values, self.y2.values)):
            x10, y10 = np.linspace(x1, x2, 10), np.linspace(y1, y2, 10)
            angle = np.arctan2(y2-y1, x2-x1)
            x10R = np.round(x10 + gap * np.cos(angle + np.pi/2)).astype(int) % img.shape[1]
            y10R = np.round(y10 + gap * np.sin(angle + np.pi/2)).astype(int)
            y10R_inboard = np.logical_and(y10R >= 0, y10R < img.shape[0])
            x10R, y10R = x10R[y10R_inboard], y10R[y10R_inboard]
            right.values[i] = img[y10R, x10R].mean()
        return right

    def get_high(self, img, gap=1.0):
        """
        Returns the higher value of the point along the line segment that is gap pixels away.
        Parameters
        ----------
        img : numpy ndarray
            Image to get a value.
        gap : int or float, default 1.0
            Distance from line segment.
        """
        high = xr.DataArray(np.zeros(self.i.size), name="high", dims=('i'), coords={'i':self.i}, attrs={'units':"1"})
        for i, (l, r) in enumerate(zip(self.get_left(img, gap=gap), self.get_right(img, gap=gap))):
            high[i] = max(l, r).values
        return high

    def get_low(self, img, gap=1.0):
        """
        Returns the lower value of the point along the line segment that is gap pixels away.
        Parameters
        ----------
        img : numpy ndarray
            Image to get a value.
        gap : int or float, default 1.0
            Distance from line segment.
        """
        low = xr.DataArray(np.zeros(self.i.size), name="low", dims=('i'), coords={'i':self.i}, attrs={'units':"1"})
        for i, (l, r) in enumerate(zip(self.get_left(img, gap=gap), self.get_right(img, gap=gap))):
            low[i] = min(l, r).values
        return low

    def limit_value(self, img, min_value=None, max_value=None, gap=1, which="low"):
        """
        Returns a LineSegment consisting only of line segments with limited values along lines.
        
        Parameters
        ----------
        img : numpy ndarray
            Image to get a value.
        min_value: float
            Minimum value.
        max_value: float
            Maximum value.
        gap: int or float, default 1
            Gap along to line segments.
        which: str, default "low"
            Limit value along this side.
        Returns
        -------
        new_ls: LineSegment
        """
        if min_value is max_value is None:
            raise ValueError("min_value or max_value must not be None")
        
        if which == "high":
            get_func = self.get_high
        if which == "low":
            get_func = self.get_low

        val = get_func(img, gap=gap).values

        if min_value is not None:
            larger_index = (min_value <= val)
        else:
            larger_index = np.ones_like(self.lines.i, bool)
        if max_value is not None:
            smaller_index = (val < max_value)
        else:
            smaller_index = np.ones_like(self.lines.i, bool)

        range_index = larger_index * smaller_index
        new_ls = copy.deepcopy(self)
        new_ls.lines = new_ls.lines.isel(i=range_index)
        return new_ls

    def sort(self, which="y", large_2=True):
        if large_2:
            large = which + "2"
            small = which + "1"
        else:
            large = which + "1"
            small = which + "2"
        swap_index = self.lines[small] > self.lines[large]
        new_ls = copy.deepcopy(self)
        new_ls.lines["x1"][swap_index] = self.lines["x2"][swap_index]
        new_ls.lines["x2"][swap_index] = self.lines["x1"][swap_index]
        new_ls.lines["y1"][swap_index] = self.lines["y2"][swap_index]
        new_ls.lines["y2"][swap_index] = self.lines["y1"][swap_index]
        return new_ls
        
    def to_netcdf(self, oname):
        encoding={
            "x1":{'dtype':"f4", "_FillValue": np.finfo(np.float32).max},
            "x2":{'dtype':"f4", "_FillValue": np.finfo(np.float32).max},
            "y1":{'dtype':"f4", "_FillValue": np.finfo(np.float32).max},
            "y2":{'dtype':"f4", "_FillValue": np.finfo(np.float32).max}
        }
        self.lines.to_netcdf(oname, encodign=encoding)
