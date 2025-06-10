import numpy as np
from datetime import datetime

from .utils import get_closest_point

from typing import Any, Dict, List

__all__ = ['datetime_params', 'lookback_params', 'window_params', 'lagged_params',
           'cycle_params']


def datetime_params(timestamps: np.ndarray) -> Dict[str,float]:
    """
    Given a set of unix timestamps, compute and return the fractional day and
    year for the midpoint of the data.
    """
    
    midpt = len(timestamps)//2
    dt = datetime.utcfromtimestamp(timestamps[midpt]).timetuple()
    
    results = {}
    results['day_frac'] = (dt[3] + dt[4]/60 + dt[5]/3600) / 24
    results['year_frac'] = dt[7] / 365.25
    return results


def lookback_params(timestamps: np.ndarray, data: np.ndarray,
                    lookback_min: List[float]=[15, 30, 60, 120, 180, 240]) -> Dict[str,Any]:
    """
    Given a set of unix timestamps and associated data values, find values that
    are at certain points in the past.
    """
    
    reshape_needed = False
    if len(data.shape) == 1:
        reshape_needed = True
        data = data.reshape(*data.shape, 1)
        
    ages = (timestamps[-1] - timestamps) / 60.0 # ages in minutes
    
    results = {}
    for l in lookback_min:
        value = get_closest_point(timestamps[-1]-l*60, timestamps, data)
        results[f"{l:.0f}min"] = np.array(value)
        
    if reshape_needed:
        for k0 in results:
            results[k0] = results[k0].item()
            
    return results


def window_params(timestamps: np.ndarray, data: np.ndarray,
                  windows_min: List[float]=[15, 30, 60, 120, 180, 240]) -> Dict[str,Any]:
    """
    Given a set of unix timestamps and associated data values, compute
    parameters over the specified window lengths.  Parameters include:
     * mean
     * standard deviation
     * max - min
     * slope of the line that best fits the data
     
    ..note::  If a window contains fewer than three data points no analysis is
              performed on that window.
    """
    
    reshape_needed = False
    if len(data.shape) == 1:
        reshape_needed = True
        data = data.reshape(*data.shape, 1)
        
    ages = (timestamps[-1] - timestamps) / 60.0 # ages in minutes
    
    results = {}
    for w in windows_min:
        good = np.where((ages > 0) & (ages <= w))[0]
        if len(good) >= 3:
            slopes = []
            for i in range(data.shape[1]):
                slopes.append(np.polyfit(ages[good], data[good,i], 1)[0])
                
            results[f"{w:.0f}min_mean"]   = data[good,:].mean(axis=0)
            results[f"{w:.0f}min_std"]    = data[good,:].std(axis=0)
            results[f"{w:.0f}min_minmax"] = data[good,:].max(axis=0) - data[good].min(axis=0)
            results[f"{w:.0f}min_slope"]  = np.array(slopes)
            
    if reshape_needed:
        for k0 in results:
            results[k0] = results[k0].item()
            
    return results


def lagged_params(timestamps: np.ndarray, data: np.ndarray,
                  lags_min: List[float]=[15, 30, 60, 120, 180, 240]) -> Dict[str,Any]:
    """
    Given a set of unix timestamps and associated data values, compute
    mean values over the specified lags.  Parameters include:
     * mean
     * standard deviation
     * max - min
     * slope of the line that best fits the data
    
    ..note::  If a lag contains fewer than two data points no analysis is
              performed on that window.
    """
    
    reshape_needed = False
    if len(data.shape) == 1:
        reshape_needed = True
        data = data.reshape(*data.shape, 1)
        
    ages = (timestamps[-1] - timestamps) / 60.0 # ages in minutes
    timestep = abs(np.median(np.diff(ages)))    # median time step in minutes
    
    results = {}
    for l in lags_min:
        stepsize = int(round(l/timestep))
        stepstart = (data.shape[0] - 1) % stepsize
        good = np.s_[stepstart:-1:stepsize]
        subdata = data[good,:]
        if subdata.shape[0] >= 2:
            slopes = []
            for i in range(subdata.shape[1]):
                slopes.append(np.polyfit(ages[good], subdata[:,i], 1)[0])
                
            results[f"{l:.0f}min_lag_mean"] = subdata.mean(axis=0)
            results[f"{l:.0f}min_lag_std"]      = subdata.std(axis=0)
            results[f"{l:.0f}min_lag_minmax"]   = subdata.max(axis=0) - subdata.min(axis=0)
            results[f"{l:.0f}min_lag_slope"]    = np.array(slopes)
            
    if reshape_needed:
        for k0 in results:
            results[k0] = results[k0].item()
            
    return results


def cycle_params(timestamps: np.ndarray, data: np.ndarray, window_min: float=120) -> Dict[str,Any]:
    """
    Given a set of unix timestamps and associated data values, pull the data
    from the most recent window, remove the mean, compute the power spectrum,
    and return the indicies of the top five frequencies.
    
    .. note::  If there are fewer than 15 data points no analysis is performed.
    """
    
    reshape_needed = False
    if len(data.shape) == 1:
        reshape_needed = True
        data = data.reshape(*data.shape, 1)
        
    ages = (timestamps[-1] - timestamps) / 60.0 # ages in minutes
    
    results = {}
    good = np.where((ages > 0) & (ages <= window_min))[0]
    if len(good) > 15:
        subdata = data[good,:] - data[good,:].mean(axis=0)
        pwr = np.fft.fft(subdata, axis=0)
        pwr = np.abs(pwr[:pwr.shape[0]//2,:])**2
        norm = pwr.sum(axis=0)
        
        peaks, powers = [], []
        for i in range(data.shape[1]):
            o = np.argsort(pwr[:,i])[::-1]
            for j in range(5):
                peaks.append(o[j])
                if norm[i] > 0:
                    powers.append(pwr[o[j],i]/norm[i])
                else:
                    powers.append(0.0)
                    
        peaks, powers = np.array(peaks), np.array(powers)
        peaks, powers = peaks.reshape(-1,5), powers.reshape(-1,5)
        for i in range(5):
            results[f"peak{i}_idx"] = peaks[:,i]
            results[f"rel_power{i}_idx"] = powers[:,i]
            
    if reshape_needed:
        for k0 in results:
            results[k0] = results[k0].item()
            
    return results
