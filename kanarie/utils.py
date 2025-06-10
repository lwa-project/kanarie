import gzip
import numpy as np

from typing import Any, List, Union, Tuple

__all__ = ['read_shelter', 'read_wview', 'get_closest_point', 'get_closest_window']


def read_shelter(filenames: Union[str,List[str]]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Given a filename or list of filenames pointing to SHL temperature logs, load
    the data and return a three-element tuple of:
     * unix timestamps
     * first temperature sensor in C
     * second temperature sensor in C
    """
    
    if isinstance(filenames, str):
        filenames = [filenames,]
    filenames.sort()
    
    timestamps, temp0, temp1 = [], [], []
    for filename in filenames:
        opener = open
        omode = 'r'
        if filename.endswith('.gz'):
            opener = gzip.open
            omode = 'rt'
            
        with opener(filename, omode) as fh:
            for line in fh:
                fields = line.split(',')
                try:
                    t, s0, s1 = float(fields[0]), float(fields[1]), float(fields[2])
                    timestamps.append(t)
                    temp0.append(s0)
                    temp1.append(s1)
                except (IndexError, ValueError):
                    pass
                    
    timestamps = np.array(timestamps)
    temp0 = np.array(temp0)
    temp1 = np.array(temp1)
    
    return timestamps, temp0, temp1


def read_wview(filenames: Union[str,List[str]]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Given filename or list of filenames pointing to SHL wview weather data 
    backups, load the data and return a seven-element tuple of:
     * unix timestamps
     * outside temperature in F
     * atomospheric pressure in in of Hg
     * relative humidity
     * enthaply in kJ/kg
     * Wind speed in mph
     * Wind direction in degrees
    """
    
    if isinstance(filenames, str):
        filenames = [filenames,]
    filenames.sort()
    
    timestamps, temp, pressure, humidity, enthalpy, windsp, winddr = [], [], [], [], [], [], []
    for filename in filenames:
        opener = open
        omode = 'r'
        if filename.endswith('.gz'):
            opener = gzip.open
            omode = 'rt'
            
        with opener(filename, omode) as fh:
            for line in fh:
                if not line.startswith('INSERT'):
                    continue
                    
                _, _, fields, values = line.split()
                fields = fields.replace('archive(', '').replace(')', '')
                fields = fields.split(',')
                values = values.replace('VALUES(', '').replace(');', '')
                values = values.split(',')
                try:
                    ts = values[fields.index('dateTime')]
                    tm = values[fields.index('outTemp')]
                    pr = values[fields.index('pressure')]
                    hm = values[fields.index('outHumidity')]
                    wd = values[fields.index('windSpeed')]
                    dr = values[fields.index('windDir')]
                    if dr == 'NULL':
                        dr = '0'
                        
                    ts, tm, pr, hm, wd, dr = float(ts), float(tm), float(pr), float(hm), float(wd), float(dr)
                    
                    timestamps.append(ts)
                    temp.append(tm)
                    pressure.append(pr)
                    humidity.append(hm)
                    windsp.append(wd)
                    winddr.append(dr)
                    
                    # Compute the enthalpy of the air
                    tc = temp_F_to_C(tm)
                    pr = press_inHg_kPa(pr)
                    en = air_enthalpy(tc, pr, hm)
                    enthalpy.append(en)
                except (IndexError, ValueError) as e:
                    pass
                    
    timestamps = np.array(timestamps)
    temp = np.array(temp)
    pressure = np.array(pressure)
    humidity = np.array(humidity)
    enthalpy = np.array(enthalpy)
    windsp = np.array(windsp)
    winddr = np.array(winddr)
    
    return timestamps, temp, pressure, humidity, enthalpy, windsp, winddr


def get_closest_point(ts: float, timestamps: np.ndarray, *args: np.ndarray) -> List[Any]:
    """
    Given a unix timestamp value and a collection of measurments made at
    different times, find and return measurements closest to the timestamp.
    """
    
    best = np.argmin(np.abs(timestamps - ts))
    results = [data[best] for data in args]
    if len(results) == 1:
        results = results[0]
    return results


def get_closest_window(ts: float, timestamps: np.ndarray, *args: np.ndarray,
                       window_min: float=10) -> List[np.ndarray]:
    """
    Given a unix timestamp value and a collection of measurments made at
    different times, find and return measurements within the specified window
    of the timestamp.
    """
    
    age = (ts - timestamps) / 60.0  # age in minutes
    
    good = np.where((age >= 0) & (age < window_min))[0]
    results = [data[good] for data in args]
    if len(args) == 1:
        results = results[0]
    return results


def temp_F_to_C(temp):
    """
    Convert a temperature from F to C.
    """
    
    return (temp - 32) * 5/9.


def press_inHg_kPa(press):
    """
    Convert a pressue from in of Hg to kPa.
    """
    
    return press * 3.38639


def air_enthalpy(temp_C, press_kPa, rh_percent):
    """
    Given the temperature in C, the atmospheric pressure in kPa, and the
    relative humidity as a percentage (0-100%), estimate and return the
    enthalpy of the air in kJ/kg.
    """
    
    # Saturation vapor pressure of water (kPa)
    sat_press = 0.6108 * np.exp(17.27*temp_C / (temp_C + 237.3))
    # Actual vapor pressure of water (kPa)
    wv_press = (rh_percent / 100.) * sat_press
    # Specific humidity (kg/kg)
    sh = 0.622 * wv_press / (press_kPa - wv_press)
    # Enthalpy (kJ/kg)
    return 1.006*temp_C + sh*(1.86*temp_C + 2501)
