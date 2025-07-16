#!/usr/bin/env python3

import os
import json
import math
import time
import numpy as np
import argparse
from urllib import request
from datetime import datetime

from kanarie.utils import temp_F_to_C, press_inHg_kPa, air_enthalpy
from kanarie.model import observations_to_features, ShelterTemperatureModel


def dir_to_ang(dr: str) -> float:
    """
    Convert a cardinal direction to an approximate azimuth in degrees.
    """
    
    conv = ['N', 'NNE', 'NE', 'ENE',
            'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW',
            'W', 'WNW', 'NW', 'NNW',
            'N']
    return conv.index(dr)*22.5


def format_temps(values, units=''):
    """
    Helper function to help make printing temperatures/differences cleaner.
    """
    
    if units != '':
        units = ' '+units
        
    try:
        return ', '.join([f"{v:.1f}{units}" for v in values])
    except:
        return f"{values:.1f}{units}"


def main(args):
    # Download the current list of shelter temperatures
    shl_ts, shl_temp0, shl_temp1 = [], [], []
    with request.urlopen(f"https://lwalab.phys.unm.edu/OpScreen/{args.station}/shl.dat",
                         timeout=30) as uh:
        data = uh.read()
        data = data.decode()
        for line in data.split('\n'):
            if len(line) < 3:
                continue
                
            fields = line.split(',')
            ts, temp0, temp1 = float(fields[0]), float(fields[1]), float(fields[2])
            shl_ts.append(ts)
            shl_temp0.append(temp0)
            shl_temp1.append(temp1)
    shl_ts, shl_temp0, shl_temp1 = np.array(shl_ts), np.array(shl_temp0), np.array(shl_temp1)
    shl_temp = np.stack([shl_temp0,shl_temp1], axis=1)
    
    # Download the current weather conditions
    wx_ts, wx_temp, wx_press, wx_humid, wx_enth, wx_sp, wx_dr = [], [], [], [], [], [], []
    with request.urlopen(f"https://lwalab.phys.unm.edu/OpScreen/{args.station}/weather.dat",
                         timeout=30) as uh:
        data = uh.read()
        data = data.decode()
        for line in data.split('\n'):
            fields = line.split()
            if line.find('Temperature') != -1:
                _, t = fields[-2].split('>', 1)
                wx_temp.append(float(t))
                wx_temp.append(float(t))
                wx_temp.append(float(t))
            elif line.find('Humidity') != -1:
                h, _ = fields[-1].split('%', 1)
                _, h = h.split('>', 1)
                wx_humid.append(float(h))
                wx_humid.append(float(h))
                wx_humid.append(float(h))
            elif line.find('Pressure') != -1:
                _, p = fields[-2].split('>', 1)
                wx_press.append(float(p))
                wx_press.append(float(p))
                wx_press.append(float(p))
            elif line.find('Wind<') != -1:
                _, sp = fields[-5].split('>', 1)
                dr, _ = fields[-1].split('<', 1)
                wx_sp.append(float(sp))
                wx_sp.append(float(sp))
                wx_sp.append(float(sp))
                wx_dr.append(dir_to_ang(dr))
                wx_dr.append(dir_to_ang(dr))
                wx_dr.append(dir_to_ang(dr))
            elif line.find('Updated') != -1:
                wx_ts.append(time.time() -   0)     # This updates every 10 min
                wx_ts.append(time.time() - 600)
                wx_ts.append(time.time() - 900)
    for tm,pr,rh in zip(wx_temp, wx_press, wx_humid):
        tc = temp_F_to_C(tm)
        pr = press_inHg_kPa(pr)
        pr *= ((293 - 0.0065*2000) / 293.0)**5.26 # ~Uncorrect for the station altitude
        en = air_enthalpy(tc, pr, rh)
        wx_enth.append(en)
    wx_ts = np.array(wx_ts)
    wx_temp = np.array(wx_temp)
    wx_press = np.array(wx_press)
    wx_humid = np.array(wx_humid)
    wx_enth = np.array(wx_enth)
    wx_sp = np.array(wx_sp)
    wx_dr = np.array(wx_dr)
    
    # Load in the model
    mdl = ShelterTemperatureModel(f"{args.station}.pkl")
    threshold = mdl.validation_std * 3
    
    # Extract a set of features for the last three slugs of data
    features, values, tstamps = [], [], []
    for stp in (-3, -2, -1):
        features.append(observations_to_features(shl_ts[:stp], shl_temp[:stp],
                                                 wx_ts, wx_temp, wx_press, wx_humid, wx_enth, wx_sp, wx_dr))
        values.append(shl_temp[stp])
        tstamps.append(shl_ts[stp])
        
    # Predict
    predicted = []
    for f in features:
        predicted.append(mdl.predict(f))
        
    # Analyze
    nhigh = 0
    for temp,p in zip(values, predicted):
        ## Only check if we are over 82 F
        if temp[0] > 27.78 or temp[1] > 27.78:
            if temp[0] > p[0] + threshold:
                nhigh += 1
            if temp[1] > p[1] + threshold:
                nhigh += 1
                
    # Gather everything together
    json_data = {'station': args.station,
                 'model_uncertainty_C': mdl.validation_std,
                 'high_threshold_sigma': threshold/mdl.validation_std,
                 'nsensor': 2,
                 'timestamps': [datetime.utcfromtimestamp(ts).isoformat() for ts in tstamps],
                 'observed_temperatures_C': [v.tolist() for v in values],
                 'predicted_temperatures_C': [p.tolist() for p in values],
                 'diff_temperatures_C': [(v-p).tolist() for v,p in zip(values, predicted)],
                 'diff_temperatures_sigma': [((v-p)/mdl.validation_std).tolist() for v,p in zip(values, predicted)],
                 'nhigh': nhigh,
                 'status': ''}
    
    # Make a judgement
    json_data['status'] = f"Shelter temperatures appear normal (nhigh={nhigh})"
    if nhigh >= 3:
        json_data['status'] = f"NOTICE: {args.station} might be in danger of overheating"
        args.verbose = True
        
    # Report
    if args.json:
        print(json.dumps(json_data))
    else:
        print(json_data['status'])
        if args.verbose:
            print("Details:")
            for t,v,p in zip(tstamps, values, predicted):
                sigs = np.abs(v - p) / mdl.validation_std
                print(f"  Timestamp: {t:.0f} s ({datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')} UTC)")
                print(f"    Actual: {format_temps(v, 'C')}")
                print(f"    Predicted: {format_temps(p, 'C')}")
                print(f"    Difference: {format_temps(v-p, 'C')}")
                print(f"                {format_temps(sigs, 'sigma')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use data from the OpScreen page to check on a station',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('station', type=str, default='lwa1',
                        help='station to check')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose about the predictions')
    parser.add_argument('--json', action='store_true',
                        help='return the output as JSON')
    args = parser.parse_args()
    args.station = args.station.lower().replace('-', '')
    main(args)
