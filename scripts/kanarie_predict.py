#!/usr/bin/env python3

import os
import time
import numpy as np
import argparse
from urllib import request
from datetime import datetime

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
    wx_ts, wx_temp, wx_humid, wx_sp, wx_dr = [], [], [], [], []
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
    wx_ts = np.array(wx_ts)
    wx_temp = np.array(wx_temp)
    wx_humid = np.array(wx_humid)
    wx_sp = np.array(wx_sp)
    wx_dr = np.array(wx_dr)
    
    # Load in the model
    mdl = ShelterTemperatureModel(f"{args.station}.pkl")
    threshold = mdl.validation_std * 3
    
    # Extract a set of features for the last three slugs of data
    features, values, tstamps = [], [], []
    for stp in (-3, -2, -1):
        features.append(observations_to_features(shl_ts[:stp], shl_temp[:stp],
                                                 wx_ts, wx_temp, wx_humid, wx_sp, wx_dr))
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
    if nhigh > 3:
        print(f"NOTICE: {args.station} might be in danger of overheating")
        for ts,temp,p in zip(tstamps, values, predicted):
            print(f"  {ts:.1f} -> {temp} vs {p} predicted")
    else:
        print(f"Shelter temperatures appear normal (nhigh={nhigh})")
        if args.verbose:
            print("Details:")
            for t,v,p in zip(tstamps, values, predicted):
                print(f"  Timestamp: {t:.0f} s")
                print(f"    Actual: {' C, '.join(['%.1f' % v2 for v2 in v])} C")
                print(f"    Predicted: {' C, '.join(['%.1f' % p2 for p2 in p])} C")
                print(f"    Difference: {' C, '.join(['%.1f' % (v2-p2) for v2,p2 in zip(v,p)])} C")
                print(f"                {' sigma, '.join(['%.1f' % ((v2-p2)/mdl.validation_std) for v2,p2 in zip(v,p)])} sigma")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='use data from the OpScreen page to check on a station',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('station', type=str, default='lwa1',
                        help='station to check')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose about the predictions')
    args = parser.parse_args()
    args.station = args.station.lower().replace('-', '')
    main(args)
