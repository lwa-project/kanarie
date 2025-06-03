#!/usr/bin/env python3

import os
import glob
import time
import numpy as np
import argparse
from datetime import datetime

from kanarie.utils import read_shelter, read_wview
from kanarie.model import observations_to_features, ShelterTemperatureModel

from matplotlib import pyplot as plt


def main(args):
    # Load in the shelter data
    shelter_data = [args.temperatures]
    shl_ts, shl_temp0, shl_temp1 = read_shelter(shelter_data)
    shl_temp = np.stack([shl_temp0,shl_temp1], axis=1)
    
    # Load in the weather data
    weather_data = []
    for ext in ('', '.gz'):
        weather_data.extend(glob.glob(os.path.join(args.weather_dir, f"wview_backup_*{ext}")))
    weather_data.sort()
    wx_ts, wx_temp, wx_rh, wx_sp, wx_dr = read_wview(weather_data)
    
    # Load in the model
    mdl = ShelterTemperatureModel(args.model)
    
    # Extract features and compare to reality
    t0 = time.time()
    diff = []
    if args.show_plot:
        fig = plt.figure()
        ax = fig.gca()
    print("Analyzing data vs predictions...")
    for i in range(180, shl_ts.size, 10):
        if i % 500 == 0:
            print(f"  Working on data window {i} of {shl_ts.size} ({i/shl_ts.size:.1%} complete)")
            
        w = np.s_[i-180:i]
        try:
            features = observations_to_features(shl_ts[w], shl_temp[w],
                                                wx_ts, wx_temp, wx_rh, wx_sp, wx_dr)
            p = mdl.predict(features)
            diff.append(shl_temp0[i] - p)
            
            if args.show_plot:
                if i % 180 == 0:
                    ax.plot(shl_ts[w], shl_temp0[w], color='blue')
                if i % 30 == 0:
                    ax.scatter(shl_ts[i], p, color='orange')
                    
        except RuntimeError as e:
            pass
    diff = np.array(diff)
    t1 = time.time()
    print(f"Finished analysis in {t1-t0:.1f} s")
    print('')
    
    med = np.median(diff)
    print(f"Mean actual - predicted: {diff.mean():.1f} C")
    print(f"Median actual - predicted: {med:.1f} C")
    print(f"MAD actual - predicted: {np.median(np.abs(diff-med)):.1f} C")
    
    if args.show_plot:
        ax.set_xlabel('UNIX Timestamp [s]')
        ax.set_ylabel('Temperature [C]')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='given a shelter thermometer/enviromux file and a directory containing weather data estimate how the HVACs are performing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('model', type=str,
                        help='model to use for the analysis')
    parser.add_argument('temperatures', type=str,
                        help='therometer or enviromux file containing the shelter data to analyze')
    parser.add_argument('weather_dir', type=str,
                        help='directory containing weather data')
    parser.add_argument('--show-plot', action='store_true',
                        help='show snapshots of the predicted temperatures')
    args = parser.parse_args()
    main(args)
