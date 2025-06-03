#!/usr/bin/env python3

import os
import glob
import time
import numpy as np
import argparse
from datetime import datetime

from scipy.stats import skew, kurtosis

from kanarie.utils import read_shelter, read_wview
from kanarie.model import observations_to_features, ShelterTemperatureModel

from matplotlib import pyplot as plt


def main(args):
    # Load in the shelter data
    shelter_data = [args.temperatures]
    shl_ts, shl_temp0, shl_temp1 = read_shelter(shelter_data)
    shl_temp = np.stack([shl_temp0,shl_temp1], axis=1)
    
    # Load in the weather data
    weather_data = glob.glob(os.path.join(args.weather_dir, 'wview_backup_*'))
    weather_data.sort()
    wx_ts, wx_temp, wx_rh, wx_sp, wx_dr = read_wview(weather_data)
    
    # Load in the model
    mdl = ShelterTemperatureModel(args.model)
    threshold = mdl.validation_std * 3
    
    # Extract features and compare to reality
    t0 = time.time()
    diff = []
    if args.show_plot:
        fig = plt.figure()
        ax = fig.gca()
    print("Analyzing data vs predictions...")
    for i in range(180, shl_ts.size, 15):
        if i % 500 == 0:
            print(f"  Working on data window {i} of {shl_ts.size} ({i/shl_ts.size:.1%} complete)")
            
        w = np.s_[i-180:i]
        try:
            features = observations_to_features(shl_ts[w], shl_temp[w],
                                                wx_ts, wx_temp, wx_rh, wx_sp, wx_dr)
            p = mdl.predict(features)
            diff.append(shl_temp[i] - p)
            
            overheat = False
            if shl_temp[i,0] > (82-32)*5/9 or shl_temp[i,1] > (82-32)*5/9:
                if shl_temp[i,0] > p[0] + threshold:
                    overheat = True
                    print(f"NOTICE: danger of overheating (sensor #1) at {datetime.utcfromtimestamp(shl_ts[i])}")
                    if args.show_plot:
                        ax.axvline(shl_ts[i], color='blue', linestyle=':')
                if shl_temp[i,1] > p[1] + threshold:
                    overheat = True
                    print(f"NOTICE: danger of overheating (sensor #2) at {datetime.utcfromtimestamp(shl_ts[i])}")
                    if args.show_plot:
                        ax.axvline(shl_ts[i], color='orange', linestyle=':')
                        
            if args.show_plot:
                if i % 180 == 0 or overheat:
                    ax.plot(shl_ts[w], shl_temp[w,0], color='blue')
                    ax.plot(shl_ts[w], shl_temp[w,1], color='orange')
                if i % 30 == 0 or overheat:
                    p2 = mdl.predict_with_uncertainty(features, 0.95)
                    
                    ax.errorbar(shl_ts[i], p2[0][0], [[-p2[1][0]+p2[0][0],],[p2[2][0]-p2[0][0]],], marker='o', color='blue')
                    ax.errorbar(shl_ts[i], p2[0][1], [[-p2[1][1]+p2[0][1],],[p2[2][1]-p2[0][1]],], marker='o', color='orange')
                    
        except RuntimeError as e:
            pass
    diff = np.array(diff)
    t1 = time.time()
    print(f"Finished analysis in {t1-t0:.1f} s")
    print('')
    
    med = np.median(diff)
    print(f"Mean actual - predicted: {diff.mean():.3f} C")
    print(f"Median actual - predicted: {med:.3f} C")
    print(f"MAD actual - predicted: {np.median(np.abs(diff-med)):.3f} C")
    print(f"Skew actual - predicted: {skew(diff.flatten()):.3f}")
    
    if args.show_plot:
        ax.set_xlabel('UNIX Timestamp [s]')
        ax.set_ylabel('Temperature [C]')
        plt.show()
        
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(diff[:,0])
    ax.hist(diff[:,1])
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
