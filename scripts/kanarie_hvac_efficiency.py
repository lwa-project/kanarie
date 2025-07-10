#!/usr/bin/env python3

import os
import glob
import time
import numpy as np
import argparse
from datetime import datetime, timedelta

from kanarie.utils import read_shelter, read_wview
from kanarie.model import observations_to_features, ShelterTemperatureModel

from matplotlib import pyplot as plt
from matplotlib import dates as mdates


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


def autofmt_xdate(ax, fmt='%m-%d'):
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
        
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def main(args):
    # Load shelter data
    shl_ts, shl_temp0, shl_temp1 = read_shelter(args.temperatures)
    shl_temp = np.stack([shl_temp0, shl_temp1], axis=1)
    
    # Load weather data
    weather_data = []
    for ext in ('', '.gz'):
        weather_data.extend(glob.glob(os.path.join(args.weather_dir, f"wview_backup_*{ext}")))
    weather_data.sort()
    wx_ts, wx_temp, wx_press, wx_rh, wx_en, wx_sp, wx_dr = read_wview(weather_data)
    
    # Load model
    mdl = ShelterTemperatureModel(args.model)
    print(f"HVAC Efficiency Analysis - {os.path.basename(args.model)}")
    print(f"  Model validation std dev: {mdl.validation_std:.3f} C")
    print(f"  3-sigma threshold: {3*mdl.validation_std:.3f} C")
    print('')
    
    # Extract residuals
    t0 = time.time()
    values, predicted, residuals, tstamps, otemps = [], [], [], [], []
    get_names = True
    print("Extracting residuals...")
    for i in range(180, shl_ts.size, 30):
        if i % 1000 == 0:
            print(f"  Working on data window {i} of {shl_ts.size} ({i/shl_ts.size:.1%} complete)")
            
        w = np.s_[i-180:i]
        try:
            features = observations_to_features(shl_ts[w], shl_temp[w],
                                                wx_ts, wx_temp, wx_press, wx_rh, wx_en, wx_sp, wx_dr,
                                                return_feature_names=get_names)
            if get_names:
                features, names = features
                otemp_idx = names.index('wx_temp')
                get_names = False
                
            tstamps.append(shl_ts[i])
            values.append(shl_temp[i])
            predicted.append(mdl.predict(features))
            residuals.append(values[-1]-predicted[-1])
            
            otemps.append(features[otemp_idx])
        except Exception as e:
            print(str(e))
    tstamps = np.array(tstamps)
    residuals = np.array(residuals)
    abs_residuals = np.abs(residuals)
    sig_residuals = residuals / mdl.validation_std
    otemps = np.array(otemps)
    t1 = time.time()
    print(f"Finished residuals extraction in {t1-t0:.1f} s")
    print(f"  Entries: {tstamps.size}")
    print('')
    
    t0 = time.time()
    print("Analysis...")
    print(f"  Mean residuals: {format_temps(residuals.mean(axis=0), units='C')}")
    print(f"  Mean |residuals|: {format_temps(abs_residuals.mean(axis=0), units='C')}")
    print(f"  Std residuals: {format_temps(residuals.std(axis=0), units='C')}")
    print(f"  Std |residuals|: {format_temps(abs_residuals.std(axis=0), units='C')}")
    
    print("  Residuals Distribution:")
    for bins in ((-np.inf,-2), (-2,-1), (-1,1), (1,2), (2,3), (3,np.inf)):
        counts = []
        if not np.isfinite(bins[0]):
            for i in range(residuals.shape[1]):
                counts.append( len(np.where(sig_residuals[:,i] < bins[1])[0]) )
        elif not np.isfinite(bins[1]):
            for i in range(residuals.shape[1]):
                counts.append( len(np.where(sig_residuals[:,i] >= bins[0])[0]) )
        else:
            for i in range(residuals.shape[1]):
                counts.append( len(np.where((sig_residuals[:,i] >= bins[0]) & (sig_residuals[:,i] < bins[1]))[0]) )
        print(f"    {bins[0]:4.0f} <= r/sigma < {bins[1]:4.0f}: {counts[0]/tstamps.size:.1%}, {counts[1]/tstamps.size:.1%}")
    t1 = time.time()
    print(f"Finished analysis in {t1-t0:.1f} s")
    print('')
    
    time_trends = None
    if tstamps.size > 50:
        t0 = time.time()
        print("Exploring trending with time...")
        twindows, windows = [], []
        for i in range(tstamps.size):
            window = np.where(np.abs(tstamps - tstamps[i]) < (args.window_days*86400*0.5))[0]
            if len(window) < 20:
                continue
                
            window_residuals = residuals[window]
            high = np.sum(window_residuals > (3*mdl.validation_std)) / len(window)
            low = np.sum(window_residuals < (-2*mdl.validation_std)) / len(window)
            twindows.append(tstamps[window].mean())
            windows.append([high,low])
        twindows = np.array(twindows)
        windows = np.array(windows)
        time_trends = {'times': twindows, 'results': windows}
        
        start = np.s_[:twindows.size//3]
        middle = np.s_[twindows.size//3:2*twindows.size//3]
        end = np.s_[2*twindows.size//3:]
        high_res = 'steady'
        if windows[start,0].mean() < windows[middle,0].mean() and windows[middle,0].mean() < windows[end,0].mean():
            high_res = 'degrading'
        elif windows[start,0].mean() > windows[middle,0].mean() and windows[middle,0].mean() > windows[end,0].mean():
            high_res = 'improving'
            
        low_res = 'steady'
        if windows[start,1].mean() < windows[middle,1].mean() and windows[middle,1].mean() < windows[end,1].mean():
            low_res = 'improving'
        elif windows[start,1].mean() > windows[middle,1].mean() and windows[middle,1].mean() > windows[end,1].mean():
            low_res = 'degrading'
            
        print(f"  >  3-sigma events: {windows[start,0].mean():.1%} -> {windows[middle,0].mean():.1%} -> {windows[end,0].mean():.1%} = {high_res}")
        print(f"  < -2-sigma events: {windows[start,1].mean():.1%} -> {windows[middle,1].mean():.1%} -> {windows[end,1].mean():.1%} = {low_res}")
    t1 = time.time()
    print(f"Finished trend analysis in {t1-t0:.1f} s")
    print('')
    
    # Generate plots, if requested
    if args.show_plots:
        print("Plotting...")
        
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.scatter(otemps, residuals[:,0], alpha=0.6, s=20, label='Sensor 1')
        ax.scatter(otemps, residuals[:,1], alpha=0.6, s=20, label='Sensor 2')
        ax.axhline(0.0, color='black', alpha=0.8)
        ax.axhline( 1*mdl.validation_std, color='orange', linestyle='--', alpha=0.7, label='|1$\\sigma$|')
        ax.axhline(-1*mdl.validation_std, color='orange', linestyle='--', alpha=0.7)
        ax.axhline( 2*mdl.validation_std, color='red', linestyle='--', alpha=0.7, label='|2$\\sigma$|')
        ax.axhline(-2*mdl.validation_std, color='red', linestyle='--', alpha=0.7)
        ax.axhline( 3*mdl.validation_std, color='purple', linestyle='--', alpha=0.7, label='|3$\\sigma$|')
        ax.axhline(-3*mdl.validation_std, color='purple', linestyle='--', alpha=0.7)
        ax.set_xlabel('Outdoor Temperature [F]')
        ax.set_ylabel('Residual (Actual - Predicted) [C]')
        ax.set_title('Residuals vs. Outdoor Temperature')
        ax.legend(loc=0)
        
        ax = fig.add_subplot(2, 2, 2)
        ax.scatter([datetime.utcfromtimestamp(t) for t in tstamps], residuals[:,0], alpha=0.6, s=20, label='Sensor 1')
        ax.scatter([datetime.utcfromtimestamp(t) for t in tstamps], residuals[:,1], alpha=0.6, s=20, label='Sensor 2')
        ax.axhline(0.0, color='black', alpha=0.8)
        ax.axhline( 1*mdl.validation_std, color='orange', linestyle='--', alpha=0.7, label='|1$\\sigma$|')
        ax.axhline(-1*mdl.validation_std, color='orange', linestyle='--', alpha=0.7)
        ax.axhline( 2*mdl.validation_std, color='red', linestyle='--', alpha=0.7, label='|2$\\sigma$|')
        ax.axhline(-2*mdl.validation_std, color='red', linestyle='--', alpha=0.7)
        ax.axhline( 3*mdl.validation_std, color='purple', linestyle='--', alpha=0.7, label='|3$\\sigma$|')
        ax.axhline(-3*mdl.validation_std, color='purple', linestyle='--', alpha=0.7)
        ax.set_xlabel('Date/Time')
        ax.set_ylabel('Residual (Actual - Predicted) [C]')
        ax.set_title('Residuals vs. Date/Time')
        ax.legend(loc=0)
        autofmt_xdate(ax)
        
        ax = fig.add_subplot(2, 2, 3)
        ax.hist(residuals, bins=50, alpha=0.7, density=True)
        ax.axhline(0.0, color='black', alpha=0.8)
        ax.axvline( 1*mdl.validation_std, color='orange', linestyle='--', alpha=0.7, label='|1$\\sigma$|')
        ax.axvline(-1*mdl.validation_std, color='orange', linestyle='--', alpha=0.7)
        ax.axvline( 2*mdl.validation_std, color='red', linestyle='--', alpha=0.7, label='|2$\\sigma$|')
        ax.axvline(-2*mdl.validation_std, color='red', linestyle='--', alpha=0.7)
        ax.axvline( 3*mdl.validation_std, color='purple', linestyle='--', alpha=0.7, label='|3$\\sigma$|')
        ax.axvline(-3*mdl.validation_std, color='purple', linestyle='--', alpha=0.7)
        ax.set_xlabel('Residual (Actual - Predicted) [C]')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution')
        
        if time_trends is not None:
            ax = fig.add_subplot(2, 2, 4)
            ax.plot([datetime.utcfromtimestamp(t) for t in time_trends['times']], time_trends['results'][:,0]*100, label='>3$\\sigma$')
            ax.plot([datetime.utcfromtimestamp(t) for t in time_trends['times']], time_trends['results'][:,1]*100, label='<-2$\\sigma$')
            ax.set_xlabel('Fraction [%]')
            ax.set_ylabel('Date/Time')
            ax.set_title('Extrema Time Trending')
            ax.legend(loc=0)
            autofmt_xdate(ax)
            
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Uncertainty-based HVAC efficiency analysis using model validation std dev',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model', type=str,
                        help='model to use for the analysis')
    parser.add_argument('temperatures', type=str,
                        help='thermometer or enviromux file containing shelter data')
    parser.add_argument('weather_dir', type=str,
                        help='directory containing weather data')
    parser.add_argument('--window-days', type=int, default=7,
                        help='rolling window size in days for trend analysis')
    parser.add_argument('--show-plots', action='store_true',
                        help='show diagnostic plots')
    args = parser.parse_args()
    main(args)
