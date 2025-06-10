#!/usr/bin/env python3

import os
import glob
import time
import numpy as np
import argparse
from datetime import datetime

from kanarie.utils import read_shelter, read_wview
from kanarie.model import observations_to_features, ShelterTemperatureModel


def main(args):
    # Find the shelter data
    shelter_data = []
    for ext in ('', '.gz'):
        for name in ('thermometer01_*', 'enviromux_*'):
            shelter_data.extend(glob.glob(os.path.join(args.data_dir, f"{name}{ext}")))
    shelter_data.sort()
    
    if not shelter_data:
        raise RuntimeError(f"No shelter temperature data found in '{args.data_dir}'")
        
    # Find the weather data
    weather_data = []
    for ext in ('', '.gz'):
        weather_data.extend(glob.glob(os.path.join(args.data_dir, f"wview_backup_*{ext}")))
    weather_data.sort()
    
    if not weather_data:
        raise RuntimeError(f"No weather observations found in '{args.data_dir}'")
        
    # Load the shelter and weather data, then report on what we have
    shl_ts, shl_temp0, shl_temp1 = read_shelter(shelter_data)
    wx_ts, wx_temp, wx_press, wx_rh, wx_en, wx_sp, wx_dr = read_wview(weather_data)
    
    shl_start, shl_stop = datetime.utcfromtimestamp(shl_ts[0]), datetime.utcfromtimestamp(shl_ts[-1])
    wx_start, wx_stop = datetime.utcfromtimestamp(wx_ts[0]), datetime.utcfromtimestamp(wx_ts[-1])
    print(f"Loaded shelter data spanning {shl_start.strftime('%Y-%m-%d')} to {shl_stop.strftime('%Y-%m-%d')}")
    print(f"Loaded weather observations spanning {wx_start.strftime('%Y-%m-%d')} to {wx_stop.strftime('%Y-%m-%d')}")
    print('')
    
    # Select the right temperature to model
    if args.temp_select == 'both':
        shl_temp = np.stack([shl_temp0,shl_temp1], axis=1)
    elif args.temp_select == '0':
        shl_temp = shl_temp0
    elif args.temp_select == '1':
        shl_temp = shl_temp1
    else:
        raise ValueError(f"Unknown temperature selection '{args.temp_select}'")
        
    # Extract the features
    t0 = time.time()
    first = True
    features, values, feature_names = [], [], None
    vfeatures, vvalues = [], []
    print("Extracting features...")
    for i in range(180, shl_ts.size, 30):
        if i % 1000 == 0:
            print(f"  Working on data window {i} of {shl_ts.size} ({i/shl_ts.size:.1%} complete)")
            
        w = np.s_[i-180:i]
        try:
            f = observations_to_features(shl_ts[w], shl_temp[w],
                                         wx_ts, wx_temp, wx_press, wx_rh, wx_en, wx_sp, wx_dr,
                                         return_feature_names=first)
            if first:
                f, feature_names = f
                first = False
                
            if np.random.rand() > 0.8:
                vfeatures.append(f)
                vvalues.append(shl_temp[i])
            else:
                features.append(f)
                values.append(shl_temp[i])
        except RuntimeError as e:
            if args.show_errors:
                print(f"Error at timestamp {shl_ts[i]}: {str(e)}")
    features = np.array(features)
    values = np.array(values)
    vfeatures = np.array(vfeatures)
    vvalues = np.array(vvalues)
    t1 = time.time()
    print(f"Finished feature extraction in {t1-t0:.1f} s")
    print(f"  Training entries: {features.shape[0]}")
    print(f"  Validation entries: {vfeatures.shape[0]}")
    print('')
    
    # Build the model
    t0 = time.time()
    print("Building the model...")
    mdl = ShelterTemperatureModel(None)
    if args.auto_select:
        mdl.fit_with_feature_selection(features, values, feature_names)
    else:
        mdl.fit(features, values, feature_names=feature_names)
    t1 = time.time()
    print(f"Finished building the model in {t1-t0:.1f} s")
    print('')
    
    t0 = time.time()
    print("Validating the model...")
    cod = mdl.score(vfeatures, vvalues)
    print(f"Coefficient of Determination: {cod:.4f}")
    print(f"Data variance: {np.var(vvalues):.4f}")
    t1 = time.time()
    print(f"Finished validating the model in {t1-t0:.1f} s")
    print('')
    
    # Save the model
    mdl.save(args.output, overwrite=args.overwrite)
    print(f"Model saved to '{args.output}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='given a collection of shelter temperature and weather data, build a model to predict the shelter temperature',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('data_dir', type=str,
                        help='directory containing the shelter and weather data to fit')
    parser.add_argument('-s', '--show-errors', action='store_true',
                        help='do not show feature extraction errors')
    parser.add_argument('--temp-select', type=str, default='0',
                        help='build the model using the specified temperature sensor')
    parser.add_argument('-a', '--auto-select', action='store_true',
                        help='auto-select which features to use based on importance')
    parser.add_argument('-o', '--output', type=str, default='model.pkl',
                        help='filename to save the model to')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite the output model file if it exists')
    args = parser.parse_args()
    main(args)
