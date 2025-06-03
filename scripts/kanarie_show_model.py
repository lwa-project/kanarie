#!/usr/bin/env python3

import os
import argparse

from kanarie.model import ShelterTemperatureModel


def main(args):
    for filename in args.filename:
        mdl = ShelterTemperatureModel(filename)
        desc = mdl.description
        print(f"Loaded '{filename}'")
        
        print("Properties:")
        print(f"  Number of trees: {desc['ntree']}")
        print(f"  Number of features: {desc['nfeature']}")
        print(f"  Number of values predicted per feature set: {desc['noutput']}")
        print(f"  Estimated training score: {desc['training_score']:.4%}")
        if 'validation_r_sq' in desc:
            print(f"  Validation R^2: {desc['validation_r_sq']:.4f}")
        if 'validation_std' in desc:
            print(f"  Validation std. dev.: {desc['validation_std']:.4f}")
        print('')
        
        print("Relative feature importance:")
        names = desc['feature_names']
        for i,f in enumerate(desc['feature_importance']):
            if names is not None:
                print(f"  {names[i]:20s}: {f:8.4%}")
            else:
                print(f"  Feature {i+1:2d}: {f:8.4%}")
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='show details about a shelter temperature model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str, nargs='+',
                        help='model(s) to report on')
    args = parser.parse_args()
    main(args)
