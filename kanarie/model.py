import os
import pickle as pkl
import numpy as np
from textwrap import fill as tw_fill

from sklearn.ensemble import RandomForestRegressor

from .utils import get_closest_window
from .analysis import *

from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ['observations_to_features', 'ShelterTemperatureModel']


def observations_to_features(shelter_timestamps: np.ndarray,
                             shelter_temps: np.ndarray,
                             weather_timestamps: np.ndarray,
                             weather_temp: np.ndarray,
                             weather_pressure: np.ndarray,
                             weather_humidity: np.ndarray,
                             weather_enthalpy: np.ndarray,
                             weather_windspeed: np.ndarray,
                             weather_winddir: np.ndarray,
                             window_min: float=180,
                             return_feature_names: bool=False) -> Union[np.ndarray,Tuple[np.ndarray,List[str]]]:
    """
    Given a collection of observations, extract a set of features for use with
    a model.  Returns the feature set associated with the last window in the
    data.
    """
    
    # All possible windows to use, up to 6 hours into the past
    windows_all = [ 5,10,15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                   70,80,90,100,110,120,140,160,180,200,220,240,270,
                   300,330,360]
    windows = list(filter(lambda x: x <= window_min, windows_all))
    
    # List hold feature names
    feature_names = []
    
    # Basic time-of-day and time-of-year for the midpoint of the data
    basic_dt = datetime_params(shelter_timestamps)
    feature_names.extend(list(basic_dt.keys()))
    
    # Direct shelter temperature measurments
    direct_obs = lookback_params(shelter_timestamps, shelter_temps, lookback_min=windows)
    if len(direct_obs) != len(windows):
        raise RuntimeError("Provided shelter data do not span a long enough time range")
    if len(shelter_temps.shape) == 1:
        feature_names.extend(list(direct_obs.keys()))
    else:
        for n in direct_obs.keys():
            for i in range(shelter_temps.shape[1]):
                feature_names.append(f"temp{i}_{n}")
                
    # Sensor differencing
    diff_obs = {}
    if len(shelter_temps.shape) > 1:
        for n in direct_obs.keys():
            for i in range(shelter_temps.shape[1]):
                for j in range(i+1, shelter_temps.shape[1]):
                    diff_obs[f"temp{i}_{j}_diff_{n}"] = direct_obs[n][i] - direct_obs[n][j]
        for n in diff_obs.keys():
            feature_names.append(n)
            
    # Simple temperature analysis over time windows
    windows2 = list(filter(lambda x: x == 30 or x % 60 == 0, windows))
    span_obs = window_params(shelter_timestamps, shelter_temps, windows_min=windows2)
    if len(span_obs) != len(windows2)*4:
        raise RuntimeError("Provided shelter data do not span a long enough time range")
    if len(shelter_temps.shape) == 1:
        feature_names.extend(list(span_obs.keys()))
    else:
        for n in span_obs.keys():
            for i in range(shelter_temps.shape[1]):
                feature_names.append(f"temp{i}_{n}")
                
    # Spectral analysis of the shelter temperature over the last 2 hr
    cycle_obs = cycle_params(shelter_timestamps, shelter_temps, window_min=120)
    if len(cycle_obs) == 0:
        raise RuntimeError("Provided shelter data do not span a long enough time range")
    if len(shelter_temps.shape) == 1:
        feature_names.extend(list(cycle_obs.keys()))
    else:
        for n in cycle_obs.keys():
            for i in range(shelter_temps.shape[1]):
                feature_names.append(f"temp{i}_{n}")
                
    # Convert the wind speed into N-S and E-W vectors
    winds_ns = weather_windspeed*np.cos(weather_winddir * np.pi/180)
    winds_ew = weather_windspeed*np.sin(weather_winddir * np.pi/180)
    
    # Average weather over the last 1 hr
    wx_obs = get_closest_window(shelter_timestamps[-1],
                                weather_timestamps, weather_temp, weather_pressure,
                                weather_humidity, weather_enthalpy, winds_ns, winds_ew,
                                window_min=60)
    if len(wx_obs) == 0:
        raise RuntimeError("Provided weather data no not overlap with shelter data")
    if len(wx_obs[0]) == 0:
        raise RuntimeError("Provided weather data no not overlap with shelter data")    
    wx_obs = [entry.mean() for entry in wx_obs]
    feature_names.extend(['wx_temp', 'wx_press', 'wx_humid', 'wx_enth', 'wx_wind_ns', 'wx_wind_ew'])
    
    # Put it all together
    features = []
    features.extend(basic_dt.values())
    if len(shelter_temps.shape) == 1:
        features.extend(direct_obs.values())
        features.extend(span_obs.values())
        features.extend(cycle_obs.values())
    else:
        for v in direct_obs.values():
            features.extend(v)
        for v in diff_obs.values():
            features.append(v)
        for v in span_obs.values():
            features.extend(v)
        for v in cycle_obs.values():
            features.extend(v)
    features.extend(wx_obs)
    features = np.array(features)
    if return_feature_names:
        return features, feature_names
    else:
        return features


def _build_repr(name, attrs=[]):
    name = '.'.join(name.split('.')[-2:])
    output = "<%s" % name
    first = True
    for key,value in attrs:
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        output += "%s %s=%s" % (('' if first else ','), key, value_str)
        first = False
    output += ">"
    return output


class ShelterTemperatureModel:
    """
    Class that wraps `sklearn.ensemble.RandomForestRegressor` to build a model
    of the shelter temperature.
    """
    
    def __init__(self, model_params: Optional[str]=None):
        self.feature_names = None
        if model_params is None:
            self.model = RandomForestRegressor(n_estimators=300,
                                               random_state=625)
        else:
            with open(model_params, 'rb') as fh:
                data = pkl.load(fh)
                for k,v in data.items():
                    setattr(self, k, v)
                    
    def __repr__(self):
        desc = self.description
        n = self.__class__.__module__+'.'+self.__class__.__name__
        a = []
        for attr in ('ntree', 'nfeature', 'noutput', 'training_score', 'validation_r_sq', 'validation_std'):
            if attr in desc:
                a.append((attr, desc[attr]))
            elif getattr(self, attr, None) is not None:
                a.append((attr, getattr(self, attr)))
        if 'feature_names' in desc:
            a.append(('feature_names', 'yes'))
        return tw_fill(_build_repr(n,a), subsequent_indent='    ')
        
    def fit(self, features: np.ndarray, values: np.ndarray, feature_names: Optional[List[str]]=None):
        """
        Given a 2D array of features (samples x feature) and a 1D array of
        expected values (samples), update the model.
        """
        
        if feature_names is not None:
            if len(features[0]) != len(feature_names):
                raise ValueError("Mismatch between last dimension of 'features' and the number of feature names")
            self.feature_names = feature_names
            
        self.model.fit(features, values)
        
        try:
            del self.validation_r_sq
        except AttributeError:
            pass
        try:
            del self.validation_std
        except AttributeError:
            pass
            
    def _apply_feature_selection(self, features: np.ndarray) -> np.ndarray:
        """
        Apply the stored feature selection to new features.
        """
        
        if self.features_mask is not None:
            if len(features.shape) == 1:
                return features[self.features_mask]
            else:
                return features[:, self.features_mask]
                
        return features
        
    def fit_with_feature_selection(self, features: np.ndarray, values: np.ndarray, feature_names: List[str], threshold=0.0005):
        """
        Given a 2D array of features (samples x feature) and a 1D array of
        expected values (samples), update the model.
        """
        
        # Initial fit with all the data
        self.fit(features, values, feature_names=feature_names)
        
        # Select
        importance = self.model.feature_importances_
        self.features_mask = importance >= threshold
        
        # Prune
        selected_features = self._apply_feature_selection(features)
        selected_feature_names = [n for i,n in enumerate(feature_names) if self.features_mask[i]]
        
        # Re-fit
        self.fit(selected_features, values, feature_names=selected_feature_names)
        
    def predict(self, features: Union[List,np.ndarray]) -> Union[float,np.ndarray]:
        """
        Given a list of features, make a prediction about the current shelter
        temperature.
        """
        
        if isinstance(features, list):
            features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(1,-1)
        
        selected_features = self._apply_feature_selection(features)    
        pred = self.model.predict(selected_features)
        if features.shape[0] == 1:
            pred = pred[0]
            
        return pred
        
    def predict_with_uncertainty(self, features: Union[List,np.ndarray],
                                       confidence: float=0.6827) ->  Tuple[Union[float,np.ndarray],
                                                                           Union[float,np.ndarray],
                                                                           Union[float,np.ndarray]]:
        """
        Given a list of features, make a prediction about the current shelter
        temperature and return that along with the lower and upper bounds of
        the confidence interval set by the `confidence` keyword.
        """
        
        if confidence < 0 or confidence > 1:
            raise ValueError(f"Invalid confidence level: {confidence:.4f}")
            
        if isinstance(features, list):
            features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(1,-1)
            
        selected_features = self._apply_feature_selection(features)
        preds = np.array([tree.predict(selected_features) for tree in self.model.estimators_])
        pred = preds.mean(axis=0)
        lwr = np.percentile(preds, (1 - confidence)/2 * 100, axis=0)
        upr = np.percentile(preds, (1 + confidence)/2 * 100, axis=0)
        if features.shape[0] == 1:
            pred = pred[0]
            lwr = lwr[0]
            upr = upr[0]
            
        return pred, lwr, upr
        
    def score(self, features: np.ndarray, values: np.ndarray) -> float:
        """
        Given a validation set of features and values, return the coefficient
        of determination for the model.
        """
       
        selected_features = self._apply_feature_selection(features) 
        cod = self.model.score(selected_features, values)
        self.validation_r_sq = cod
        self.validation_std = np.sqrt((1 - cod)*np.var(values))
        
        return cod
        
    @property
    def description(self) -> Dict[str,Any]:
        """
        Return a dictionary that descibes various aspects of the model.
        """
        
        desc = {}
        desc['ntree'] = len(self.model.estimators_)
        desc['nfeature'] = self.model.n_features_in_
        desc['noutput'] = self.model.n_outputs_
        try:
            desc['training_score'] = self.model.oob_score_
        except AttributeError:
            pass
        try:
            desc['validation_r_sq'] = self.validation_r_sq
        except AttributeError:
            pass
        try:
            desc['validation_std'] = self.validation_std
        except AttributeError:
            pass
        try:
            desc['feature_names'] = self.feature_names
        except AttributeError:
            pass
        desc['feature_importance'] = self.model.feature_importances_
        return desc
        
    def save(self, model_params: str, overwrite: bool=False):
        """
        Save the current model to the specified filename.
        """
        
        if os.path.exists(model_params):
            if not overwrite:
                raise RuntimeError(f"File '{model_params}' already exists")
                
        with open(model_params, 'wb') as fh:
            r = {'model': self.model, 'feature_names': self.feature_names}
            r['features_mask'] = getattr(self, 'features_mask', None)
            if getattr(self, 'validation_r_sq', None) is not None:
                r['validation_r_sq'] = self.validation_r_sq
            if getattr(self, 'validation_std', None) is not None:
                r['validation_std'] = self.validation_std
                
            pkl.dump(r, fh)
