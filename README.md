Kanarie
=======
Kanarie is an experiment to see how well machine learning works for modeling
the shelter temperature of an LWA station.  The primary goal of this is to
identify times when the temperatures are anomalous, which would indicate a
failure of the shelter's HVAC units.  A secondary goal is to see if this can
be used to gauge the efficiency of the HVACs for preventative maintenance
purposes.

Approach
--------
Kanarie uses a random forest regressor to predict the current shelter
temperatures using a combination of the recent internal temperatures as well
as weather conditions.  This predicted temperature is then compared against
the actual temperatures and the model's uncertainty to look for anomalies.
If too many anomalies are detected over a certain period of time then this is
taken to be indicative of a problem with the HVACs.

Training
--------
The models are trained on shelter temperature and weather data taken from 2024.
The data consist of 3 hr snapshots that are split into 80% for training and
20% for validation.

Using
-----
The easiest way to use kanarie is through the scripts, particularly the
`kanarie_predict.py` script.  This uses data from the OpScreen pages to provide
a quick ok/possible overheating determination for a station.  The conditions
that lead to a "possible overheating" determination are:
 * The shelter temperature is over 82 F
 * The last three measured temperatures are more than 3*sigma above the
   predictions for either of the shelter's temperature sensors.

Here sigma corresponds to the model's uncertainty that was determined through
validation.


