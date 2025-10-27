Problem Overview
This is an electric transformer hour-by-hour dataset. We provide you with the data of 2 transformers.
Dataset
The dataset contains 6 features per time point which is as follows HUFL = High Voltage Useful Load (active power on high voltage side) HULL = High Voltage Useless Load (reactive power on high voltage side) MUFL = Medium Voltage Useful Load (active power on medium voltage side) MULL = Medium Voltage Useless Load (reactive power on medium voltage side) LUFL = Low Voltage Useful Load (active power on low voltage side) LULL = Low Voltage Useless Load (reactive power on low voltage side) OT = Oil Temperature (the target variable to predict)
Task Description:
You need to predict the oil temperature of a time point based on data from previous time points without ever using oil temperature information from any time point
We want you to try three configurations based on the usage of data from previous time points: (1) use past N (you can tune this) time point data starting from the immediate previous time point of the one to be forecasted for - you are trying to predict oil temperature for the next hour (2) use past N (you can tune this) time point data starting from the 6 time points prior to the one to be forecasted for - you are trying to predict what will happen to oil temperature after 1 day  (3) use past N (you can tune this) time point data starting from the 24 time points prior to the one to be forecasted for - you are trying to predict what will happen to oil temperature after 1 week Note:
You cannot use any load information from the time point for which you are predicting the oil temperature for in any of the above configurations
You cannot use oil temperature from any time point
Train-Test Data:
Your training data should be completely disjoint from your test data
If you are using time point 1-50,000 in any of your training samples, then the test samples cannot belong to any of those time points
My recommendation is to divide the data into time point groups and randomly assign 80% of the groups to training and 20% to testing
Consider some mean normalization
Perform cross validation on the training dataset to develop and choose your model; the test set should be kept only for testing the final model.
Expectations:
Proper time-series feature visualization
Feature selection attempts through correlation analysis
You are expected to explore both deep learning based time series modeling and traditional ML techniques

