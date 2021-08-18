#Sample data transformation steps for extracting features for a risk score generation in pregnancies

import pandas as pd
import numpy as np
import seaborn as sns
import pyarrow
import datetime
import warnings
pd.set_option('max_rows', 500)
warnings.filterwarnings("ignore")

#df = read_csv('sample.csv') Load the form data and cases from the csv file that were exported from CommCare

#Tranform the raw data, Eg : risk score generation for pregnancies where preg_outcome and mother_outcome refer to outcome of the pregnancy wrt mother and child
#This step is not needed for test or real-time tracked cases since the outcome of their pregnancy is yet to be determined
df['mortality'] = 1
df.loc[df['preg_outcome'].str.contains('live_birth') | df['mother_outcome'].str.contains('maternal_dead'), 'mortality'] = 0

#Extract time features from CommCare metadata features such as completed_time, received_on_server etc
df['completed_time'] = pd.to_datetime(df['completed_time'])
df['completed_date'] = df['completed_time'].dt.date
df['completed_hour'] = df['completed_time'].dt.hour
df['completed_month'] = df['completed_time'].dt.month
df['completed_weekday_name'] = df['completed_time'].dt.dayofweek
df['completed_year'] = df['completed_time'].dt.year

#Calculate secondary metdata featues such a total time in application
df['time_in_app'] = df.groupby('case_name')['form_duration'].transform(sum)
df['mean_form_duration'] = df.groupby('case_name')['form_duration'].transform(np.mean)
df['median_form_duration'] = df.groupby('case_name')['form_duration'].transform(np.median)

#Calculate the Estimated Delivery Date and days from the Estimated Delivery Date 
df['edd'] = pd.to_datetime(df['edd'],errors = 'coerce')
df['days_from_edd']= (df['edd']-df['received_on']).dt.days

#Other common interventional features include 'tetanus_1','tetanus_2','takes_ifa','takes_nutrition','prep_for_cost','inst_delivery_plan'