import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

#importing the data
df = pd.read_csv('data.gitignore/dior_jan_2025_us.csv')

#print(df.head())

#this function takes all of the columns and turns the them into lowercase strings this is just so i avoid mistakes with capitalization
def lower_case_columns(dataframe):
    dataframe.columns = [col.lower() for col in dataframe.columns]
    return dataframe
df = lower_case_columns(df)

#print(df.columns)

#selecting only the columns that are relevant for the analysis
df = df[['order number', 
         'order status',
         #'latest status date local', internal metric not relevant for customer side analysis
         'created status date local', #used in time data section
         'timespan booked_shipped order', #used in time data section
         'fulfilment status',
         'return number', #used to identify returns will change to bool true or false 'return y/n' later
         'return status',
         'return reason en',
         'sub-return reason en',
         'return type',
         'return address – state',
         #'product sku', wont be using sku because we have higher level catagorizatins like unvierse and category
         #'product name', same for name
         'product universe', #It may be interesting to to the category by product universe because there are only 5 and they are general
         'product category',
         'product category',
         'product sub-category',
         'quantity',
         'requested qty',
         'filled qty',
         'rejected qty',
         'shipping cost excl. taxes (local)', #we are not using the coverted values because we want customer side view
         'total taxes (local)',
         'shipping method',
         'shipping address - state',
         'shipping address - gender',
         'is_eligiblepremium delivery',
         'payment type',
         'gift message flag',
         'gift message',
         'orderorreturn',
         'sales type',
         'guest',
         'csc agent id',
         'is csc gift',
         'csc commercial gestures',
         'is pbl order',
         'is stw',
         'is exchange',
         'is c&c',
         'abc dior orders',
         'is_ltsourcing',
         'is_pre-sales',
         'ecopackaging',
         'emptygiftnote',
         'mensualized status',
         'isdeliverywaitandtry',
         'has video gift message',
         'sub-region'
         ]]

#print(df.head())

#This will be where i preprocess the timestamp columns to become something meaningful we can use in a classtifcation

    # Created Status Date Local cleaning

        #This will then be split into 2 different times of day: 10:01pm-5am and 5:01am-10pm
        #This is because late at night people are more likely to make impulse purchases so this may have an effect on return outcome
        #The result is adding a column called is_night_order with boolean values 

print(df['created status date local'].dtype) #checking the datatype of the column, returns object

def is_night_order(created_col):
    # Convert to datetime (ignores invalid formats safely)
    created_dt = pd.to_datetime(created_col, errors='coerce')

    # Extract just the time
    t = created_dt.dt.time

    # Define your cutoff times
    night_start = time(22, 1)  # 10:01 PM
    night_end = time(5, 0)     # 5:00 AM

    # Create boolean mask
    # True if between 10:01PM–11:59PM OR 12:00AM–5:00AM
    is_night = (t >= night_start) | (t <= night_end)

    return is_night

df['is_night_order'] = is_night_order(df['created status date local'])
#print(df[['created status date local', 'is_night_order']].head())
#print(df['is_night_order'].value_counts()) #check if there is enough data
df = df.drop(columns=['created status date local']) #drop original column after creating new one

    # Timespan Booked_Shipped Order cleaning

# I need to take the time from booking the order to shipping and break it down into different classes that can be used for analysis
    # Ill change all of the data to num of days and then use get_dummies at some point later on
df['timespan_days'] = pd.to_timedelta(df['timespan booked_shipped order']).dt.days
print(df['timespan_days'].value_counts().sort_index())

df['timespan_days_less_than_10'] = df['timespan_days'] < 10
df['timespan_days_between_10_20'] = (df['timespan_days'] >= 10) & (df['timespan_days'] < 20)
df['timespan_days_between_20_50'] = (df['timespan_days'] >= 20) & (df['timespan_days'] < 50)
df['timespan_days_between_50_100'] = (df['timespan_days'] >= 50) & (df['timespan_days'] < 100)
df['timespan_days_greater_than_100'] = df['timespan_days'] >= 100

#drop original column after creating new ones
df = df.drop(columns=['timespan booked_shipped order', 'timespan_days'])

#now to check how the new columns look
#for col in ['timespan_days_less_than_10',
            #'timespan_days_between_10_20',
           # 'timespan_days_between_20_50',
           # 'timespan_days_between_50_100',
           # 'timespan_days_greater_than_100']:
    #print(col)
    #print(df[col].value_counts())
    #print('---')

#The check went well so I'll cut this code out

# **This is the end of the time section next I will work on the return section** 


# Return Feature cleaning

#here i will clean 'return number','return status','return reason en','sub-return reason en', 'return type','return address – state',



