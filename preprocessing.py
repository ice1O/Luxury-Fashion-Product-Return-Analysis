import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the data
df = pd.read_csv('dior_jan_2025_us.csv')

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
         'latest status date local',
         'created status date local',
         'timespan booked_shipped order',
         'fulfilment status',
         'return number',
         'return status',
         'return reason en',
         'sub-return reason en',
         'return type',
         'return address – state',
         'product sku',
         'product name',
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

print(df.head())