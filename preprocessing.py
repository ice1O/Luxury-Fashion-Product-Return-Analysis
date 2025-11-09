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

#print(df['created status date local'].dtype) #checking the datatype of the column, returns object

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
#print(df['timespan_days'].value_counts().sort_index())

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

#retrurn number to boolean
df['is_return'] = df['return number'].notnull()
df = df.drop(columns=['return number']) #drop original column after creating new one

#print(df['is_return'].value_counts())
#we can see around 48215 returns and 48215 non returns which is a good balance for machine learning

#return status cleaning
    #after thinking about what to do with this column i have decided to drop it as it is not relevant to whether or not a product is returned just the status of the return itself
df = df.drop(columns=['return status'])

#return reason en cleaning
    #checking for unique values
#print(df['return reason en'].unique())
    # There are not too many unique values so I wont group them i will just use get dummies later on
    #however after thinking about it the return reason is not usefull in classifying a return because its done after the fact the customer decided to return the product
df = df.drop(columns=['return reason en'])

#same goes for sub-return reason en
df = df.drop(columns=['sub-return reason en'])

#return type cleaning
    #checking for unique values
#print(df['return type'].unique())
    #after looking at the unique values i have decided to drop this column as it is not relevant to whether or not a product is returned just the type of return
    #RTV = "Return to Vendor"
    #LP  = "Loyalty Program Return or Local Pickup Return"
    #RTS = "Return to Stock / Return to Sender / Return to Supplier"
df = df.drop(columns=['return type'])

#return address – state cleaning
    #will be dropped because we have info in the shipping address state and this would be the better indicator of location based returns
df = df.drop(columns=['return address – state'])


# Product Feature cleaning
# here i will clean 'product universe','product category','product sub-category',

#print(df['product universe'].unique())
#print(df['product category'].unique())
    #These have a low enough number of unique values to use get dummies later

#print(df['product sub-category'].unique())
    #We need to group some of these sub-categories together because there are too many unique values

# As I was doing my analysis I discovered there were some missing values with no subgroup so here is where I fx them

# Identify which rows are missing
mask = df["product sub-category"].isna()

# Fill missing sub-categories with values from 'Product Category'
df.loc[mask, "product sub-category"] = df.loc[mask, "product category"]

# Get the unique filled values
filled_values = df.loc[mask, "product sub-category"].unique()

#print(filled_values)
#Now i insert these filled values into the mapping below so they get grouped correctly
#DONE

#mapping 
group_map = {
    # Apparel (Clothing)
    "JACKETS": "Apparel (Clothing)",
    "COATS": "Apparel (Clothing)",
    "PEACOATS": "Apparel (Clothing)",
    "OUTERWEAR": "Apparel (Clothing)",
    "SWEATERS": "Apparel (Clothing)",
    "PANTS": "Apparel (Clothing)",
    "JEANS": "Apparel (Clothing)",
    "SKIRT": "Apparel (Clothing)",
    "SHIRTS": "Apparel (Clothing)",
    "TOPS": "Apparel (Clothing)",
    "TSHIRTS": "Apparel (Clothing)",
    "DRESSES": "Apparel (Clothing)",
    "DAY DRESSES": "Apparel (Clothing)",
    "EVENING DRESSES": "Apparel (Clothing)",
    "COCKTAIL DRESSES": "Apparel (Clothing)",
    "WAISTCOATS": "Apparel (Clothing)",
    "VESTS": "Apparel (Clothing)",
    "SUITS": "Apparel (Clothing)",
    "LEATHER RTW": "Apparel (Clothing)",
    "UNDERWEAR": "Apparel (Clothing)",
    "BEACHWEAR": "Apparel (Clothing)",
    "UNDERWEAR BEACHWEAR": "Apparel (Clothing)",
    "JERSEY": "Apparel (Clothing)",
    "BOTTOMS": "Apparel (Clothing)",
    "KNIT": "Apparel (Clothing)",

    # Footwear
    "BOOTS": "Footwear",
    "SNEAKERS": "Footwear",
    "RUNNERS": "Footwear",
    "FLATS": "Footwear",
    "LOAFERS": "Footwear",
    "LACE UPS": "Footwear",
    "MONKS": "Footwear",
    "BALLERINAS": "Footwear",
    "PUMPS": "Footwear",
    "CHUNKIES": "Footwear",
    "SKATERS": "Footwear",
    "SANDALS": "Footwear",

    # Bags & Small Leather Goods (SLG)
    "HANDBAGS": "Bags & SLG",
    "CROSS BODY BAGS": "Bags & SLG",
    "CHEST BAGS": "Bags & SLG",
    "BACKPACKS": "Bags & SLG",
    "BACK PACKS": "Bags & SLG",
    "SLG BAGS": "Bags & SLG",
    "CLASSIC SLG": "Bags & SLG",
    "PURE SLG": "Bags & SLG",
    "CASES": "Bags & SLG",
    "DIAPER BAGS": "Bags & SLG",
    "TRAVEL": "Bags & SLG",
    "W BAGS": "Bags & SLG",
    "M BAGS": "Bags & SLG",

    # Jewellery & Timepieces
    "COSTUME JEWELLERY": "Jewellery & Timepieces",
    "ACCESS JEWELLERY": "Jewellery & Timepieces",
    "MEDIUM JEWELLERY": "Jewellery & Timepieces",
    "PRECIOUS TIMEPIECES": "Jewellery & Timepieces",
    "ACCESS TIMEPIECES": "Jewellery & Timepieces",

    # Accessories (incl. Eyewear)
    "BELTS": "Accessories",
    "TIES & SCARVES": "Accessories",
    "TIES AND SCARVES": "Accessories",
    "HATS": "Accessories",
    "TEXTILE ACCESSORIES": "Accessories",
    "TEXTILES ACCESSORIES": "Accessories",
    "LEATHER GLOVES": "Accessories",
    "STRAPS": "Accessories",
    "STRAPS & BUCKLES": "Accessories",
    "OTHER ACCESSORIES": "Accessories",
    "UMBRELLAS": "Accessories",
    "EYEWEAR": "Accessories",
    "M SLG": "Accessories",

    # Baby & Kids
    "BABY CARE CLOTHES": "Baby & Kids",
    "BABY CARE ACCESSORIES": "Baby & Kids",
    "BABY CARE OTHERS": "Baby & Kids",
    "STROLLERS": "Baby & Kids",
    "PRAMS": "Baby & Kids",
    "BABY": "Baby & Kids",

    # Home & Lifestyle
    "TABLEWARE": "Home & Lifestyle",
    "FURNITURE": "Home & Lifestyle",
    "HOME LINENS": "Home & Lifestyle",
    "TOYS": "Home & Lifestyle",
    "OBJECTS": "Home & Lifestyle",
    "MAISON": "Home & Lifestyle",

    # Sports & Innovative
    "SPORTS GEARS": "Sports & Innovative",
    "TENNIS": "Sports & Innovative",
    "BASKETBALLS 20S": "Sports & Innovative",
    "BASKETBALLS 80S": "Sports & Innovative",
    "INNOVATIVE FUNCTIONS": "Sports & Innovative",
    "SPECIAL PRODUCTS": "Sports & Innovative",
}

# --- 2) Normalize the text before mapping ---
def normalize_value(x):
    if pd.isna(x):
        return x
    s = str(x).upper().strip()
    while "  " in s:  # collapse double spaces
        s = s.replace("  ", " ")
    return s

# --- 3) Apply mapping and create new column ---
df["product sub category grouped"] = (
    df["product sub-category"]
    .apply(normalize_value)
    .map(group_map)
)

print(df["product sub category grouped"].value_counts())
print(df["product sub category grouped"].head(100))
print(df["product sub category grouped"].isna().sum())

#After checking how many did not get mapped only 4 are still NA and out of 150,000 rows i can live with that so i will drop them later on when I makesure all data if filled for final clean 

