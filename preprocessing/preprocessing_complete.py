import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', None)

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
         #'fulfilment status', not relevant for customer side analysis and lots of misisng dirty data
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
         #'requested qty', dropped
         #'filled qty', dropped
         #'rejected qty', dropped
        #'shipping cost excl. taxes (local)', #we are not using the coverted values because we want customer side view #sales including taxes is a better proxy for customer spending
         #'total taxes (local)',
         'sales incl. taxes (local)', #only sale number we use because is total customer POV
         #'shipping method', dropping because it leaks info on returns
         'shipping address - state',
         'shipping address - gender',
         'is_eligiblepremium delivery', # already bool
        # 'payment type', only for info on non return we dont want info leaking
         'gift message flag', #already a boolean for gift message
         #'gift message',
         'sales type', #make dummies later
         'guest', #already bool
         #'csc agent id',
         #'is csc gift', info leaking
         'csc commercial gestures',# make into bool
         #'is pbl order', info leaking
         #'is stw', info leaking
         'is exchange', #already bool
         #'is c&c', info leaking
         'abc dior orders', #already bool
         'is_ltsourcing', #make into bool
         #'is_pre-sales', #already bool 
         'ecopackaging',
         #'emptygiftnote',  not used in USA
         #'mensualized status', not used in USA
         #'isdeliverywaitandtry', not used in USA
         #'has video gift message', not used in USA
         #'sub-region' just USA in this case
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

#print(df["product sub category grouped"].value_counts())
#print(df["product sub category grouped"].head(100))
#print(df["product sub category grouped"].isna().sum())

#After checking how many did not get mapped only 4 are still NA and out of 150,000 rows i can live with that so i will drop them later on when I makesure all data if filled for final clean 

#Quantity cleaning - we will be cleaning 'quantity', 'requested qty', 'filled qty', 'rejected qty',
#After looking at the data the main thing we can get meaning from is the absolute value of the quanity colum
#When there is a return the filled and rejected are missing so they are not usedfull for the analysis
#so we will take that and drop the rest 

#we want the abs so we get the qaunity info without leaking the info about returns
df['quantity'] = df['quantity'].abs()

#here we will clean sales including taxes

#first to check to max and min values and we'll once again take abs and make price bands to keep computing less intense
#print(df['sales incl. taxes (local)'].describe())
#the max is 54375 we'll make capital bands of 0-500, 500-2000, 2000-4000, 4000-6000, 6000-10000, 10000-20000, 20000-50000, 50000+
# take abs
df['sales_incl_taxes_abs'] = df['sales incl. taxes (local)'].abs()

# define your bands
bins = [0, 500, 2000, 4000, 6000, 10000, 20000, 50000, np.inf]
labels = [
    "0_500",
    "500_2000",
    "2000_4000",
    "4000_6000",
    "6000_10000",
    "10000_20000",
    "20000_50000",
    "50000_plus"
]

# 3. create band variable
df['sales_band'] = pd.cut(
    df['sales_incl_taxes_abs'],
    bins=bins,
    labels=labels,
    right=False
)

# 6. drop the original columns
df = df.drop(columns=['sales incl. taxes (local)', 'sales_incl_taxes_abs'])

#print(df['sales_band'].head())

#check for NA values in the new columns created
#print(df['sales_band'].isna().sum())
#there are only 4 NA values and i have the assumption it is the same 4 as above so these will just be dropped in final cleaning

# Next we will clean  'shipping method', 'shipping address - state', 'shipping address - gender'
    #checking unique values
#print(df['shipping method'].unique(),
     # df['shipping address - state'].unique(),
    #  df['shipping address - gender'].unique()
     # )

#print(df['shipping method'].isna().sum(), # because we have 48219 N/A for the shipping method and this is a large chunck something is up
    #  df['shipping address - state'].isna().sum(),
     # df['shipping address - gender'].isna().sum() # There are 404999 missing values but this could have an effect on return rates so we will keep it and find a way to clean it
#)

#All of the returns are missing information for the shipping method which was used and so this info does not leak into out answer we will drop this column


#shipping address - state we will keep as location may have an effect on return rates we'll make it into masks latter on

#Shipping address - gender
#after looking at the collection method I will fill all of the missing values with rathernotsay
df["shipping address - gender"] = df["shipping address - gender"].fillna("rathernotsay")    
#print(df["shipping address - gender"].sample(100))
#checking unique values again
#print(df['shipping address - gender'].unique())

#making into booleans
#print(df['csc commercial gestures'].unique())
df['csc commercial gestures'] = df['csc commercial gestures'].notna()

#print(df['is_ltsourcing'].unique())
df['is_ltsourcing'] = df['is_ltsourcing'].notna()

#This is the last group to check for the first step of preproccessing
#we will check 'ecopackaging','emptygiftnote','mensualized status','isdeliverywaitandtry', 'has video gift message', 'sub-region'
#eco pakaging
#print(df['ecopackaging'].unique())
#all the NA are the ones which ares upposed to be eco packaging so we will fillna with 'eco packaging'
df['ecopackaging'] = df['ecopackaging'].fillna('ECO')

#After consulting with the source of my data some of the columns in this section are not relevant for USA data so I will drop them
 #include menstrilzed status, empyt gift note, isdeliverywaitandtry, has video gift message, sub-region

# This is the end of the precleaning now we need to deal with missing values, and make bools

#missing values
#print(df.dtypes)
#print(df.isna().sum())

#decided to dope fulfilment statue because of loads of missing dirty data and the fact that there were no non fullments in the return data
#The only columns with an alarming amount are 'is-presales' and 'quantity'
# such an insignificant amount of data is coming from pre-sales 90/around 150k and we dont know how to fill missing data so we will drop it

# for quanitiy we will just drop the observations with missing values

#Data frame is now
df = df.dropna()

# there was also or two wierd values with **** we will drop these
df = df[
    (df['shipping address - gender'] != '****') &
    (df['shipping address - state']  != '****')
]

# now i need to change the sales_band to bool, choose a grouping (we'll use our own because its rational and cool), then check unique values in each

band_dummies = pd.get_dummies(df['sales_band'], prefix='sales_band').astype(bool)
df = pd.concat([df, band_dummies], axis=1)
df = df.drop(columns=['sales_band'])


# Choosing grouping - we will used the grouped sub catagories which we made outselves (we are the best)
df=df.drop(columns=['product universe','product category','product sub-category'])



# now we make a bool for the sub catagory
subcat_dummies = (
    pd.get_dummies(
        df['product sub category grouped'], 
        prefix='product_subcategory'
    ).astype(bool)
)
# Add them to the dataframe
df = pd.concat([df, subcat_dummies], axis=1)
# Drop the original column
df = df.drop(columns=['product sub category grouped'])

#Still need to turn 'shipping adress-gender', 'shipping adress-state', 'sales type', 'quantity' into bools
#Also eco packaging into a bool

df['ecopackaging'] = df['ecopackaging'].map({'ECO': True, 'STD': False})

cols_to_dummy = [
    'shipping address - gender',
    'shipping address - state',
    'sales type',
    'quantity'
]

for col in cols_to_dummy:
    dummies = pd.get_dummies(df[col], prefix=col, dtype=bool)
    df = pd.concat([df, dummies], axis=1)

df = df.drop(columns=cols_to_dummy)

df = df[
    [
        # 1. IDENTIFIERS
        'order number',
        'order status',

        # 2. TARGET VARIABLE
        'is_return',

        # 3. CUSTOMER / ORDER CONTEXT FEATURES
        'guest',
        'is_eligiblepremium delivery',
        'gift message flag',
        'csc commercial gestures',
        'is exchange',
        'abc dior orders',
        'is_ltsourcing',
        'ecopackaging',

        # 4. SALES BAND FEATURES
        'sales_band_0_500',
        'sales_band_500_2000',
        'sales_band_2000_4000',
        'sales_band_4000_6000',
        'sales_band_6000_10000',
        'sales_band_10000_20000',
        'sales_band_20000_50000',
        'sales_band_50000_plus',

        # 5. PRODUCT SUBCATEGORY FEATURES
        'product_subcategory_Accessories',
        'product_subcategory_Apparel (Clothing)',
        'product_subcategory_Baby & Kids',
        'product_subcategory_Bags & SLG',
        'product_subcategory_Footwear',
        'product_subcategory_Home & Lifestyle',
        'product_subcategory_Jewellery & Timepieces',
        'product_subcategory_Sports & Innovative',

        # 6. SHIPPING ADDRESS - GENDER DUMMIES
        'shipping address - gender_miss',
        'shipping address - gender_mr',
        'shipping address - gender_mrs',
        'shipping address - gender_ms',
        'shipping address - gender_mx',
        'shipping address - gender_rathernotsay',

        # 7. SHIPPING ADDRESS - STATE DUMMIES
        'shipping address - state_AK',
        'shipping address - state_AL',
        'shipping address - state_AR',
        'shipping address - state_AZ',
        'shipping address - state_CA',
        'shipping address - state_CO',
        'shipping address - state_CT',
        'shipping address - state_DC',
        'shipping address - state_DE',
        'shipping address - state_FL',
        'shipping address - state_GA',
        'shipping address - state_HI',
        'shipping address - state_IA',
        'shipping address - state_ID',
        'shipping address - state_IL',
        'shipping address - state_IN',
        'shipping address - state_KS',
        'shipping address - state_KY',
        'shipping address - state_LA',
        'shipping address - state_MA',
        'shipping address - state_MD',
        'shipping address - state_ME',
        'shipping address - state_MI',
        'shipping address - state_MN',
        'shipping address - state_MO',
        'shipping address - state_MS',
        'shipping address - state_MT',
        'shipping address - state_NC',
        'shipping address - state_ND',
        'shipping address - state_NE',
        'shipping address - state_NH',
        'shipping address - state_NJ',
        'shipping address - state_NM',
        'shipping address - state_NV',
        'shipping address - state_NY',
        'shipping address - state_OH',
        'shipping address - state_OK',
        'shipping address - state_OR',
        'shipping address - state_PA',
        'shipping address - state_RI',
        'shipping address - state_SC',
        'shipping address - state_SD',
        'shipping address - state_TN',
        'shipping address - state_TX',
        'shipping address - state_UT',
        'shipping address - state_VA',
        'shipping address - state_VT',
        'shipping address - state_WA',
        'shipping address - state_WI',
        'shipping address - state_WV',
        'shipping address - state_WY',

        # 8. SALES TYPE DUMMIES
        'sales type_ALL',
        'sales type_DIRECT',
        'sales type_ON BEHALF',
        'sales type_STW',

        # 9. QUANTITY DUMMIES
        'quantity_0.0',
        'quantity_1.0',
        'quantity_2.0',
        'quantity_3.0',
        'quantity_4.0',
        'quantity_5.0',
        'quantity_6.0',
        'quantity_7.0',
        'quantity_8.0',
        'quantity_9.0',
        'quantity_10.0',
        'quantity_11.0',
        'quantity_12.0',
        'quantity_13.0',
        'quantity_14.0',
        'quantity_16.0',
        'quantity_24.0',
        'quantity_30.0',
        'quantity_49.0',
        'quantity_65.0',

        # 10. TIME FEATURES
        'is_night_order',
        'timespan_days_less_than_10',
        'timespan_days_between_10_20',
        'timespan_days_between_20_50',
        'timespan_days_between_50_100',
        'timespan_days_greater_than_100',
    ]
]
df.to_csv("cleaned_dior_jan_2025_us_with_dummies.csv", index=False)



