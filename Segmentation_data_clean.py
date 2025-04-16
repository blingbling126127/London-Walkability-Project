#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn import preprocessing

from natsort import natsorted
from natsort import index_natsorted
import cv2
import numpy as np
import glob
import re


# In[17]:


Final_Result = pd.read_csv('FinalPoints2.csv')
OtherPOI = pd.read_csv('OtherPOI.csv', index_col = 0)
GreenPOI = pd.read_csv('GreenPOI.csv', index_col = 0)
TrafficPOI = pd.read_csv('TrafficFacilties.csv', index_col = 0)




Final_Result = Final_Result.fillna(0)
TrafficPOI = TrafficPOI.fillna(0)
GreenPOI = GreenPOI.fillna(0)
OtherPOI = OtherPOI.fillna(0)


# In[18]:


Final_Result['TARGET_FID'] = Final_Result['filename']
TrafficPOI.reset_index(inplace = True)
GreenPOI.reset_index(inplace = True)
TrafficPOI.reset_index(inplace = True)


# In[19]:


Mer_data = pd.merge(Final_Result,TrafficPOI,on='TARGET_FID',how='left')
Mer_data.head()


# In[20]:


Mer_data2 = pd.merge(Mer_data,GreenPOI,on='TARGET_FID',how='left')
MergeAll = pd.merge(Mer_data2,OtherPOI,on='TARGET_FID',how='left')
MergeAll.fillna(0)
MergeAll.head()


# In[21]:


Level_Data = MergeAll[['filename','Invalid']]
#Safety level

Level_Data['Terrain/Pavement'] = MergeAll[' Id_12']

Level_Data['Road'] = MergeAll[' Id_7']

Level_Data['Obstacles'] = MergeAll[' Id_39']+MergeAll[' Id_33']+ MergeAll[' Id_122']

Level_Data['Traffic Volume'] = MergeAll['Volume']

Level_Data['Traffic Facilties'] = MergeAll['Facilties_Count']

Level_Data['TotalSafety'] =  MergeAll[' Id_12']-MergeAll[' Id_7']-( MergeAll[' Id_39']+MergeAll[' Id_33']+ MergeAll[' Id_122'])+MergeAll['Facilties_Count']

#Comfort Level

Level_Data['Tree+Plant'] = MergeAll[' Id_5'] + MergeAll[' Id_18'] 

Level_Data['Sky'] = MergeAll[' Id_3']

Level_Data['VisualCrowdeness'] = MergeAll[' Id_65'] + MergeAll[' Id_43']

Level_Data['TotalComfort'] =MergeAll[' Id_5'] + MergeAll[' Id_18'] + MergeAll[' Id_3']-(MergeAll[' Id_65'] + MergeAll[' Id_43'])

#Access Level

Level_Data['Station'] = MergeAll['Station']

Level_Data['Canteen'] = MergeAll['Canteen']

Level_Data['Shop'] = MergeAll['SHOP']

Level_Data['Green'] = MergeAll['Join_Count']

Level_Data['TotalAccess'] = MergeAll['Join_Count']+MergeAll['SHOP']+MergeAll['Canteen']+MergeAll['Station']

#Pleasure Level
Level_Data['Landscape'] = MergeAll[' Id_67'] + MergeAll[' Id_105'] 

Level_Data['Water'] = MergeAll[' Id_27'] + MergeAll[' Id_61']

Level_Data['Bench'] = MergeAll[' Id_70']

Level_Data['TotalPleasure'] = MergeAll[' Id_70']+MergeAll[' Id_27'] + MergeAll[' Id_61']+MergeAll[' Id_67'] + MergeAll[' Id_105'] 


# Organize the sheet title 
columns = [('BASIC','filename'),('BASIC','invalid'),('Safety','Terrain/Pavement'),('Safety','Road'),('Safety','Obstacles'),('Safety','Traffic Volume'),('Safety','Traffic Facilties'),('Safety','TotalSafety'),
           ('Comfort','Tree+Plant'),('Comfort','Sky'),('Comfort','VisualCrowdeness'),('Comfort','TotalComfort'),
           ('Accessibility','Station'),('Accessibility','Canteen'),('Accessibility','Shop'),('Accessibility','Green'),('Accessibility','TotalAccess'),
           ('Pleasure','Landscape'),('Pleasure','Water'),('Pleasure','Bench'),('Pleasure','TotalPleasure')]
Level_Data.columns = pd.MultiIndex.from_tuples(columns)

Level_Data.head(20)


# In[22]:


na_locations = Level_Data.isna()

# print all NaN
for column in na_locations.columns:
    if na_locations[column].any():
        print(f"NaN found in column {column} at rows: {na_locations[na_locations[column]].index.tolist()}")


# In[23]:


Level_Data.to_csv('Level_Data2.csv')


# In[26]:


Level_clean = Level_Data[[('BASIC','filename'),('Safety','TotalSafety'), ('Comfort','TotalComfort'), ('Accessibility','TotalAccess'), ('Pleasure','TotalPleasure')]]

Level_clean .columns = Level_clean .columns.get_level_values(0)
Level_clean


# **Split Level by using Natural breaks**

# In[27]:


import jenkspy


# In[12]:


jenkspy.jenks_breaks(Level_clean['Safety'], n_classes=5)


# In[32]:


def normalize_to_levels(column, n_levels=5):
    # Calculate the Jenks natural breaks
    breaks = jenkspy.jenks_breaks(column, n_classes=n_levels)
    
    # The breaks will create n_levels + 1 bins, we categorize each value to a level 1-5
    levels = pd.cut(column, bins=breaks, include_lowest=True, labels=range(1, n_levels+1))
    return levels

last_four_columns = Level_clean.columns[-4:]

# Apply the normalization to each column in the DataFrame
for col in last_four_columns:
    Level_clean[col + '_Level'] = normalize_to_levels(Level_clean[col], n_levels=5)


# In[33]:


Level_clean


# In[34]:


Level_clean.to_csv('Level_clean.csv')


# In[ ]:




