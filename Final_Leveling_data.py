#!/usr/bin/env python
# coding: utf-8

# ## DEALING WITH FEATURE DATA ##

# ***connect two CSV***

# In[2]:


import pandas as pd
import numpy as np
from sklearn import preprocessing

from natsort import natsorted
from natsort import index_natsorted
import cv2
import numpy as np
import glob
import re


# In[13]:


pip install natsort


# In[11]:


Result = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features.csv', index_col = 0)


# In[10]:


Frames = [Feature1,Feature2,Feature3,Feature4,Feature5,Feature6]
Result = pd.concat(Frames)
Result


# In[26]:


Feature6 


# In[13]:


Result .reset_index(inplace = True)


# In[14]:


length=len(Result ['filename'])
length


# In[33]:


Result['filename'][0][:-1]


# In[15]:


for i in range(0,length):
    Result['filename'][i] = Result ['filename'][i][:-1]

Result ['filename']


# In[14]:


Result['filename'].astype(np.int32)


# In[15]:


natsorted(Result["filename"])


# In[16]:


Result_sorted = Result.iloc[natsorted(Result.index, key=lambda x: Result.loc[x, 'filename'])]
Result_sorted.drop_duplicates(subset=['filename'],keep='first',inplace=True)
Result_sorted.reset_index(inplace = True) 
Result_sorted.drop(columns = 'index',inplace = True)
Result_sorted.head(50)


# In[17]:


Result_sorted["Invalid"] = np.where( 
(Result_sorted[' Id_2'] == 0) | (Result_sorted[' Id_3'] == 0)|(Result_sorted[' Id_6'] > 0.02
                                                              ),
    1,
    0
)

Point_invalid = Result_sorted[Result_sorted["Invalid"] == 1]
Point_invalid.head(20)

Point_valid = Result_sorted[Result_sorted["Invalid"] == 0]




# In[20]:


Point_invalid


# In[69]:





# In[23]:


Point_invalid.to_csv('Point_invalid_2.csv')
Point_valid.to_csv('Point_valid_add.csv')


# **EXPORT TO FINAL RESULT**

# In[4]:


Feature1 = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features1.csv', index_col = 0)
Feature2 = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features2.csv', index_col = 0)
Feature3 = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features3.csv', index_col = 0)
Feature4 = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features4.csv', index_col = 0)
Feature5 = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features5.csv', index_col = 0)
Feature6 = pd.read_csv('E:/SemanticSegmentation_v1.0/Data/features6.csv', index_col = 0)


# In[5]:


Frames = [Feature1,Feature2,Feature3,Feature4,Feature5,Feature6]
Result2 = pd.concat(Frames)
Result2.reset_index(inplace = True)


# In[6]:


length2=len(Result2 ['filename'])
length2


# In[7]:


for i in range(0,length2):
    Result2['filename'][i] = Result2 ['filename'][i][:-1]

Result2 ['filename']


# In[8]:


Result_sorted2 = Result2.iloc[natsorted(Result2.index, key=lambda x: Result2.loc[x, 'filename'])]
Result_sorted2.drop_duplicates(subset=['filename'],keep='first',inplace=True)
Result_sorted2.reset_index(inplace = True) 
Result_sorted2.drop(columns = 'index',inplace = True)
Result_sorted2.head(10)


# In[9]:


Result_sorted2["Invalid"] = np.where( 
(Result_sorted2[' Id_2'] == 0) | (Result_sorted2[' Id_3'] == 0)|(Result_sorted2[' Id_6'] > 0.01
                                                              ),
    1,
    0
)

Point_invalid_origin = Result_sorted2[Result_sorted2["Invalid"] == 1]
Point_valid_origin = Result_sorted2[Result_sorted2["Invalid"] == 0]
Point_valid_origin


# In[173]:


Final_Result = pd.concat([Point_valid_origin,Point_valid] )
Final_Result.reset_index(inplace = True)
Final_Result.drop(columns = 'index',inplace = True)


# In[315]:


Final_Result.to_csv('FinalPoints2.csv')


# ## DEALING WITH Level ##

# In[459]:


Level_Data = Final_Result[['filename','Invalid']]

Level_Data


# In[460]:


OtherPOI = pd.read_csv('OtherPOI.csv', index_col = 0)
GreenPOI = pd.read_csv('GreenPOI.csv', index_col = 0)
TrafficPOI = pd.read_csv('TrafficFacilties.csv', index_col = 0)


# In[461]:


TrafficPOI['Volume']


# In[462]:


OtherPOI


# In[463]:


GreenPOI


# **SAFETY**

# In[464]:


Level_Data['Terrain/Pavement'] = Final_Result[' Id_12']

Level_Data['Road'] = Final_Result[' Id_7']

Level_Data['Obstacles'] = Final_Result[' Id_39']+Final_Result[' Id_33']+ Final_Result[' Id_122']

Level_Data['Traffic Volume'] = TrafficPOI['Volume']

Level_Data['Traffic Facilties'] = TrafficPOI['Facilties_Count']


columns = [('BASIC','filename'),('BASIC','invalid'),('Safety','Terrain/Pavement'),('Safety','Road'),('Safety','Obstacles'),('Safety','Traffic Volume'),('Safety','Traffic Facilties')]
Level_Data.columns = pd.MultiIndex.from_tuples(columns)
Level_Data


# **COMFORT**

# In[465]:


Level_Data['Tree+Plant'] = Final_Result[' Id_5'] + Final_Result[' Id_18'] 
Level_Data['Sky'] = Final_Result[' Id_3']

Level_Data['VisualCrowdeness'] = Final_Result[' Id_65'] + Final_Result[' Id_43']
columns = [('BASIC','filename'),('BASIC','invalid'),('Safety','Terrain/Pavement'),('Safety','Road'),('Safety','Obstacles'),('Safety','Traffic Volume'),('Safety','Traffic Facilties'),
           ('Comfort','Tree+Plant'),('Comfort','Sky'),('Comfort','VisualCrowdeness')]
Level_Data.columns = pd.MultiIndex.from_tuples(columns)
Level_Data


# **Accessibilty**

# In[466]:


Level_Data['Station'] = OtherPOI['Station']
Level_Data['Canteen'] = OtherPOI['Canteen']
Level_Data['Shop'] = OtherPOI['SHOP']
Level_Data['Green'] = GreenPOI['Join_Count']


columns = [('BASIC','filename'),('BASIC','invalid'),('Safety','Terrain/Pavement'),('Safety','Road'),('Safety','Obstacles'),('Safety','Traffic Volume'),('Safety','Traffic Facilties'),
           ('Comfort','Tree+Plant'),('Comfort','Sky'),('Comfort','VisualCrowdeness'),
           ('Accessibility','Station'),('Accessibility','Canteen'),('Accessibility','Shop'),('Accessibility','Green')]
Level_Data.columns = pd.MultiIndex.from_tuples(columns)
Level_Data


# **Pleasurability**

# In[467]:


Level_Data['Landscape'] = Final_Result[' Id_67'] + Final_Result[' Id_105'] 
Level_Data['Water'] = Final_Result[' Id_27'] + Final_Result[' Id_61']
Level_Data['Bench'] = Final_Result[' Id_70']

columns = [('BASIC','filename'),('BASIC','invalid'),('Safety','Terrain/Pavement'),('Safety','Road'),('Safety','Obstacles'),('Safety','Traffic Volume'),('Safety','Traffic Facilties'),
           ('Comfort','Tree+Plant'),('Comfort','Sky'),('Comfort','VisualCrowdeness'),
           ('Accessibility','Station'),('Accessibility','Canteen'),('Accessibility','Shop'),('Accessibility','Green'),
           ('Pleasure','Landscape'),('Pleasure','Water'),('Pleasure','Bench')]
Level_Data.columns = pd.MultiIndex.from_tuples(columns)
Level_Data


# In[474]:


Level_Data[('Safety','Traffic Volume')] =Level_Data[('Safety','Traffic Volume')].fillna('0') .astype('int32',errors='ignore')
Level_Data[('Safety','Traffic Facilties')] =Level_Data[('Safety','Traffic Facilties')].fillna('0') .astype('int32',errors='ignore')
Level_Data[('Accessibility','Station')] =Level_Data[('Accessibility','Station')].fillna('0') .astype('int32',errors='ignore')
Level_Data[('Accessibility','Canteen')] =Level_Data[('Accessibility','Canteen')].fillna('0') .astype('int32',errors='ignore')
Level_Data[('Accessibility','Shop')] =Level_Data[('Accessibility','Shop')].fillna('0') .astype('int32',errors='ignore')
Level_Data[('Accessibility','Green')] =Level_Data[('Accessibility','Green')].fillna('0') .astype('int32',errors='ignore')


# In[475]:


Level_Data[('Safety','Traffic Volume')]


# In[476]:


Level_Data


# **Split Level by using Natural breaks**

# In[44]:


pip install jenkspy


# In[45]:


import jenkspy


# In[469]:


Sheet_len = len(Level_Data)
Sheet_len


# In[115]:


Level_Data.iloc[0][2]


# In[470]:


col_len = len(Level_Data.columns)
col_len


# In[471]:


breaks = np.zeros((15,6))
breaks


# In[477]:


Level_Data.iloc[:,6]


# In[478]:


List= jenkspy.jenks_breaks(Level_Data.iloc[:,6],split_class)
List


# In[479]:


for i in range(2,col_len):
    List= jenkspy.jenks_breaks(Level_Data.iloc[:,i],split_class)
    for j in range (0,6):
        breaks[i-2][j] = List[j] 

breaks


# In[ ]:





# In[480]:


breaks[4]


# **Split every cols Level**

# In[248]:





# In[384]:


Level_Data.head(20)


# In[408]:


breaks[6][5]


# In[481]:


Level_Data3 = Level_Data.copy()
for i in range (2,col_len):
    for j in range(0,Sheet_len):
        
        if  (Level_Data.iloc[j,i] < breaks[i-2][1]) & (Level_Data.iloc[j,i] > breaks[i-2][0]) :
             Level_Data3.iloc[j,i] = 1
        elif (Level_Data.iloc[j,i] < breaks[i-2][2]) & (Level_Data.iloc[j,i] >= breaks[i-2][1]) :
            Level_Data3.iloc[j,i] = 2
        elif (Level_Data.iloc[j,i] < breaks[i-2][3]) & (Level_Data.iloc[j,i] >= breaks[i-2][2]) :
            Level_Data3.iloc[j,i] = 3
        elif (Level_Data.iloc[j,i] < breaks[i-2][4]) & (Level_Data.iloc[j,i] >= breaks[i-2][3]) :
            Level_Data3.iloc[j,i] = 4
        elif (Level_Data.iloc[j,i] < breaks[i-2][5]) & (Level_Data.iloc[j,i] >= breaks[i-2][4]):
            Level_Data3.iloc[j,i] = 5
        elif (Level_Data.iloc[j,i] == breaks[i-2][5]):
            Level_Data3.iloc[j,i] = 5
        else : 
            Level_Data3.iloc[j,i] = 0

Level_Data3


# In[482]:


Level_Data3.head(20)


# **DEAL WITH THE REVERSE LEVEL DATA**

# In[483]:


Level_Data4 = Level_Data3.copy()
for i in range (3,6):
    for j in range(0,Sheet_len):
        
        if  (Level_Data.iloc[j,i] < breaks[i-2][1]) & (Level_Data.iloc[j,i] >= breaks[i-2][0]) :
             Level_Data4.iloc[j,i] = 5
        elif (Level_Data.iloc[j,i] < breaks[i-2][2]) & (Level_Data.iloc[j,i] >= breaks[i-2][1]) :
            Level_Data4.iloc[j,i] = 3
        elif (Level_Data.iloc[j,i] < breaks[i-2][3]) & (Level_Data.iloc[j,i] >= breaks[i-2][2]) :
            Level_Data4.iloc[j,i] = 1
        elif (Level_Data.iloc[j,i] < breaks[i-2][4]) & (Level_Data.iloc[j,i] >= breaks[i-2][3]) :
            Level_Data4.iloc[j,i] = -3
        elif (Level_Data.iloc[j,i] < breaks[i-2][5]) & (Level_Data.iloc[j,i] >= breaks[i-2][4]):
            Level_Data4.iloc[j,i] = -5
        elif (Level_Data.iloc[j,i] == breaks[i-2][5]):
            Level_Data4.iloc[j,i] = -5
        else : 
            Level_Data3.iloc[j,i] = 5

Level_Data4


# In[484]:


Level_Data5 = Level_Data4.copy()
for i in range (9,10):
    for j in range(0,Sheet_len):
        
        if  (Level_Data.iloc[j,i] < breaks[i-2][1]) & (Level_Data.iloc[j,i] > breaks[i-2][0]) :
             Level_Data5.iloc[j,i] = 5
        elif (Level_Data.iloc[j,i] < breaks[i-2][2]) & (Level_Data.iloc[j,i] >= breaks[i-2][1]) :
            Level_Data5.iloc[j,i] = 3
        elif (Level_Data.iloc[j,i] < breaks[i-2][3]) & (Level_Data.iloc[j,i] >= breaks[i-2][2]) :
            Level_Data5.iloc[j,i] = 1
        elif (Level_Data.iloc[j,i] < breaks[i-2][4]) & (Level_Data.iloc[j,i] >= breaks[i-2][3]) :
            Level_Data5.iloc[j,i] = -3
        elif (Level_Data.iloc[j,i] < breaks[i-2][5]) & (Level_Data.iloc[j,i] >= breaks[i-2][4]):
            Level_Data5.iloc[j,i] = -5
        elif (Level_Data.iloc[j,i] == breaks[i-2][5]):
            Level_Data5.iloc[j,i] = -5
        else : 
            Level_Data5.iloc[j,i] = 5
Level_Data5


# In[422]:


Level_Data5.columns


# In[485]:


Level_Data5


# In[494]:


Level_Data5['Satety_Level'] = (Level_Data5[(       'Safety',  'Terrain/Pavement')] + Level_Data5[(       'Safety',              'Road')] + Level_Data5[(       'Safety',         'Obstacles')]  + Level_Data5[(       'Safety', 'Traffic Facilties')])/4

Level_Data5['Comfort_Level'] = (Level_Data5[(      'Comfort',        'Tree+Plant')]+ Level_Data5[(      'Comfort',               'Sky')]+ Level_Data5[(      'Comfort',  'VisualCrowdeness')])/3
        
Level_Data5['Accessibility_Level'] = (Level_Data5[('Accessibility',           'Station')]+ Level_Data5[('Accessibility',           'Canteen')]+ Level_Data5[('Accessibility',              'Shop')]+ Level_Data5[('Accessibility',             'Green')])/4

Level_Data5['Pleasure_Level']      = (Level_Data5[(     'Pleasure',         'Landscape')]+ Level_Data5[(     'Pleasure',             'Water')]+ Level_Data5[(     'Pleasure',             'Bench')])/3
         


# In[495]:


Level_Data5


# In[496]:


Level_Data6 = Level_Data5.copy()

Level_Data6['Total_Score'] = Level_Data6['Satety_Level']+ Level_Data6['Comfort_Level'] + Level_Data6['Accessibility_Level'] + Level_Data6['Pleasure_Level']
Level_Data6['Weight_Score'] = (0.4 * Level_Data6['Satety_Level']) + (0.4* Level_Data6['Comfort_Level']) + (0.1* Level_Data6['Accessibility_Level']) + 0.1*(Level_Data6['Pleasure_Level'])
Level_Data6


# In[497]:


Level_Data6.to_csv('LevelData_Final.csv')


# In[ ]:




