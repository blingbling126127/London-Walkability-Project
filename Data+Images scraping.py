#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn import preprocessing


# **Dealing with DATA**

# In[4]:


get_ipython().run_line_magic('reset', '')


# In[34]:


PT = pd.read_csv('Info_Point.csv', index_col = 0)


# In[35]:


PT.head()


# In[32]:


def ARea_norm(df):
    min_value = abs(df['TQ_GreenspaceSite.Area']).min()
    max_value = abs(df['TQ_GreenspaceSite.Area']).max()
    
    PT['Area_norm']=df['TQ_GreenspaceSite.Area'].apply(
        lambda x: (abs(x)-min_value)/(max_value - min_value))
    return df


# In[42]:


min_value = abs(PT['TQ_GreenspaceSite.Area']).min()
max_value = abs(PT['TQ_GreenspaceSite.Area']).max()


# In[52]:


PT['Area_norm'] = PT['TQ_GreenspaceSite.Area'].apply(lambda x: (10/(max_value - min_value) * (x-1)))


# In[45]:





# In[53]:


PT.sort_values(by="Area_norm")


# In[54]:


PT.to_csv('outPT.csv')


# **Dealing with Geo**

# In[27]:


import os
import requests
import csv
import urllib.request
import re
import pandas as pd
import numpy as np


# In[48]:


GEO = pd.read_csv('Point_add.csv', index_col = 0)
GEO.reset_index(inplace = True)
GEO


# In[65]:


GEO['X'].dtypes


# In[69]:


GEO['TARGET_FID'] = GEO['TARGET_FID'].astype(str)
GEO['X'] = GEO['X'].astype(str)
GEO['Y'] = GEO['Y'].astype(str)
GEO['X']


# In[70]:


length = len(GEO['X'])
length


# In[71]:


GEO.iloc[3][0]


# In[72]:


latitudes = []
longitudes = []
for i in range(0 , length):
    latitude = GEO.iloc[i][2]
    latitudes.append(latitude)
    longitude = GEO.iloc[i][1]
    longitudes.append(longitude)


# In[102]:


GEO[GEO['TARGET_FID'] == str(11622)]


# In[61]:


GEO['TARGET_FID'] = GEO['TARGET_FID'].astype(str)


# **Download all images from google**

# In[74]:


def download(url, name): 
    conn = urllib.request.urlopen(url)
    f = open(name, 'wb') 
    f.write(conn.read())
    f.close() 
    print('Pic Saved!') 


# In[101]:


for i in range(2310, num):
    for j in range(0, 360, 90):
        
        url = "https://maps.googleapis.com/maps/api/streetview?size=600x300&location=" \
          + latitudes[i] + "," + longitudes[i] \
          + "&fov=90&heading=" + str(j) \
          + "&pitch=0" + "&key=AIzaSyDVilxBH9ccXlMKxmEOAzwaKv2o1WrCWr8"
        
        path = os.getcwd()+'\\download\\'
        
        name = path  + GEO.iloc[i][0] + "_" + str(j) + '.jpg'
        
        print (url)
        print(name)
        download(url, name)
        
fp.close()  


# **Stitcher all images into ONE**

# In[3]:


pip install opencv-python


# In[103]:


import cv2
import numpy as np
import glob
import re


# In[42]:


img1 = cv2.imread('download/0_90.jpg')
img2 = cv2.imread('download/0_270.jpg')


# In[43]:


img1


# In[44]:


images = [img1, img2]


# In[45]:


res = np.hstack(images)
cv2.imwrite("res.jpg", res)


# In[104]:


input_path = "download/*.jpg"
list_images = glob.glob(input_path)
list_images.sort(key=lambda l: int(re.findall('\d+', l)[0]),reverse=False) 
print(list_images)


# In[105]:


len_img = len(list_images)


# In[107]:


step = 4 
split_img = [list_images[i:i+step] for i in range(0,len_img,step)]
print(split_img)


# In[108]:


len_split = len(split_img )
len_split


# In[109]:


for i in range(0, len_split):
    index =  split_img[i].pop(-1)
    split_img[i].insert(1,index)

print(split_img)


# In[111]:


split_img[2538]


# In[56]:


arry_split = []
for i in range(0, len_split):
    for j in range(0,2):
        img = cv2.imread(split_img[i][j])
        arry_split.append(img)   


# In[66]:


img1 = cv2.imread(split_img[1][0])
img2 = cv2.imread(split_img[1][1])


# In[64]:


split_img[1][1]


# In[55]:


len_split


# In[67]:


res = np.hstack((img1,img2))
cv2.imwrite("ress.jpg", res)


# In[112]:


for i in range(0, len_split):
    res11 = np.hstack((cv2.imread(split_img[i][0]),cv2.imread(split_img[i][1]),cv2.imread(split_img[i][2]),cv2.imread(split_img[i][3])))
    name = "Images/" + GEO.iloc[i][0] + "_" + ".jpg"
    cv2.imwrite(name , res11)


# In[ ]:





# **Image Segmentation**

# In[ ]:




