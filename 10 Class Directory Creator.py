
# coding: utf-8

# In[10]:


import os
import shutil
import pickle as pkl
import pandas as pd
from multiprocessing import Pool


# In[11]:


with open("unique_classes.pkl","rb") as f:
    uc = pkl.load(f)


# In[12]:


shutil.rmtree("Class wise Data")
os.mkdir("Class wise Data")
for i in uc:
    os.mkdir("Class wise Data/"+i)


# In[13]:


df = pd.read_csv("occurrences_train.csv")


# In[14]:


li = []
for i in range(df.shape[0]):
    im_id = df['patch_id'][i]
    im_dir = df['patch_dirname'][i]
    cls = df['class'][i]
    li.append((im_dir,im_id,cls))


# In[15]:


i = 0

def f(l):
    global i
    i+=1
    p1 = l[0]
    p2 = l[1]
    p3 = l[2]
    source = "patchTrain/"+str(p1)+"/patch_"+str(p2)+".tif"
    dest = "Class wise Data/"+str(p3)+"/"
    shutil.copy(source,dest)
    if(i%100 == 0): print(i)


# In[16]:


with Pool(4) as p:
    p.map(f,li)

