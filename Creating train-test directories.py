
# coding: utf-8

# In[1]:


import os
import shutil
import math
import numpy as np
import pickle as pkl
import pandas as pd
import random
from multiprocessing import Pool


# In[2]:


os.listdir("Class wise Data/")
try:
	shutil.rmtree("Class wise Data/train")
	shutil.rmtree("Class wise Data/test")
except:
	os.mkdir("Class wise Data/train")
	os.mkdir("Class wise Data/test")
with open("unique_classes.pkl","rb") as f:
    uc = pkl.load(f)
for i in uc:
    os.mkdir("Class wise Data/train/"+i)
    os.mkdir("Class wise Data/test/"+i)


# In[3]:


split = 0.15
for folder in os.listdir("Class wise Data/"):
    if(folder not in ["train","test"]):
        x = math.ceil(split*len(os.listdir("Class wise Data/"+folder)))
        l = []
        for im in os.listdir("Class wise Data/"+folder):
            l.append(im)
        random.shuffle(l)
        # train copy
        source = "Class wise Data/"+str(folder)+"/"
        dest = "Class wise Data/train/"+str(folder)+"/"
        for item in l[x:]:
            source1=source+str(item)
            shutil.copy(source1,dest)
        # test copy
        dest = "Class wise Data/test/"+str(folder)+"/"
        for item in l[:x]:
            source1=source+str(item)
            shutil.copy(source1,dest)

