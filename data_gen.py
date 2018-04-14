import tifffile as tif
from PIL import Image
import pickle as pkl
import numpy as np
import os

train_x = []
test_x = []
train_y = []
test_y = []

with open("unique_classes.pkl","rb") as f:
    uc = pkl.load(f)

#extracting images for each class and soring them in train_x, test_x
#similarly for each image also create a train_y and test_y arrays with the correspondng class labels in them
g = ['Gnetopsida','Equisetopsida','Ginkgoopsida']

def reshaper(va):
    t = []
    for i in range(va.shape[1]):
        d = []
        for j in range(va.shape[2]):
            l = []
            for k in range(va.shape[0]):
                l.append(va[k][i][j])
            d.append(l)
        t.append(d)
    return np.array(t)
    
for i in g:
    print(i+" train")
    for image in os.listdir("Class wise Data/train/"+i):
        train_x.append(reshaper(np.array(tif.imread("Class wise Data/train/"+i+"/"+image))))
        l = np.zeros((1,len(g)))
        l[0][g.index(i)] = 1
        train_y.append(l)
    print(i+" test")
    for image in os.listdir("Class wise Data/test/"+i):
        test_x.append(reshaper(np.array(tif.imread("Class wise Data/test/"+i+"/"+image))))
        l = np.zeros((1,len(g)))
        l[0][g.index(i)] = 1
        test_y.append(l)        

#converting the lists to numpy array
train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)


test_y = test_y.reshape((test_y.shape[0],len(g)))
train_y = train_y.reshape((train_y.shape[0],len(g)))
#dumping the numpy arrays for future use

with open("train_x.pkl","wb") as f:
	pkl.dump(train_x,f);
with open("test_x.pkl","wb") as f:
	pkl.dump(test_x,f);
with open("train_y.pkl","wb") as f:
	pkl.dump(train_y,f);
with open("test_y.pkl","wb") as f:
	pkl.dump(test_y,f);
