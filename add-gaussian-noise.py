#!/usr/bin/env python
# coding: utf-8

# # Add noise

# In[1]:


import handcalcs.render
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage


# In[2]:


plt.imshow(mpimg.imread("./input/test1.bmp"))
plt.title("Original image")
plt.show()


# In[3]:


def add_gaussian_noise(load_path, save_path, mean, var):
    original_img = cv2.imread(load_path)
    noisy_img = skimage.util.random_noise(original_img, mode='gaussian', mean=mean, var=var)
    noisy_img = np.array(255*noisy_img, dtype = 'uint8')
    cv2.imwrite(save_path, noisy_img)
    plt.imshow(mpimg.imread(save_path))
    plt.title("Image with "+str(var)+ " noise variance")
    plt.show()
    return 


# In[4]:


add_gaussian_noise("./input/test1.bmp","input/test1noisy0-01-.bmp", 0, 0.01)


# In[5]:


add_gaussian_noise("./input/test1.bmp","input/test1noisy0-05-.bmp", 0, 0.05 )


# In[6]:


add_gaussian_noise("./input/test1.bmp","input/test1noisy0-09-.bmp", 0, 0.09)


# In[7]:


add_gaussian_noise("./input/test1.bmp","input/test1noisy0-3-.bmp", 0, 0.3 )

