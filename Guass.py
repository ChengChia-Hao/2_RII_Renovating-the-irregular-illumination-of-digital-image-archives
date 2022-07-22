# import numpy as np
# import matplotlib.pyplot as plt

# #3*3 Gassian filter
# x, y = np.mgrid[-1:2, -1:2]
# gaussian_kernel = np.exp(-(x**2+y**2))

# #Normalization
# gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

# plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
# plt.colorbar()
# plt.show()

# -*- coding: utf-8 -*-
"""
Creat 3*3 Gassian Kernel
"""
import numpy as np
import matplotlib.pyplot as plt

#3*3 Gassian filter
x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp(-(x**2+y**2))

#Normalization
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()

"""
Convolve 3*3 Gassian Kernel with Image
"""

import numpy as np
import cv2
from scipy import signal
from scipy import misc

image = mpimg.imread('noisy_view.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #¦Ç¶¥
plt.imshow(gray, cmap=plt.get_cmap('gray'))

grad = signal.convolve2d(gray, gaussian_kernel, boundary='symm', mode='same') #¨÷¿n
plt.imshow(grad, cmap=plt.get_cmap('gray'))