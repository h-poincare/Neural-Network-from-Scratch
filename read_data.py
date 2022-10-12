import numpy as np
from sklearn import datasets

# load dataset of handwritten digits from sklearn
digits = datasets.load_digits()

# load Xdata
Xdata = digits['data']
# load labels
ydata = digits['target']
np.save("data/digit_images", Xdata)
np.save("data/digit_labels", ydata)


# Xdata = np.load("data/digit_images.npy")
# plt.gray()
# plt.imshow(np.reshape(Xdata[20], [8,8]))
# plt.show()