import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
import warnings; warnings.simplefilter('ignore') # Fix Numpy issue
from sklearn.cluster import MiniBatchKMeans
#%matplotlib inline

sampleImage = load_sample_image("") # Image path

ax = plt.axes(xticks=[], yticks=[])
ax.imshow(sampleImage)
sampleImage.shape

data = sampleImage/255.0 # Use 0...1 scale
data = data.reshape(427*640, 3)
data.reshape

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # Choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
    ax[1].scatter(R, G, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)

plot_pixels(data, title="Input Color Space")

kmeans = MiniBatchKMeans(16) # Number of color clusters = 16
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title='Reduced Color Space')

sampleImage_recolored = new_colors.reshape(sampleImage.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(sampleImage)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(sampleImage_recolored)
ax[1].set_title('Color Compressed Image', size=16)