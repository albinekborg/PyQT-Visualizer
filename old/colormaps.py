#import pyaudio



#def setup_audio():
#    gp = pyaudio.PyAudio()
#
#    # Automatically find the BlackHole device index
#    num_devices = gp.get_device_count()
#    print(num_devices)
#    for i in range(num_devices):
#        device_info = gp.get_device_info_by_index(i)
#        print(device_info)
#
#setup_audio()





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the original colormap
original_cmap = plt.cm.get_cmap('summer')

# Define the number of colors in the final colormap
num_colors = 256

# Create an array of colors from the original colormap
colors = original_cmap(np.linspace(0, 1, num_colors))

# Number of entries for pure white
num_white = 100

# Number of entries in the transition
num_transition = 50

# Create the white section
white_section = np.tile([1, 1, 1, 1], (num_white, 1))  # RGBA for white

# Generate a gradient from white to the first color of the original colormap
first_color = colors[0]
white_to_color_gradient = np.vstack([np.linspace(1, first_color[i], num_transition) for i in range(4)]).T

# Combine the white section, the gradient, and the original colors
blended_colors = np.vstack((white_section, white_to_color_gradient, colors))

# Create a new colormap from the array of blended colors
new_cmap = LinearSegmentedColormap.from_list("extended_white_viridis", blended_colors, N=num_white + num_transition + num_colors)

### Test the new colormap
#img = np.random.rand(10, 10)  # Generate some random data
#plt.imshow(img, cmap=new_cmap)  # Display the data using the new colormap
#plt.colorbar()  # Show the color bar
#plt.show()

