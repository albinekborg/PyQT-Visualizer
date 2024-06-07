
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pyqtgraph as pg
import json
import os

theme_directory = f"./themes"


class ColorMap:
    def __init__(self, theme_name):
        """
        Initialize the custom colormap.

        Parameters:
            matplotlib_name (str): Name of the base colormap to use from matplotlib.
            gradient_offset (int): Number of entries to represent pure white.
        """
        # Try: read json, else: default:theme_json
        
        self.theme_name = theme_name

        try:
            with open(os.path.join(theme_directory,self.theme_name + ".json")) as f:
                theme_dict = json.load(f)

            print(theme_dict)
        except:
            print("Error loading theme. Using Default")
            theme_dict = {"use_mpl": "True",
                            "matplotlib_theme": "summer",
                            "gradient_offset": 100,
                            "border_color": "#4baf4b",
                            "background_color": "#FFFFFF"}

        self.use_mpl = theme_dict["use_mpl"]
        self.gradient_offset = theme_dict["gradient_offset"]
        self.border_color = theme_dict["border_color"]
        self.background_color = theme_dict["background_color"]
    
        if eval(self.use_mpl):
            self.matplotlib_name = theme_dict["matplotlib_theme"]

        try:
            self.custom_gradient = theme_dict["custom_gradient"] # dict of two lists!
        except: pass

        self.pg_colormap = self.generate_pg_colormap()

    def generate_pg_colormap(self):
        """ Generate a colormap compatible with pyqtgraph from a given matplotlib colormap. """
        # Load the original colormap
        
        if eval(self.use_mpl):
            original_cmap = plt.cm.get_cmap(self.matplotlib_name)
        else:
            gradient_positions = self.custom_gradient["positions"]
            gradient_colors = self.custom_gradient["colors"]
            norm_gradient_pos = [p / 100 for p in gradient_positions]
            original_cmap = LinearSegmentedColormap.from_list("custom_gradient",list(zip(norm_gradient_pos,gradient_colors)))
        
        # Define the number of colors in the final colormap
        num_colors = 256

        # Create an array of colors from the original colormap
        colors = original_cmap(np.linspace(0, 1, num_colors))

        # Number of entries for pure white
        
        
        if self.gradient_offset != 0:

            # Number of entries in the transition                                   
            num_transition = 50

            # Create the offset section
            white_section = np.tile([1, 1, 1, 1], (self.gradient_offset, 1))  # RGBA for white

            # Generate a gradient from white to the first color of the original colormap
            first_color = colors[0]
            white_to_color_gradient = np.vstack([np.linspace(1, first_color[i], num_transition) for i in range(4)]).T

            # Combine the white section, the gradient, and the original colors
            blended_colors = np.vstack((white_section, white_to_color_gradient, colors))

            # Create a new colormap from the array of blended colors
            new_cmap = LinearSegmentedColormap.from_list("extended_white_" + self.theme_name, blended_colors)
        
        else: 
            new_cmap = LinearSegmentedColormap.from_list(self.theme_name, colors)

        # Convert to pyqtgraph format: scale to 0-255 and cast to integers
        positions = np.linspace(0, 1, new_cmap.N)
        rgb_values = (new_cmap(np.linspace(0, 1, new_cmap.N))[:, :3] * 255).astype(int)

        # Return a pyqtgraph colormap
        pg_cmap = pg.ColorMap(positions, rgb_values)

        return pg_cmap

class Theme():
    def __init__(self,theme_name = None):

        if theme_name:
            self.colormap = ColorMap(theme_name)
        else: 
            self.colormap = ColorMap(None)

    def get_cmap(self):
        return self.colormap
    
    def get_border_color(self):
        return self.colormap.border_color
    
    def get_background_color(self):
        return self.colormap.background_color

    def get_lookup_table(self, num_colors=256):
        """ Return a lookup table compatible with PyQt5. """
        return self.colormap.pg_colormap.getLookupTable(nPts=num_colors)
    
    def get_pg_colormap(self):
        """ Return the full pyqtgraph colormap object. """
        return self.colormap.pg_colormap
    #TODO: Add support for custom themes via json in theme_directory
