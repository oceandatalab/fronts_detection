from matplotlib import colors
import os


def load_cmap(directory: str, name: str):
    """load colormap"""
    colors_list = []
    file_cmap = open(os.path.join(directory, name), 'r')
    for line in file_cmap:
        colors_list.append([float(col)/255 for col in line.split()])
    file_cmap.close()
    return colors.ListedColormap(colors_list, name='custom')
