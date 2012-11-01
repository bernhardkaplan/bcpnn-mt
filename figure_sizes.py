import numpy as np

# --------------------------------------------------------------------------
def get_figsize(fig_width_pt):
    inches_per_pt = 1.0/72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

def get_figsize_landscape(fig_height_pt):
    inches_per_pt = 1.0/72.0                # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
    fig_height = fig_height_pt*inches_per_pt  # height in inches
    fig_width = fig_height*golden_mean      # width in inches
    fig_size =  [fig_height,fig_width]      # exact figsize
    return fig_size

def get_figsize_A4():
    fig_width = 8.27
    fig_height = 11.69
    fig_size =  [fig_width,fig_height]      # exact figsize
    return fig_size

# --------------------------------------------------------------------------
