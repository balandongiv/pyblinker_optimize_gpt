'''
Let us see how to open a .mat file in python
'''

import numpy as np
from scipy.io import loadmat

# Specify the path to your .mat file
mat_file_path = r'..\Devel\step1bi_data_input_getBlinkPositions.mat'

# Load the .mat file
data = loadmat(mat_file_path)

# Display the keys in the loaded data
print("Keys in the .mat file:", data.keys())

# Access a specific variable from the file (replace 'variable_name' with an actual key from the .mat file)
if 'variable_name' in data:
    variable_data = data['variable_name']
    print("Shape of variable_data:", variable_data.shape)
    print("Contents of variable_data:\n", variable_data)
else:
    print("'variable_name' not found in the .mat file.")
