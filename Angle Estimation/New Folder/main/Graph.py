import numpy as np 
txt_file = 'sully_dataset/data1.txt'
data_array = []

with open(txt_file, 'r') as file:
    lines = file.readlines()

    # Skip the first line and process the rest
    for line in lines[1:]:
        # Process each line as needed and append to the array
        
        data_array.append((line.strip().split()[1]))

# Now, data_array contains the data from the file (excluding the first line)
print(data_array)

def save_temporary_numpy(array):
    # Create a temporary file
        # Save the NumPy array to the temporary file
    np.save('abcd', array)
        
        # Get the filename of the temporary file
       

    return 'abcd'

def load_temporary_numpy(temp_file_path):
    # Load the NumPy array from the temporary file
    loaded_array = np.load(temp_file_path)
    
    return loaded_array


pre = load_temporary_numpy('predicted.npy')
act=  np.array(data_array,dtype=np.float64)
act= act[1:10000]
print(pre.shape)
print(act.shape)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt


# Create a continuous function for visualization
# act =act[4500:4600]
# pre = pre[4500:4600]
# x = np.linspace(0, len(act) - 1, len(act))
# plt.plot(x, act, label='Actual', marker='o')
# plt.plot(x, pre, label='Predicted', marker='x')

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Values')
# plt.title('Comparison of Actual and Predicted Values')
# plt.legend()  # Show legend

# # Display the plot
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np

# # Example vectors
# x_values = np.array([1, 2, 3, 4, 5])
# y_values = np.array([2, 4, 6, 8, 10])

# Plotting the continuous line
# plt.plot(act, , label='Continuous Line')
act = act[200:300]
pre = pre[200:300]
x1_values = np.arange(1, len(act) + 1)
x2_values = np.arange(1, len(pre) + 1)
# Plotting the values against the total number of data points
plt.plot(x1_values, act, marker='o', linestyle='-', color='b', label='Ground Truth')
plt.plot(x2_values, pre, marker='o', linestyle='-', color='r', label='Predicted Angle')
# Adding labels and title
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Values vs. Data Points Plot')

# Adding a legend
plt.legend()

# Display the plot
plt.show()
# Adding labels and title



