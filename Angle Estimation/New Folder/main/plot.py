import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_actual_vs_predicted(actual, predicted):
    # Create a DataFrame for Seaborn plotting
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})

    # Set up Seaborn
    sns.set(style="whitegrid")

    # Plot the data using Seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, markers=True)

    # Add labels and title
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')

    # Show the plot
    plt.show()

# Example usage
actual_values = np.array([1, 2, 3, 4, 5])
predicted_values = np.array([1.1, 2.2, 2.8, 3.7, 4.9])

# Call the function
plot_actual_vs_predicted(actual_values, predicted_values)
