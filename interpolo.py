import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import Rbf

# Read the CSV file into a DataFrame
df = pd.read_csv("Analytical_value_55K03.csv")

# Define a function to create an interpolated map using Inverse Distance Weighting (IDW)
def create_idw_interpolated_map(element):
    # Filter the data for the specified element
    element_data = df[['longitude', 'latitude', element]].dropna()

    # Extract latitude, longitude, and the element values
    X = element_data[['longitude', 'latitude']].values
    y = element_data[element].values

    # Fit an IDW model
    idw = KNeighborsRegressor(n_neighbors=4, weights='distance')
    idw.fit(X, y)

    # Create a grid of points for interpolation
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Interpolate using IDW
    interpolated_values = idw.predict(grid_points).reshape(xx.shape)

    # Plot the interpolated map
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, interpolated_values, cmap='viridis')
    plt.colorbar(contour, label=element)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Interpolated Map using IDW for {element}')

    # Add annotations
    plt.annotate('Higher Values', xy=(0.05, 0.05), xycoords='axes fraction', color='white', fontsize=10)
    plt.annotate('Lower Values', xy=(0.05, 0.95), xycoords='axes fraction', color='white', fontsize=10)

    # Add a legend with explanations
    plt.text(0.85, 0.1, 'Color Scale:', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, color='white')
    plt.text(0.85, 0.05, 'Contour Lines:', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, color='white')
    plt.text(0.85, 0.9, 'Higher Values', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')
    plt.text(0.85, 0.85, 'Lower Values', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')
    plt.text(0.85, 0.15, 'Ranges of element values\nrepresented by colors', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')
    plt.text(0.85, 0.1, 'Ranges of element values\nrepresented by contour lines', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')

    plt.show()

# Define a function to create an interpolated map using Kriging
def create_kriging_interpolated_map(element):
    # Filter the data for the specified element
    element_data = df[['longitude', 'latitude', element]].dropna()

    # Extract latitude, longitude, and the element values
    X = element_data[['longitude', 'latitude']].values
    y = element_data[element].values

    # Fit a Kriging model
    rbfi = Rbf(X[:, 0], X[:, 1], y, function='gaussian')

    # Create a grid of points for interpolation
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Interpolate using Kriging
    interpolated_values = rbfi(xx, yy)

    # Plot the interpolated map
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, interpolated_values, cmap='viridis')
    plt.colorbar(contour, label=element)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=20)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Interpolated Map using Kriging for {element}')

    # Add annotations
    plt.annotate('Higher Values', xy=(0.05, 0.05), xycoords='axes fraction', color='white', fontsize=10)
    plt.annotate('Lower Values', xy=(0.05, 0.95), xycoords='axes fraction', color='white', fontsize=10)

    # Add a legend with explanations
    plt.text(0.85, 0.1, 'Color Scale:', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, color='white')
    plt.text(0.85, 0.05, 'Contour Lines:', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10, color='white')
    plt.text(0.85, 0.9, 'Higher Values', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')
    plt.text(0.85, 0.85, 'Lower Values', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')
    plt.text(0.85, 0.15, 'Ranges of element values\nrepresented by colors', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')
    plt.text(0.85, 0.1, 'Ranges of element values\nrepresented by contour lines', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, fontsize=8, color='white')

    plt.show()

# Get user input for the element
element_name = input("Enter the name of the element for which you want to see the interpolated map: ")

# Example usage
create_idw_interpolated_map(element_name)
create_kriging_interpolated_map(element_name)
