import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Read the Excel file into a DataFrame
df = pd.read_csv("Analytical_value_55K03.csv")

# Define a function to create an anomaly map for a specific element
def create_anomaly_map(element):
    # Filter the data for the specified element
    element_data = df[['longitude', 'latitude', element]].dropna()

    # Extract latitude, longitude, and the element values
    X = element_data[['longitude', 'latitude']].values
    y = element_data[element].values.reshape(-1, 1)

    # Apply an anomaly detection algorithm (Isolation Forest)
    clf = IsolationForest(contamination=0.1)
    clf.fit(y)
    y_pred = clf.predict(y)

    # Plot the anomalies on a map
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=np.where(y_pred == -1, 'red', 'blue'), alpha=0.6) 
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Anomaly Map for {element}')
    plt.show()

# Example usage
create_anomaly_map('mgo')
