import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate some random data points for clustering
np.random.seed(0)
points = np.random.rand(100, 2) * 100  # 100 points in a 2D space

# Apply DBSCAN clustering
db = DBSCAN(eps=5, min_samples=3)  # eps is the max distance between points in a cluster
labels = db.fit_predict(points)

# Plot the clustered points
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        # Noise points
        color = 'k'
    else:
        # Assign a unique color to each cluster
        color = plt.cm.jet(float(label) / max(unique_labels + {1}))
    
    plt.scatter(points[labels == label, 0], points[labels == label, 1], c=color, label=f'Cluster {label}' if label != -1 else 'Noise')

plt.title("DBSCAN Clustering Example")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()
