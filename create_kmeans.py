from sklearn.cluster import KMeans
import pickle
import numpy as np

# Create a simple KMeans model with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
data = np.array([(i,) for i in range(10)])
kmeans.fit(data)

# Save the model
with open('./models/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)