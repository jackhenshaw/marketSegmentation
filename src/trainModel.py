import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from joblib import dump

df = pd.read_csv("data/customer_data_complete.csv")

# Normalise data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Create ML model
meanShift = MeanShift()
meanShift.fit_predict(scaled_df)

# Print some information about the model
print("Number of clusters = ", meanShift.n_features_in_)
print("Silhouette score = ", silhouette_score(scaled_df, meanShift.labels_, metric = 'euclidean'))

# Save ML model
dump(meanShift, 'models/meanShift_model.joblib')