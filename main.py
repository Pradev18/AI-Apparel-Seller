import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('female_data.csv')

def preprocess_data(df):
    # For male data, we don't need Cup Size
    if 'Cup Size' in df.columns:
        df = df.drop('Cup Size', axis=1)
    return df

def cluster_and_size(df, features, n_sizes=7):
    X = StandardScaler().fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_sizes, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
 
    cluster_weights = df.groupby('Cluster')['Weight'].mean().sort_values()
    size_categories = ['XS','S', 'M', 'L', 'XL', 'XXL', 'XXXL'][:n_sizes]
    cluster_to_size = dict(zip(cluster_weights.index, size_categories))
    df['Size'] = df['Cluster'].map(cluster_to_size)
    
    size_means = df.groupby('Size')[features].mean().reindex(size_categories)
    return df, size_means

def post_process_size_means(size_means):
    # Ensure logical progression of sizes
    for feature in size_means.columns:
        size_means[feature] = size_means[feature].sort_values().values
    return size_means

# Preprocess the data
data = preprocess_data(data)

# Upper body model
upper_features = ['Weight', 'Bust/Chest']
upper_data, upper_size_means = cluster_and_size(data, upper_features)

# Lower body model
lower_features = ['Weight', 'Waist', 'Hips']
lower_data, lower_size_means = cluster_and_size(data, lower_features)

# Post-process to ensure logical size progression
upper_size_means = post_process_size_means(upper_size_means)
lower_size_means = post_process_size_means(lower_size_means)

# Print results
print("Upper Body Size Averages:")
print(upper_size_means)
print("\nLower Body Size Averages:")
print(lower_size_means)

# Combine results
data['Upper_Size'] = upper_data['Size']
data['Lower_Size'] = lower_data['Size']

print("\nIndividual Measurements with Size Classifications:")
print(data[['Height in cms', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'Upper_Size', 'Lower_Size']])

# Calculate size distribution
print("\nUpper Body Size Distribution:")
print(data['Upper_Size'].value_counts().sort_index())
print("\nLower Body Size Distribution:")
print(data['Lower_Size'].value_counts().sort_index())