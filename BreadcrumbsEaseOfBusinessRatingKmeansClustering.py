# Install required packages
# !pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set the working directory to Lab11 folder
# Replace the path with your actual path
working_directory = "C:/Users/ual-laptop/Desktop/Eller Fall 2022/MIS545/Lab11"
countries_path = "CountryData.csv"
countries_full_path = f"{working_directory}/{countries_path}"

# Reading CountryData.csv into a pandas DataFrame
countries = pd.read_csv(countries_full_path)

# Displaying the countries DataFrame on the console
print(countries)

# Displaying the structure of the countries DataFrame
print(countries.info())

# Displaying a summary of the countries DataFrame
print(countries.describe())

# Converting the column containing the country name to the 
# row index of the DataFrame
countries.set_index('Country', inplace=True)

# Removing rows with missing data in any feature
countries.dropna(inplace=True)
 
# Viewing the summary of the countries DataFrame again to 
# ensure there are no NA values
print(countries.describe())

# Scaling the DataFrame to take only CorruptionIndex and DaysToOpenBusiness
scaler = StandardScaler()
countries_scaled = scaler.fit_transform(countries[['CorruptionIndex', 'DaysToOpenBusiness']])

# Setting the random seed to 679
seed = 679

# Generating the k-means clusters in an object called countries4Clusters 
# using 4 clusters
kmeans = KMeans(n_clusters=4, n_init=25, random_state=seed)
countries4Clusters = kmeans.fit(countries_scaled)

# Displaying cluster sizes on the console
print(pd.Series(countries4Clusters.labels_).value_counts())

# Displaying cluster centers (z-scores) on the console
print(pd.DataFrame(scaler.inverse_transform(countries4Clusters.cluster_centers_), columns=['CorruptionIndex', 'DaysToOpenBusiness']))

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=countries_scaled[:, 0], y=countries_scaled[:, 1], hue=countries4Clusters.labels_, palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Corruption Index (Scaled)')
plt.ylabel('Days to Open Business (Scaled)')
plt.show()

# Determining the optimal value for k using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    kmeans.fit(countries_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Determining the optimal value for k using the silhouette score
from sklearn.metrics import silhouette_score

sil_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=seed)
    labels = kmeans.fit_predict(countries_scaled)
    sil_scores.append(silhouette_score(countries_scaled, labels))

# Plotting silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker='o', linestyle='--')
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.show()

# Regenerating the cluster analysis using the optimal number of clusters of 3
kmeans_optimal = KMeans(n_clusters=3, random_state=seed)
countries3Clusters = kmeans_optimal.fit(countries_scaled)

# Displaying cluster sizes on the console
print(pd.Series(countries3Clusters.labels_).value_counts())

# Displaying cluster centers (z-scores) on the console
print(pd.DataFrame(scaler.inverse_transform(countries3Clusters.cluster_centers_), columns=['CorruptionIndex', 'DaysToOpenBusiness']))

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=countries_scaled[:, 0], y=countries_scaled[:, 1], hue=countries3Clusters.labels_, palette='viridis')
plt.title('K-Means Clustering (Optimal k=3)')
plt.xlabel('Corruption Index (Scaled)')
plt.ylabel('Days to Open Business (Scaled)')
plt.show()

# Determining similarities and differences among the clusters
countries['Cluster'] = countries3Clusters.labels_
countries_clusters = countries.groupby('Cluster').mean()

print(countries_clusters[['GiniCoefficient', 'GDPPerCapita', 'EduPercGovSpend', 'EduPercGDP', 'CompulsoryEducationYears']])
