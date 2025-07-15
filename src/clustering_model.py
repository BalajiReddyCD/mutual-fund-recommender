# Step-by-Step Clustering Plan:
# Aggregate fund-level features (NAV, volatility, CAGR)
# Standardize features using StandardScaler
# Use Elbow Method to find optimal k (no. of clusters)
# Fit K-Means and visualize
# Interpret clusters and export results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:/Users/BALA/OneDrive - University of Hertfordshire/Desktop/mutual-fund-recommender/app/data/processed/preprocessed_mutual_funds.csv")

# Aggregate fund-level features for clustering
cluster_df = df.groupby('Scheme_Code').agg({
    'NAV': 'mean',
    'Rolling_Std_NAV': 'mean',
    'CAGR': 'first',
    'Scheme_Name': 'first'
}).dropna()

# Keep only numeric features for clustering
X_cluster = cluster_df[['NAV', 'Rolling_Std_NAV', 'CAGR']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Preview scaled data shape and basic stats
X_cluster.describe(), X_scaled.shape

# Step 2: Find Optimal Number of Clusters (k)
# We’ll use the Elbow Method:
# Plot k vs. inertia (SSE) and look for the “elbow point” where gains flatten.

# Find optimal k using Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (SSE)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Fit KMeans with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels back to original DataFrame
cluster_df['Cluster'] = cluster_labels

# Plot 2D clusters (CAGR vs Volatility)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=cluster_df,
    x='Rolling_Std_NAV',
    y='CAGR',
    hue='Cluster',
    palette='tab10',
    s=100,
    alpha=0.8
)
plt.title('Mutual Fund Clusters – Risk (Volatility) vs Return (CAGR)')
plt.xlabel('Average Volatility (Rolling Std NAV)')
plt.ylabel('CAGR (%)')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
