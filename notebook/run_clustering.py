"""
Clustering ML Model - Student Gaming and Music Preferences
This script runs the K-Prototypes clustering analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from kmodes.kprototypes import KPrototypes
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CLUSTERING ML MODEL - STUDENT PREFERENCES ANALYSIS")
print("="*60)

# Load the dataset
print("\n[1/7] Loading dataset...")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset.csv')

# Change to script directory to save outputs there
os.chdir(script_dir)

try:
    df = pd.read_csv(dataset_path)
    print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"✗ Error: dataset.csv not found at {dataset_path}!")
    print("Please ensure dataset.csv is in the same directory as this script.")
    exit(1)

# --- DATA EXPLORATION ---
print("\n" + "="*60)
print("[2/7] DATA EXPLORATION")
print("="*60)

print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nFirst few rows:")
print(df.head())

# Summary statistics
print("\n--- Listening Hours Statistics ---")
print(df['listening_hours'].describe())

# Value counts for categorical columns
print("\n--- Categorical Features Distribution ---")
for col in df.columns[:-1]:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head(5))

# Visualization 1: Distribution of listening hours
plt.figure(figsize=(6,4))
df['listening_hours'].hist(bins=10, edgecolor='black')
plt.title("Distribution of Listening Hours")
plt.xlabel("Hours")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('01_listening_hours_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_listening_hours_distribution.png")
plt.close()

# Visualization 2: Top game genres
game_genres = pd.concat([df['game_genre_top_1'], df['game_genre_top_2'], df['game_genre_top_3']])
plt.figure(figsize=(10,5))
game_genres.value_counts().head(10).plot(kind='bar', color='steelblue')
plt.title("Top 10 Game Genres (All Preferences Combined)")
plt.xlabel("Game Genre")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('02_top_game_genres.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_top_game_genres.png")
plt.close()

# Visualization 3: Top music genres
music_genres = pd.concat([df['music_genre_top_1'], df['music_genre_top_2'], df['music_genre_top_3']])
plt.figure(figsize=(10,5))
music_genres.value_counts().head(10).plot(kind='bar', color='coral')
plt.title("Top 10 Music Genres (All Preferences Combined)")
plt.xlabel("Music Genre")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_top_music_genres.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_top_music_genres.png")
plt.close()

# --- DATA PREPROCESSING ---
print("\n" + "="*60)
print("[3/7] DATA PREPROCESSING")
print("="*60)

print("\nWhy Ordinal Encoding?")
print("- Our dataset has ORDERED preferences (top_1 > top_2 > top_3)")
print("- One-hot encoding would treat all genres equally (no order)")
print("- Ordinal encoding preserves the preference ranking")

print("\nWhy Standard Scaling?")
print("- Numerical features need to be on the same scale")
print("- Prevents features with larger values from dominating")

# Define features
categorical_features = ['game_genre_top_1', 'game_genre_top_2', 'game_genre_top_3', 
                        'music_genre_top_1', 'music_genre_top_2', 'music_genre_top_3']
numerical_features = ['listening_hours']

# Create a copy for preprocessing
df_processed = df.copy()

# Ordinal Encoding for categorical features
ordinal_encoder = OrdinalEncoder()
df_processed[categorical_features] = ordinal_encoder.fit_transform(df[categorical_features])

# Standard Scaling for numerical features
scaler = StandardScaler()
df_processed[numerical_features] = scaler.fit_transform(df[numerical_features])

# Convert to numpy array for K-Prototypes
data_matrix = df_processed.values

# Get indices of categorical columns
categorical_indices = [df_processed.columns.get_loc(col) for col in categorical_features]

print(f"\n✓ Preprocessing complete")
print(f"  - Categorical features: {len(categorical_features)}")
print(f"  - Numerical features: {len(numerical_features)}")
print(f"  - Total features: {data_matrix.shape[1]}")

# Visualization 4: Game genre preferences by rank
pref_counts = {
    "Top 1": df['game_genre_top_1'].value_counts(),
    "Top 2": df['game_genre_top_2'].value_counts(),
    "Top 3": df['game_genre_top_3'].value_counts(),
}
pref_df = pd.DataFrame(pref_counts).fillna(0)

plt.figure(figsize=(12,6))
pref_df.plot(kind="bar", figsize=(12,6))
plt.title("Game Genre Preferences by Rank")
plt.xlabel("Game Genre")
plt.ylabel("Count")
plt.legend(title="Preference Rank")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('04_game_preferences_by_rank.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_game_preferences_by_rank.png")
plt.close()

# Visualization 5: Before and after scaling
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before scaling
axes[0].hist(df['listening_hours'], bins=10, edgecolor="black", color='steelblue')
axes[0].set_title("Listening Hours - Original Scale")
axes[0].set_xlabel("Hours")
axes[0].set_ylabel("Count")

# After scaling
df_scaled = df.copy()
df_scaled['listening_hours_scaled'] = scaler.fit_transform(df[['listening_hours']])
axes[1].hist(df_scaled['listening_hours_scaled'], bins=10, edgecolor="black", color="coral")
axes[1].set_title("Listening Hours - After Standard Scaling")
axes[1].set_xlabel("Scaled Hours")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig('05_scaling_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_scaling_comparison.png")
plt.close()

# --- FINDING OPTIMAL K ---
print("\n" + "="*60)
print("[4/7] FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*60)

print("\nWhy K-Prototypes?")
print("- Our dataset has BOTH categorical AND numerical features")
print("- K-Means only works with numerical data")
print("- K-Modes only works with categorical data")
print("- K-Prototypes handles MIXED data types!")

print("\nTesting different values of K...")
costs = []
silhouette_scores = []
cluster_range = range(2, 8)

for k in cluster_range:
    print(f"  Testing K={k}...", end=" ")
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=10, random_state=42, verbose=0)
    clusters = kproto.fit_predict(data_matrix, categorical=categorical_indices)
    costs.append(kproto.cost_)
    sil_score = silhouette_score(data_matrix, clusters)
    silhouette_scores.append(sil_score)
    print(f"Cost: {kproto.cost_:.2f}, Silhouette: {sil_score:.3f}")

# Plot Elbow and Silhouette methods
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Method
axes[0].plot(cluster_range, costs, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].set_title('Elbow Method for K-Prototypes', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Cost', fontsize=12)
axes[0].set_xticks(cluster_range)
axes[0].grid(True, alpha=0.3)

# Silhouette Method
axes[1].plot(cluster_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='coral')
axes[1].set_title('Silhouette Method for K-Prototypes', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_xticks(cluster_range)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_elbow_silhouette_methods.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 06_elbow_silhouette_methods.png")
plt.close()

# Select optimal K
best_k = 3
print(f"\n✓ Selected optimal K = {best_k} based on elbow method")

# --- BUILDING FINAL MODEL ---
print("\n" + "="*60)
print(f"[5/7] BUILDING FINAL K-PROTOTYPES MODEL (K={best_k})")
print("="*60)

final_kproto = KPrototypes(n_clusters=best_k, init='Cao', n_init=10, random_state=42, verbose=0)
clusters = final_kproto.fit_predict(data_matrix, categorical=categorical_indices)

df['cluster'] = clusters
print(f"\n✓ Model trained successfully")
print(f"  - Final cost: {final_kproto.cost_:.2f}")

# --- CLUSTER ANALYSIS ---
print("\n" + "="*60)
print("[6/7] CLUSTER ANALYSIS & PERSONAS")
print("="*60)

cluster_profiles = {}

for i in range(best_k):
    cluster_df = df[df['cluster'] == i]
    print(f"\n{'='*50}")
    print(f"CLUSTER {i} PERSONA")
    print(f"{'='*50}")
    print(f"Number of students: {len(cluster_df)}")
    
    print("\nTop Game Genres:")
    print(cluster_df['game_genre_top_1'].value_counts().head(3))
    
    print("\nTop Music Genres:")
    print(cluster_df['music_genre_top_1'].value_counts().head(3))
    
    avg_hours = cluster_df['listening_hours'].mean()
    print(f"\nAverage Listening Hours: {avg_hours:.2f}")
    
    # Store profile
    top_games = cluster_df['game_genre_top_1'].value_counts().head(2).index.tolist()
    top_music = cluster_df['music_genre_top_1'].value_counts().head(2).index.tolist()
    
    cluster_profiles[i] = {
        "size": len(cluster_df),
        "top_games": top_games,
        "top_music": top_music,
        "avg_hours": avg_hours
    }

# --- VISUALIZATIONS ---
print("\n" + "="*60)
print("[7/7] CREATING VISUALIZATIONS")
print("="*60)

# Visualization 6: PCA
print("\nGenerating PCA visualization...")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_matrix)

pca_df = pd.DataFrame(data=principal_components, 
                      columns=['Principal Component 1', 'Principal Component 2'])
pca_df['cluster'] = df['cluster']

plt.figure(figsize=(10, 7))
colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'mediumpurple']
for i in range(best_k):
    cluster_data = pca_df[pca_df['cluster'] == i]
    plt.scatter(cluster_data['Principal Component 1'],
                cluster_data['Principal Component 2'],
                label=f'Cluster {i} (n={len(cluster_data)})',
                alpha=0.7, s=100, color=colors[i % len(colors)])

plt.title(f'K-Prototypes Clusters (K={best_k}) - PCA Visualization', 
          fontsize=14, fontweight='bold')
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_pca_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_pca_clusters.png")
plt.close()

# Visualization 7: t-SNE
print("Generating t-SNE visualization (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
tsne_components = tsne.fit_transform(data_matrix)

tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['cluster'] = df['cluster']

plt.figure(figsize=(10, 7))
for i in range(best_k):
    cluster_data = tsne_df[tsne_df['cluster'] == i]
    plt.scatter(cluster_data['t-SNE 1'],
                cluster_data['t-SNE 2'],
                label=f'Cluster {i} (n={len(cluster_data)})',
                alpha=0.7, s=100, color=colors[i % len(colors)])

plt.title(f'K-Prototypes Clusters (K={best_k}) - t-SNE Visualization', 
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('08_tsne_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_tsne_clusters.png")
plt.close()

# Visualization 8: Cluster profiles
print("Generating cluster profile visualizations...")
for i in range(best_k):
    cluster_df = df[df['cluster'] == i]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top Games
    game_counts = cluster_df['game_genre_top_1'].value_counts().head(8)
    sns.barplot(y=game_counts.index, x=game_counts.values, ax=axes[0], 
                palette='Blues_r')
    axes[0].set_title(f"Cluster {i}: Top Game Preferences", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Count", fontsize=10)
    axes[0].set_ylabel("")
    
    # Top Music
    music_counts = cluster_df['music_genre_top_1'].value_counts().head(8)
    sns.barplot(y=music_counts.index, x=music_counts.values, ax=axes[1], 
                palette='Oranges_r')
    axes[1].set_title(f"Cluster {i}: Top Music Preferences", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Count", fontsize=10)
    axes[1].set_ylabel("")
    
    plt.suptitle(f"Cluster {i} Profile (n={len(cluster_df)} students)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'09_cluster_{i}_profile.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 09_cluster_{i}_profile.png")
    plt.close()

# --- FINAL RECOMMENDATIONS ---
print("\n" + "="*60)
print("FINAL EVENT RECOMMENDATIONS")
print("="*60)

print("\nBased on the clustering analysis, here are the recommended")
print("gaming-music event combinations for maximum participation:\n")

for i in range(best_k):
    profile = cluster_profiles[i]
    
    event_games = " & ".join(profile["top_games"])
    event_music = " & ".join(profile["top_music"])
    
    print(f"\n🎉 EVENT {i+1} - Cluster {i} Target")
    print(f"   {'─'*50}")
    print(f"   Target Audience: {profile['size']} students")
    print(f"   🎮 Recommended Games: {event_games}")
    print(f"   🎵 Recommended Music: {event_music}")
    print(f"   ⏱️  Avg Listening Hours: {profile['avg_hours']:.2f}")

# Save cluster assignments
df.to_csv('clustered_data.csv', index=False)
print(f"\n✓ Saved: clustered_data.csv (dataset with cluster assignments)")

# --- SAVE TRAINED MODEL AND PREPROCESSING OBJECTS ---
print("\n" + "="*60)
print("SAVING TRAINED MODEL")
print("="*60)

# Save the trained K-Prototypes model
model_filename = 'kprototypes_model.joblib'
joblib.dump(final_kproto, model_filename)
print(f"✓ Saved: {model_filename} (trained K-Prototypes model)")

# Save the ordinal encoder
encoder_filename = 'ordinal_encoder.joblib'
joblib.dump(ordinal_encoder, encoder_filename)
print(f"✓ Saved: {encoder_filename} (ordinal encoder for categorical features)")

# Save the standard scaler
scaler_filename = 'standard_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"✓ Saved: {scaler_filename} (standard scaler for numerical features)")

# Save feature names and categorical indices for future use
model_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'categorical_indices': categorical_indices,
    'best_k': best_k,
    'cluster_profiles': cluster_profiles
}
info_filename = 'model_info.joblib'
joblib.dump(model_info, info_filename)
print(f"✓ Saved: {info_filename} (model configuration and cluster profiles)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("\n📊 Visualizations:")
print("  1. 01_listening_hours_distribution.png")
print("  2. 02_top_game_genres.png")
print("  3. 03_top_music_genres.png")
print("  4. 04_game_preferences_by_rank.png")
print("  5. 05_scaling_comparison.png")
print("  6. 06_elbow_silhouette_methods.png")
print("  7. 07_pca_clusters.png")
print("  8. 08_tsne_clusters.png")
print(f"  9-{8+best_k}. 09_cluster_X_profile.png (one per cluster)")
print("\n📁 Data Files:")
print("  10. clustered_data.csv")
print("\n🤖 Model Files:")
print("  11. kprototypes_model.joblib (trained model)")
print("  12. ordinal_encoder.joblib (encoder)")
print("  13. standard_scaler.joblib (scaler)")
print("  14. model_info.joblib (configuration)")
print("\n✓ All visualizations, data, and model files saved successfully!")

# Create a simple prediction example script
prediction_script = '''"""
Example script to load and use the trained clustering model
"""
import pandas as pd
import joblib

# Load the trained model and preprocessing objects
model = joblib.load('kprototypes_model.joblib')
encoder = joblib.load('ordinal_encoder.joblib')
scaler = joblib.load('standard_scaler.joblib')
model_info = joblib.load('model_info.joblib')

print("Model loaded successfully!")
print(f"Number of clusters: {model_info['best_k']}")
print(f"Categorical features: {model_info['categorical_features']}")
print(f"Numerical features: {model_info['numerical_features']}")

# Example: Predict cluster for new data
# new_data = pd.DataFrame({
#     'game_genre_top_1': ['FPS'],
#     'game_genre_top_2': ['Racing'],
#     'game_genre_top_3': ['Sports'],
#     'music_genre_top_1': ['Bollywood'],
#     'music_genre_top_2': ['Pop'],
#     'music_genre_top_3': ['Rock'],
#     'listening_hours': [3]
# })

# # Preprocess the new data
# new_data_processed = new_data.copy()
# new_data_processed[model_info['categorical_features']] = encoder.transform(
#     new_data[model_info['categorical_features']]
# )
# new_data_processed[model_info['numerical_features']] = scaler.transform(
#     new_data[model_info['numerical_features']]
# )

# # Predict cluster
# cluster = model.predict(new_data_processed.values, categorical=model_info['categorical_indices'])
# print(f"\\nPredicted cluster: {cluster[0]}")
# print(f"Cluster profile: {model_info['cluster_profiles'][cluster[0]]}")
'''

with open('predict_example.py', 'w') as f:
    f.write(prediction_script)
print("\n✓ Created: predict_example.py (example usage script)")
