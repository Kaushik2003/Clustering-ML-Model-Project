# Clustering-ML-Model-Project

Clustering Gamer-Music Preferences 🎮🎵
Overview

This project explores the intersection of gaming and music preferences among students using unsupervised learning (clustering). The goal is to group students into meaningful personas based on their favorite game genres, music preferences, and average listening hours per week. These insights can then be applied to event planning, ensuring maximum engagement by tailoring activities to each cluster’s unique profile.

Problem Statement

Students often show overlapping preferences in entertainment, but these patterns are not always obvious. By applying clustering to a mixed dataset (categorical + numerical), we aim to:

Identify distinct student personas.

Analyze their gaming and music behavior.

Recommend event designs that best match each group.

Methodology

Dataset: Includes categorical features (game & music genres) and numerical features (weekly listening hours).

Preprocessing:

Ordinal encoding for categorical features.

StandardScaler normalization for numerical features.

Algorithm:

Evaluated K-Means, DBSCAN, and Hierarchical clustering.

Chose K-Prototypes (from the kmodes library) since it supports mixed-type data.

Cluster Selection:

Ran for k = 2–6.

Evaluated with cost function and silhouette score.

Optimal k = 3.

Visualization:

PCA and t-SNE used for 2D visualizations of clusters.

Results

The model identified three meaningful clusters:

Cluster 0 – Mainstream Entertainment Seekers: Action/Sports + Bollywood/Lofi; ~1.5 hrs/week listening.

Cluster 1 – Intellectual & Relaxed Audience: Puzzle/Sports + Pop/Bollywood; ~4.3 hrs/week listening.

Cluster 2 – Adventurous Party-Goers: Racing/FPS + EDM/Bollywood; ~8.9 hrs/week listening.

These clusters were clearly separated in both PCA and t-SNE visualizations.

Discussion

The clusters align with distinct entertainment personas, providing actionable insights for event planning:

Cluster 0 → energetic, competitive events.

Cluster 1 → thoughtful, relaxed events.

Cluster 2 → immersive, high-energy events.

Limitations include dataset size and overlap of preferences; future improvements could include larger datasets and additional features like age, gender, or social activity.

Conclusion

Applied K-Prototypes clustering on a mixed dataset.

Discovered 3 entertainment personas.

Proposed event recommendations tailored to each cluster.

Demonstrated how clustering can enhance real-world engagement strategies.

How to Run

Clone this repository.

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn kmodes


Open the notebook:

jupyter notebook Clustering_ML_Model.ipynb


Run cells step by step to reproduce preprocessing, clustering, and visualizations.

References

Huang, Z. (1997). Clustering large data sets with mixed numeric and categorical values. PAKDD.

scikit-learn Documentation

kmodes GitHub Repository
