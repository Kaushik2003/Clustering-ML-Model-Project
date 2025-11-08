"""
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
# print(f"\nPredicted cluster: {cluster[0]}")
# print(f"Cluster profile: {model_info['cluster_profiles'][cluster[0]]}")
