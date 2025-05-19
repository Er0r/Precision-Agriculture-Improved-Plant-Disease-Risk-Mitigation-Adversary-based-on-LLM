import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import logging

logging.getLogger('tensorflow').setLevel(logging.DEBUG)

severity_map = {
    'healthy': 0.0,
    'early_blight': 0.3,
    'common_rust': 0.3,
    'leaf_spot': 0.4,
    'spider_mites': 0.4,
    'powdery_mildew': 0.4,
    'gray_leaf_spot': 0.4,
    'northern_leaf_blight': 0.4,
    'late_blight': 0.8,
    'scab': 0.7,
    'black_rot': 0.7,
    'bacterial_spot': 0.6,
    'target_spot': 0.6,
    'mosaic_virus': 0.7,
    'yellow_leaf_curl_virus': 0.7,
    'leaf_scorch': 0.6,
    'leaf_mold': 0.5,
    'septoria_leaf_spot': 0.5,
    'esca_(black_measles)': 0.6,
    'isariopsis_leaf_spot': 0.5,
    'cedar_apple_rust': 0.5,
    'apple_rust': 0.5,
    'blight': 0.8,
    'rust': 0.5,
    'anthracnose': 0.6,
    'verticillium_wilt': 0.6,
    'brown_spot': 0.5,
    'downy_mildew': 0.6,
    'phytophthora_infestans': 0.8,
}

tfds_dir = 'F:/Personal/PrecisionAgriculture/tfds_data'
os.makedirs(tfds_dir, exist_ok=True)
tfds.core.utils.gcs_utils._is_gcs_disabled = True

csv_path = './dataset/Crop_recommendation.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Download from Kaggle.")
df_crop = pd.read_csv(csv_path)


expected_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
if not all(col in df_crop.columns for col in expected_columns):
    raise ValueError("CSV missing expected columns.")
df_crop = df_crop[expected_columns]
df_crop['plot_id'] = range(1, len(df_crop) + 1)


numerical_cols = ['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall']
df_crop[numerical_cols] = df_crop[numerical_cols].fillna(df_crop[numerical_cols].median())

encoder = OneHotEncoder(sparse_output=False)
crop_encoded = encoder.fit_transform(df_crop[['label']])
df_crop_encoded = pd.DataFrame(crop_encoded, columns=encoder.get_feature_names_out())
df_crop = pd.concat([df_crop.drop('label', axis=1), df_crop_encoded], axis=1)


scaler = MinMaxScaler()
df_crop[numerical_cols] = scaler.fit_transform(df_crop[numerical_cols])

df_crop.to_csv('crop_recommendation_processed.csv', index=False)
print("Crop Recommendation Processed Sample:")
print(df_crop.head())

def load_plant_village():
    try:
        for attempt in range(3):
            print(f"Attempt {attempt + 1} to load Plant Village...")
            ds, info = tfds.load(
                'plant_village',
                split='train',
                as_supervised=True,
                with_info=True,
                data_dir=tfds_dir,
                download=True,
                download_and_prepare_kwargs={'timeout': 120}
            )
            return ds, info
    except Exception as e:
            print(f"Failed to load Plant Village via TFDS: {e}. Using mock data.")
            plant_data = [
                {
                    'image_id': f'img_{i+1}',
                    'crop_type': np.random.choice(['Apple', 'Tomato', 'Grape']),
                    'disease_label': (disease := np.random.choice(['scab', 'blight', 'healthy'])),
                    'severity': severity_map.get(disease, 0.5),
                    'hsv_h': np.random.uniform(0, 180),
                    'hsv_s': np.random.uniform(0, 1),
                    'hsv_v': np.random.uniform(0, 1)
                } for i in range(1000)
            ]
            df_plant = pd.DataFrame(plant_data)
            return df_plant, None


ds, info = load_plant_village()

if isinstance(ds, pd.DataFrame):
    df_plant = ds
    label_map = None
else:
    label_map = info.features['label'].names
    def process_image(image, label):
        image = tf.image.resize(image, [224, 224])
        image = image / 255.0
        
        crop_type = label_map[label].split('___')[0]
        disease_label = label_map[label].split('___')[1]

        severity = severity_map.get(disease_label, 0.5) 
        hsv_image = tf.image.rgb_to_hsv(image)
        hsv_mean = tf.reduce_mean(hsv_image, axis=[0, 1]).numpy()
        return {
            'severity': severity,
            'hsv_h': hsv_mean[0],
            'hsv_s': hsv_mean[1],
            'hsv_v': hsv_mean[2],
            'crop_type': crop_type,
            'disease_label': disease_label
        }

    plant_data = []
    for i, (image, label) in enumerate(ds.take(1000)):
        features = process_image(image, label)
        features['image_id'] = f'img_{i+1}'
        plant_data.append(features)
    df_plant = pd.DataFrame(plant_data)


df_plant.dropna(inplace=True)
df_plant[['severity', 'hsv_h', 'hsv_s', 'hsv_v']] = scaler.fit_transform(df_plant[['severity', 'hsv_h', 'hsv_s', 'hsv_v']])


df_plant.to_csv('plant_village_processed.csv', index=False)
print("Plant Village Processed Sample:")
print(df_plant.head())


X_plant = df_plant[['severity', 'hsv_h', 'hsv_s', 'hsv_v']].values
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_plant)
distances, _ = neigh.kneighbors(X_plant)
distances = np.sort(distances[:, 4])
plt.plot(distances)
plt.title('K-Distance Plot for Plant Village')
plt.savefig('plant_village_k_distance.png')
plt.close()

dbscan_plant = DBSCAN(eps=0.15, min_samples=5)
df_plant['cluster'] = dbscan_plant.fit_predict(X_plant)
print("Plant Village Cluster Counts:")
print(df_plant['cluster'].value_counts())

X_crop = df_crop[numerical_cols].values
neigh.fit(X_crop)
distances, _ = neigh.kneighbors(X_crop)
distances = np.sort(distances[:, 4])
plt.plot(distances)
plt.title('K-Distance Plot for Crop Recommendation')
plt.savefig('crop_recommendation_k_distance.png')
plt.close()

dbscan_crop = DBSCAN(eps=0.3, min_samples=5)  
df_crop['cluster'] = dbscan_crop.fit_predict(X_crop)
print("Crop Recommendation Cluster Counts:")
print(df_crop['cluster'].value_counts())


plant_summary = df_plant.groupby(['crop_type', 'cluster']).agg({'severity': 'mean'}).reset_index()
plant_summary['summary'] = plant_summary.apply(
    lambda x: f"{x['crop_type']} cluster {x['cluster']}: severity {x['severity']:.2f}", axis=1
)


df_crop['crop_type'] = df_crop[encoder.get_feature_names_out()].idxmax(axis=1).str.replace('crop_type_', '')
crop_summary = df_crop.groupby(['crop_type', 'cluster']).agg({
    'N': 'mean', 'ph': 'mean', 'humidity': 'mean'
}).reset_index()
crop_summary['summary'] = crop_summary.apply(
    lambda x: f"{x['crop_type']} cluster {x['cluster']}: N {x['N']:.2f}, ph {x['ph']:.2f}, humidity {x['humidity']:.2f}", axis=1
)

llm_prompt = f"""
Analyze plant disease and soil data for advisory messages:
- Plant Disease: {plant_summary['summary'].tolist()}
- Soil Data: {crop_summary['summary'].tolist()}
Generate a pest control recommendation for apple crops.
"""
print("Sample LLM Prompt:")
print(llm_prompt)