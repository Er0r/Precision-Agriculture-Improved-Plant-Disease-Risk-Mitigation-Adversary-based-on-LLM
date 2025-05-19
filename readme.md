# Precision Agriculture Data Preprocessing and Clustering

This project outlines the steps taken to preprocess and cluster two distinct agricultural datasets: the Plant Village dataset (focused on plant disease analysis) and a Crop Recommendation dataset. The goal is to prepare the data for potential downstream tasks such as disease classification, crop recommendation system development, or exploratory data analysis.

## Datasets

1.  **Plant Village Dataset:** Contains images of various plant diseases and healthy plants.
2.  **Crop Recommendation Dataset:** A tabular dataset with features like nutrient levels (N, P, K), environmental factors (temperature, humidity, pH, rainfall), and the recommended crop.

## Preprocessing and Clustering Steps

### Plant Village Dataset

1.  **Set Custom TFDS Directory:**
    ```python
    TFDS_DATA_DIR = "F:/Personal/PrecisionAgriculture/tfds_data"
    ```
    Configures TensorFlow Datasets (TFDS) to store data in the specified directory.

2.  **Attempt to Load Dataset:**
    ```python
    # Code to load Plant Village dataset via TFDS (training split, with retries)
    # Fallback to generate_mock_plant_village_data() if loading fails
    ```
    Attempts to load the Plant Village dataset using TFDS. If unsuccessful after 3 retries, it falls back to generating a synthetic dataset for initial development and testing.

3.  **Generate Mock Data (If Needed):**
    ```python
    # Function to create mock plant village entries
    # Includes random crop types, disease labels, heuristic severity, and HSV values
    ```
    Creates a DataFrame with 1000 mock entries if the TFDS loading fails. This data includes random crop types, disease labels, a heuristic severity score, and HSV color information.

4.  **Extract Features (TFDS Path):**
    ```python
    # Function to extract features from the Plant Village images (if loaded via TFDS)
    # - Resize images to 224x224
    # - Normalize pixel values
    # - Compute heuristic severity
    # - Calculate HSV metrics (e.g., mean HSV)
    # - Extract crop type and disease label
    ```
    Processes the image data from the Plant Village dataset. This involves resizing, normalization, calculating a simple severity score, extracting color features in the HSV space, and obtaining the ground truth crop type and disease label.

5.  **Create DataFrame:**
    ```python
    # Code to combine extracted features into a Pandas DataFrame
    # Columns: image_id, crop_type, disease_label, severity, hsv_h, hsv_s, hsv_v
    ```
    Organizes the extracted features into a structured Pandas DataFrame for easier manipulation and analysis.

6.  **Handle Missing Values:**
    ```python
    # Code to remove rows with any missing values in the DataFrame
    ```
    Removes any data points that have missing values across the extracted features.

7.  **Normalize Features:**
    ```python
    # Code to normalize the numerical features (severity, hsv_h, hsv_s, hsv_v) to the range [0, 1]
    ```
    Scales the numerical features to a common range to ensure that clustering algorithms are not biased by features with larger values.

8.  **Save Processed Data:**
    ```python
    # Code to save the processed Plant Village DataFrame to 'plant_village_processed.csv'
    ```
    Saves the cleaned and processed Plant Village data to a CSV file for future use.

9.  **Generate K-Distance Plot:**
    ```python
    # Code to compute k-distances (k=5) for DBSCAN and plot the curve
    # This plot is used to visually determine a suitable 'eps' value for DBSCAN
    ```
    Calculates the distance to the 5th nearest neighbor for each data point and plots these distances. This visualization helps in selecting an appropriate `eps` (epsilon) value for the DBSCAN clustering algorithm.

10. **Apply DBSCAN Clustering:**
    ```python
    # Code to apply DBSCAN clustering with eps=0.15 and min_samples=5
    ```
    Applies the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to the processed Plant Village data using a chosen `eps` value of 0.15 and a minimum of 5 samples to form a dense region.

11. **Evaluate Clusters:**
    ```python
    # Code to print the counts of data points in each identified cluster
    # This helps in understanding the outcome of the clustering process
    ```
    Prints the number of data points belonging to each cluster identified by DBSCAN, including the number of outliers (often labeled as -1).

### Crop Recommendation Dataset

1.  **Load Dataset:**
    ```python
    # Code to load the Crop Recommendation dataset from './dataset/Crop_recommendation.csv'
    # Expected to have approximately 2200 rows
    ```
    Loads the Crop Recommendation dataset from a local CSV file using a library like Pandas.

2.  **Verify Features:**
    ```python
    # Code to ensure the DataFrame contains the expected columns:
    # N, P, K, temperature, humidity, pH, rainfall, label
    # Code to add a 'plot_id' column (implementation details may vary)
    ```
    Checks if the loaded DataFrame has the necessary features. It also includes a step to add a `plot_id`, which might be used for tracking or as an identifier.

3.  **Handle Missing Values:**
    ```python
    # Code to impute missing numerical values in the DataFrame using the median of each column
    ```
    Addresses missing data in the numerical columns by replacing the missing values with the median of the respective column.

4.  **One-Hot Encode Crop Types:**
    ```python
    # Code to perform one-hot encoding on the 'label' column
    # This creates new binary columns for each unique crop type
    # Code to derive a 'crop_type' column (likely the original 'label' before encoding) for summarization
    ```
    Converts the categorical 'label' column (representing crop types) into a numerical format suitable for machine learning algorithms using one-hot encoding. The original 'label' is likely retained under the name 'crop_type' for easier interpretation.

5.  **Normalize Features:**
    ```python
    # Code to normalize the numerical features (N, P, K, pH, temperature, humidity, rainfall) to the range [0, 1]
    ```
    Scales the numerical features to a common range to prevent bias in distance-based algorithms.

6.  **Save Processed Data:**
    ```python
    # Code to save the processed Crop Recommendation DataFrame to 'crop_recommendation_processed.csv'
    ```
    Saves the cleaned and processed Crop Recommendation data to a CSV file.

7.  **Generate K-Distance Plot:**
    ```python
    # Code to compute k-distances (k=5) for DBSCAN and plot the curve
    # This plot is used to visually determine a suitable 'eps' value for DBSCAN
    ```
    Calculates the k-distances for the Crop Recommendation data to aid in selecting the `eps` parameter for DBSCAN.

8.  **Apply DBSCAN Clustering:**
    ```python
    # Code to apply DBSCAN clustering with eps=0.3 and min_samples=5
    ```
    Applies the DBSCAN algorithm to the processed Crop Recommendation data using an `eps` value of 0.3 and a minimum of 5 samples.

9.  **Evaluate Clusters:**
    ```python
    # Code to print the counts of data points in each identified cluster
    # This helps in understanding the outcome of the clustering process
    ```
    Prints the distribution of data points across the identified clusters for the Crop Recommendation dataset.

## Libraries Used

* TensorFlow Datasets (TFDS)
* Pandas
* Scikit-learn (for DBSCAN and normalization)
* Matplotlib or Seaborn (for plotting k-distance graphs)

## Usage

This project provides a framework for preprocessing and clustering agricultural data. The Python scripts implementing these steps can be executed sequentially. The saved processed CSV files can be used for further analysis or model development. The generated k-distance plots offer guidance on tuning the `eps` parameter for DBSCAN.

## Further Work

* Explore different clustering algorithms and compare their performance.
* Investigate different feature engineering techniques to potentially improve clustering results.
* Visualize the clusters in a meaningful way (e.g., using dimensionality reduction techniques like PCA or t-SNE).
* Integrate domain knowledge to interpret the meaning of the identified clusters.