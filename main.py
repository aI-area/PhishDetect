import pandas as pd
import numpy as np
import logging
import warnings
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from constants import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED
from utils import UrlUtils
from feature_extraction import FeatureExtractor
from ngram_processing import NgramProcessor
from feature_selection import FeatureSelector
from model import FocalLossLGBM
import os
import pickle
import random  # Global seeding

# --- Global Seeding for Reproducibility ---
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


sys.setrecursionlimit(2000)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("lightgbm").setLevel(logging.CRITICAL)

class PhishingDetector:
    """Main class for phishing detection pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.utils = UrlUtils()
        self.extractor = FeatureExtractor()
        self.ngram_processor = NgramProcessor()
        self.feature_selector = FeatureSelector()
        self.scaler = StandardScaler()
        self.all_feature_names_for_saving = None
        self.model_to_save = None
        self.train_features_selected_to_save = None
        self.test_features_scaled_to_save = None
        self.test_labels_to_save = None
        self.predictions_to_save = None
        self.selected_features_indices_to_save = None

    def load_data(self, file_path="your_path/dataset/PhishFusion.csv"):
         """Load the dataset."""
         try:
             self.logger.info("Loading dataset")
             df = pd.read_csv(file_path, encoding='latin1')

             url_list   = df['url']
             label_list = df['phishing']

             self.logger.info(f"Dataset loaded with {len(url_list)} samples.")
             return url_list, label_list

         except Exception as e:
             self.logger.error(f"Error loading data: {e}")
             return pd.Series([], dtype='object'), pd.Series([], dtype='object')

    def split_data(self, urls, labels):
        """Split dataset into training, validation, and test sets."""
        try:
            train_urls, temp_urls, train_labels, temp_labels = train_test_split(
                urls,
                labels,
                test_size=(VAL_SIZE + TEST_SIZE),
                random_state=RANDOM_SEED,
                stratify=labels
            )
            val_urls, test_urls, val_labels, test_labels = train_test_split(
                temp_urls,
                temp_labels,
                test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)),
                random_state=RANDOM_SEED,
                stratify=temp_labels
            )
            self.logger.info(f"Training set: {len(train_urls)}, Validation set: {len(val_urls)}, Test set: {len(test_urls)}")
            self.logger.info(f"Test set class distribution: {pd.Series(test_labels).value_counts().to_dict()}")
            return train_urls, val_urls, test_urls, train_labels, val_labels, test_labels

        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            return (
                pd.Series([], dtype='object'),
                pd.Series([], dtype='object'),
                pd.Series([], dtype='object'),
                pd.Series([], dtype='object'),
                pd.Series([], dtype='object'),
                pd.Series([], dtype='object')
            )

    def extract_features(self, urls):
        """Extract features for all URLs."""
        try:
            self.logger.info("Extracting features...")
            self.all_feature_names_for_saving = (
                self.extractor.FEATURE_NAMES +
                self.ngram_processor.get_feature_names()
            )
            feature_list = [
                self.extractor.generate_features(
                    url,
                    self.ngram_processor.vectorizer,
                    self.ngram_processor.selected_ngram_indices
                )
                for url in tqdm(urls)
            ]
            features_dataframe = pd.DataFrame(feature_list, columns=self.all_feature_names_for_saving)
            self.logger.info(f"Features extracted. Shape: {features_dataframe.shape}")
            return features_dataframe

        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()

    def run_pipeline(self):
        """Execute the phishing detection pipeline."""
        try:
            self.logger.info("Starting pipeline")
            urls, labels = self.load_data()
            if urls.empty:
                raise ValueError("No data loaded")

            # 1) Split original URLs for n-gram fitting
            train_urls, val_urls, test_urls, train_labels_orig_split, val_labels_orig_split, test_labels_orig_split = \
                self.split_data(urls, labels)

            # 2) Fit & select n-gram features
            self.logger.info("Processing n-grams...")
            self.ngram_processor.fit_vectorizer(train_urls)
            self.ngram_processor.select_features(
                train_urls,
                train_labels_orig_split,
                val_urls,
                val_labels_orig_split
            )

            # 3) Extract features for all URLs (handcrafted + selected n-grams)
            features_dataframe = self.extract_features(urls)

            # 4) Split this feature‑matrix into train/val/test sets
            train_features, temp_features, train_labels, temp_labels = train_test_split(
                features_dataframe,
                labels,
                test_size=(VAL_SIZE + TEST_SIZE),
                random_state=RANDOM_SEED,
                stratify=labels
            )
            val_features, test_features, val_labels, test_labels = train_test_split(
                temp_features,
                temp_labels,
                test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)),
                random_state=RANDOM_SEED,
                stratify=temp_labels
            )
            self.test_labels_to_save = test_labels

            # 5) Merged feature selection
            self.logger.info("Performing feature selection...")
            selected_features = self.feature_selector.perform_merged_selection(
                train_features,
                train_labels,
                stability_runs=20,
                sample_fraction=0.905,
                frequency_threshold=0.705,
                regularization_strength=0.191,
                mi_weight=0.318,
                alpha=0.767,
                num_features=1000,
                correlation_threshold=0.956,  
                sample_size=max(20000, len(urls)),
                random_seed=RANDOM_SEED
            )
            if len(selected_features) == 0:
                self.logger.warning(
                    "No features selected by feature_selector. Using all features from train_features as fallback."
                )
                selected_features = np.arange(train_features.shape[1])
                if len(selected_features) == 0:
                    raise ValueError("No features selected and train_features is empty.")

            self.selected_features_indices_to_save = selected_features

            # 6) Subset and scale
            train_features_selected = train_features.iloc[:, selected_features]
            test_features_selected = test_features.iloc[:, selected_features]
            self.train_features_selected_to_save = train_features_selected

            train_features_scaled = self.scaler.fit_transform(train_features_selected.astype(np.float32))
            test_features_scaled = self.scaler.transform(test_features_selected.astype(np.float32))
            self.test_features_scaled_to_save = test_features_scaled
            self.logger.info("Features scaled.")

            # 7) Train Focal‑Loss LightGBM
            self.logger.info("Training Focal Loss LightGBM...")
            model = FocalLossLGBM(
                gamma=1.21,    # Focus on hard examples
                alpha=0.42,    # Balance classes
                boosting_type='gbdt',
                num_leaves=50,
                max_depth=-1,
                learning_rate=0.15,
                n_estimators=1000,
                min_child_samples=40,
                reg_alpha=0.1,
                reg_lambda=0.3,
                verbose=-1,
                random_state=RANDOM_SEED,
                force_col_wise=True
            )
            model.fit(train_features_scaled, train_labels)
            self.model_to_save = model

            # 8) Evaluate on test set
            self.logger.info("Evaluating model on test set.")
            predictions = model.predict(test_features_scaled)
            self.predictions_to_save = predictions
            prediction_probabilities = model.predict_proba(test_features_scaled)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(test_labels, predictions),
                "Precision": precision_score(test_labels, predictions, zero_division=0),
                "Recall": recall_score(test_labels, predictions, zero_division=0),
                "F1 Score": f1_score(test_labels, predictions, zero_division=0),
                "ROC AUC": roc_auc_score(test_labels, prediction_probabilities)
            }
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")

            results_file_path = "your_path/results.txt" # Change this to your desired path
            with open(results_file_path, "w") as f:
                f.write("Test Set Metrics:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            self.logger.info(f"Metrics saved to {results_file_path}")

            # 9) Save artifacts for visualization
            self.logger.info("Saving artifacts for visualization...")
            output_data_dir = "artifacts"
            os.makedirs(output_data_dir, exist_ok=True)

            artifacts_to_save = {
                "model.pkl": self.model_to_save,
                "train_features_selected.pkl": self.train_features_selected_to_save,
                "test_features_scaled.pkl": self.test_features_scaled_to_save,
                "test_labels.pkl": self.test_labels_to_save,
                "predictions.pkl": self.predictions_to_save,
                "selected_features.pkl": self.selected_features_indices_to_save,
                "all_feature_names.pkl": self.all_feature_names_for_saving,
                "scaler.pkl": self.scaler
            }

            for filename, artifact in artifacts_to_save.items():
                if artifact is None:
                    self.logger.warning(f"Artifact '{filename}' is None and will not be saved.")
                    continue
                path = os.path.join(output_data_dir, filename)
                try:
                    with open(path, "wb") as f:
                        pickle.dump(artifact, f)
                    self.logger.info(f"Saved {filename} to {path}")
                except Exception as e:
                    self.logger.error(f"Failed to save {filename}: {e}", exc_info=True)
            self.logger.info("Artifact saving process complete.")

        except ValueError as ve:
            self.logger.error(f"Pipeline execution failed due to ValueError: {ve}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing detection pipeline")
    args = parser.parse_args()
    detector = PhishingDetector()
    detector.run_pipeline()
