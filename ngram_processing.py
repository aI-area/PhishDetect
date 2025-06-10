# File: ngram_processing.py
# :contentReference[oaicite:1]{index=1}

import numpy as np
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from constants import RANDOM_SEED

class NgramProcessor:
    """Class for processing n-gram features from URLs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vectorizer = None
        self.variance_filter = None
        self.selected_ngram_indices = None

    def fit_vectorizer(self, urls, max_df=0.9, min_df=8, ngram_range=(2, 3)):
        """Fit CountVectorizer on URLs."""
        try:
            self.vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df
            )
            self.vectorizer.fit(urls)
            self.logger.info(f"Total n-gram features: {len(self.vectorizer.vocabulary_)}")
        except Exception as e:
            self.logger.error(f"Error fitting vectorizer: {e}")

    def select_features(self,
                        train_urls,
                        train_labels,
                        val_urls,
                        val_labels,
                        top_ngram_count=5000  # ← Changed default from 4000 to 5000
                       ):
        """Select top n-gram features based on MI and Logistic Regression."""
        try:
            # 1) Transform and apply variance threshold
            ngram_matrix_train = self.vectorizer.transform(train_urls)
            self.variance_filter = VarianceThreshold(threshold=0.00005)
            ngram_matrix_filtered_train = self.variance_filter.fit_transform(ngram_matrix_train)
            self.logger.info(f"Reduced n-gram features to {ngram_matrix_filtered_train.shape[1]}")

            # 2) Compute MI scores and train L1-LR
            mi_scores_ngram = mutual_info_classif(ngram_matrix_filtered_train,
                                                  train_labels,
                                                  random_state=RANDOM_SEED)
            lr_ngram = LogisticRegression(penalty='l1',
                                          solver='liblinear',
                                          C=0.5,
                                          random_state=RANDOM_SEED)
            lr_ngram.fit(ngram_matrix_filtered_train, train_labels)

            lr_coefs = np.abs(lr_ngram.coef_).flatten()
            mi_norm = mi_scores_ngram / (mi_scores_ngram.max() if mi_scores_ngram.max() != 0 else 1)
            lr_norm = lr_coefs / (lr_coefs.max() if lr_coefs.max() != 0 else 1)
            combined_scores = 0.5 * mi_norm + 0.5 * lr_norm

            # 3) Pick top indices
            selected_in_variance = np.argsort(combined_scores)[::-1][:top_ngram_count]
            variance_mask_indices = np.where(self.variance_filter.get_support())[0]
            self.selected_ngram_indices = variance_mask_indices[selected_in_variance]
            self.logger.info(f"Selected {len(self.selected_ngram_indices)} n-gram features.")

            # 4) Validate on validation set
            ngram_matrix_val = self.vectorizer.transform(val_urls)
            ngram_matrix_filtered_val = self.variance_filter.transform(ngram_matrix_val)
            selected_ngram_matrix_val = ngram_matrix_filtered_val[:, self.selected_ngram_indices]

            lr_final = LogisticRegression(penalty='l1',
                                          solver='liblinear',
                                          C=5,
                                          random_state=RANDOM_SEED)
            lr_final.fit(ngram_matrix_filtered_train[:, self.selected_ngram_indices], train_labels)
            val_preds = lr_final.predict(selected_ngram_matrix_val)
            self.logger.info(
                f"N-gram Validation Accuracy: {accuracy_score(val_labels, val_preds):.4f}, "
                f"F1-Score: {f1_score(val_labels, val_preds, average='weighted'):.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error selecting n-gram features: {e}")

    def get_feature_names(self):
        """Return selected n-gram feature names."""
        if self.vectorizer and (self.selected_ngram_indices is not None):
            return self.vectorizer.get_feature_names_out()[self.selected_ngram_indices].tolist()
        return []
