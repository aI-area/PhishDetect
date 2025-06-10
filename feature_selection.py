import numpy as np
import logging
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tqdm import tqdm
from constants import RANDOM_SEED

class FeatureSelector:
    """Class for performing merged feature selection with greedy redundancy elimination."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def perform_stability_selection(self, features_np, labels_np, sample_fraction, stability_runs, regularization_strength, rng):
        """Perform stability selection using randomized logistic regression."""
        try:
            if features_np.size == 0 or labels_np.size == 0:
                raise ValueError("Features or labels are empty")
            n_features = features_np.shape[1]
            frequency_counts = np.zeros(n_features)

            for _ in tqdm(range(stability_runs), desc="Stability Selection"):
                sample_indices = rng.choice(
                    features_np.shape[0],
                    size=int(sample_fraction * features_np.shape[0]),
                    replace=False
                )
                sampled_features = features_np[sample_indices]
                sampled_labels = labels_np[sample_indices]

                model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=regularization_strength,
                    random_state=rng.randint(0, 5000)
                )
                model.fit(sampled_features, sampled_labels)
                selected = np.abs(model.coef_) > 1e-5
                if selected.ndim > 1:
                    selected = selected.ravel()
                frequency_counts += selected.astype(int)

            self.logger.info("Stability frequencies computed.")
            return frequency_counts / stability_runs

        except Exception as e:
            self.logger.error(f"Error in stability selection: {e}")
            return np.zeros(n_features)

    def perform_hybrid_selection(self, features_np, labels_np, mi_weight, n_estimators=200):
        """Perform hybrid feature selection using mutual information and XGBoost."""
        try:
            if features_np.size == 0 or labels_np.size == 0:
                raise ValueError("Features or labels are empty")
            mi_scores = mutual_info_classif(features_np, labels_np, random_state=RANDOM_SEED)
            model = XGBClassifier(
                n_estimators=n_estimators,
                random_state=RANDOM_SEED,
                tree_method="hist",
                enable_categorical=False,
                use_label_encoder=False,
                verbosity=0
            )
            model.fit(features_np, labels_np)
            xgb_scores = model.feature_importances_
            mi_scores_norm = mi_scores / (mi_scores.max() if mi_scores.max() != 0 else 1)
            xgb_scores_norm = xgb_scores / (xgb_scores.max() if xgb_scores.max() != 0 else 1)
            hybrid_scores = mi_weight * mi_scores_norm + (1 - mi_weight) * xgb_scores_norm
            self.logger.info("Hybrid importance scores computed.")
            return hybrid_scores

        except Exception as e:
            self.logger.error(f"Error in hybrid selection: {e}")
            return np.zeros(features_np.shape[1])

    def perform_merged_selection(self,
                                 features,
                                 labels,
                                 stability_runs=20,
                                 sample_fraction=0.905,
                                 frequency_threshold=0.705,
                                 regularization_strength=0.191,
                                 mi_weight=0.318,
                                 alpha=0.767,
                                 num_features=1000,
                                 correlation_threshold=0.950,
                                 sample_size=20000,
                                 random_seed=RANDOM_SEED,
                                 n_estimators=200
                                ):
        """Perform merged feature selection with stability, hybrid methods, and greedy redundancy elimination."""
        try:
            if features.empty or features.shape[1] == 0:
                raise ValueError("Features DataFrame is empty or has no columns")
            features_np = features.values.astype(np.float32)
            labels_np = labels.values
            rng = np.random.RandomState(random_seed)

            # 1) Stability selection
            stability_frequencies = self.perform_stability_selection(
                features_np, labels_np, sample_fraction, stability_runs, regularization_strength, rng
            )
            if stability_frequencies.size == 0 or np.all(stability_frequencies == 0):
                raise ValueError("No stable features selected")

            # 2) Hybrid selection
            hybrid_scores = self.perform_hybrid_selection(
                features_np, labels_np, mi_weight, n_estimators
            )
            if hybrid_scores.size == 0 or np.all(hybrid_scores == 0):
                raise ValueError("No hybrid features selected")

            # 3) Combine scores
            merged_scores = alpha * stability_frequencies + (1 - alpha) * hybrid_scores
            if merged_scores.size == 0:
                raise ValueError("Merged scores are empty")

            # 4) Filter by frequency threshold
            candidate_indices = np.where(stability_frequencies >= frequency_threshold)[0]
            if candidate_indices.size == 0:
                self.logger.warning("No features meet frequency threshold; using all features")
                candidate_indices = np.arange(len(merged_scores))

            # 5) Sort by merged score
            sorted_indices = candidate_indices[np.argsort(merged_scores[candidate_indices])[::-1]]

            # 6) Greedy redundancy elimination
            selected_features = []
            correlations = {}
            sample_indices = rng.choice(
                features_np.shape[0],
                size=min(sample_size, features_np.shape[0]),
                replace=False
            )
            for idx in tqdm(sorted_indices, desc="Greedy Feature Selection"):
                if len(selected_features) >= num_features:
                    break
                current_feature = features_np[sample_indices, idx]
                max_corr = 0.0
                for sel in selected_features:
                    if (idx, sel) not in correlations:
                        sel_feature = features_np[sample_indices, sel]
                        corr = np.corrcoef(current_feature, sel_feature)[0, 1]
                        correlations[(idx, sel)] = abs(corr)
                        correlations[(sel, idx)] = abs(corr)
                    else:
                        corr = correlations[(idx, sel)]
                    max_corr = max(max_corr, corr)

                if max_corr < correlation_threshold:
                    selected_features.append(idx)

            self.logger.info(f"Selected {len(selected_features)} features.")
            return np.array(selected_features, dtype=int)

        except Exception as e:
            self.logger.error(f"Error during merged selection: {e}")
            return np.array([], dtype=int)