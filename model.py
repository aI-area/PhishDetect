

import numpy as np
import logging
from scipy.special import expit
from lightgbm import LGBMClassifier

class FocalLossLGBM(LGBMClassifier):
    """LightGBM with custom focal loss."""

    def __init__(
        self,
        gamma=1.55,       # As in monolithic
        alpha=0.59,       # As in monolithic
        boosting_type='gbdt',
        num_leaves=50,
        max_depth=-1,
        learning_rate=0.15,
        n_estimators=1000,
        min_child_samples=40,
        reg_alpha=0.1,
        reg_lambda=0.3,
        verbosity=-1,
        random_state=42,
        **kwargs
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.logger = logging.getLogger(__name__)
        params = {
            'boosting_type': boosting_type,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'verbosity': verbosity,
            'random_state': random_state,
            'num_class': 1  # Binary classification
        }
        kwargs.pop('objective', None)
        params.update(kwargs)
        super().__init__(objective=self.custom_obj, **params)

    def custom_obj(self, y_true, y_pred_logits):
        """Focal loss gradient and hessian calculation."""
        try:
            p = expit(y_pred_logits)
            y_true = y_true.astype(np.float64)

            # Calculate pt and clip to avoid numerical instability
            pt = y_true * p + (1 - y_true) * (1 - p)
            pt = np.clip(pt, 1e-15, 1 - 1e-15)

            alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

            # Focal weight
            focal_weight = (1 - pt) ** (self.gamma - 1)

            # Gradient factor
            grad_factor = self.gamma * np.log(pt) - (1 - pt) / pt

            # Final gradient
            grad = alpha_weight * (2 * y_true - 1) * p * (1 - p) * focal_weight * grad_factor

            # Hessian approximation
            hess = p * (1 - p) * focal_weight * alpha_weight

            return np.nan_to_num(grad), np.nan_to_num(hess)
        except Exception as e:
            self.logger.error(f"Error in custom objective: {e}")
            return np.zeros_like(y_pred_logits), np.zeros_like(y_pred_logits)

    def predict_proba(self, X, **kwargs):
        """Convert logits to probabilities using sigmoid."""
        try:
            raw_logits = self.booster_.predict(X, raw_score=True, **kwargs)
            prob_class_1 = expit(raw_logits)
            prob_class_0 = 1 - prob_class_1
            return np.vstack((prob_class_0, prob_class_1)).T
        except Exception as e:
            self.logger.error(f"Error in predict_proba: {e}")
            return np.zeros((X.shape[0], 2))

    def predict(self, X, **kwargs):
        """Predict class labels based on probability threshold."""
        try:
            probs = self.predict_proba(X, **kwargs)
            return (probs[:, 1] >= 0.5).astype(int)
        except Exception as e:
            self.logger.error(f"Error in predict: {e}")
            return np.zeros(X.shape[0], dtype=int)
