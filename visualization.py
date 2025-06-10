
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import argparse
import logging
import os  # Needed for os.path.join
import pickle  # Needed in __main__
from feature_extraction import FeatureExtractor  # Assuming this file exists and is correct
import sys  # For __main__

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Visualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def visualize_handcrafted_feature_importance(
        self, model, selected_features, all_feature_names
    ):
        """Plot top 60 handcrafted feature importances."""
        self.logger.info("Visualizing handcrafted feature importance...")
        valid_indices = [i for i in selected_features if i < len(all_feature_names)]
        if len(valid_indices) != len(selected_features):
            self.logger.warning("Some selected_features indices are out of bounds for all_feature_names.")
        feature_names_for_plot = [all_feature_names[i] for i in valid_indices]
        if not feature_names_for_plot:
            self.logger.warning("No valid feature names could be derived for handcrafted importance plot.")
            return

        try:
            feature_importances = model.feature_importances_
            if len(feature_names_for_plot) != len(feature_importances):
                self.logger.warning(
                    f"Length mismatch: {len(feature_names_for_plot)} names vs {len(feature_importances)} importances. "
                    "This might be due to invalid selected_feature indices. Plotting with available."
                )
                min_len = min(len(feature_names_for_plot), len(feature_importances))
                feature_names_for_plot = feature_names_for_plot[:min_len]
                feature_importances = feature_importances[:min_len]
                if not feature_names_for_plot:
                    self.logger.error("No features left after aligning names and importances.")
                    return
        except AttributeError:
            self.logger.error("Model does not have 'feature_importances_' attribute.")
            return

        importance_df = pd.DataFrame({
            'Feature': feature_names_for_plot,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        handcrafted_mask = importance_df['Feature'].isin(FeatureExtractor.FEATURE_NAMES)
        handcrafted_df = importance_df[handcrafted_mask].head(60)

        if handcrafted_df.empty:
            self.logger.info("No handcrafted features found among the selected important features to plot.")
            return

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(
            data=handcrafted_df,
            x='Importance',
            y='Feature',
            palette='viridis'
        )
        plt.title(f'Top {len(handcrafted_df)} Handcrafted Features (Selected by Pipeline)', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('')
        plt.tick_params(axis='y', labelsize=9)
        for i, (_, row) in enumerate(handcrafted_df.iterrows()):
            ax.text(
                row['Importance'] + 0.005,
                i,
                f"{row['Importance']:.2f}",
                va='center',
                fontsize=7,
                bbox=dict(facecolor='none', alpha=0.7, pad=5, edgecolor='none')
            )
        plt.grid(axis='x', alpha=0.3, linestyle='')
        plt.tight_layout()
        plt.savefig("Handcrafted_Features.png")
        self.logger.info("Saved Handcrafted_Features.png")
        plt.close()

    def visualize_character_importance(
        self, model, train_features_selected_df 
    ):
        """Plot top 60 n-gram feature importances in four subplots."""
        self.logger.info("Visualizing n-gram character importance...")
        if not hasattr(model, 'feature_importances_'):
            self.logger.error("Model does not have 'feature_importances_' attribute.")
            return
        if not isinstance(train_features_selected_df, pd.DataFrame):
            self.logger.error("train_features_selected must be a pandas DataFrame for n-gram visualization.")
            return

        feature_importances = model.feature_importances_
        feature_names = train_features_selected_df.columns.tolist()

        if len(feature_names) != len(feature_importances):
            self.logger.error(
                f"Mismatch: {len(feature_names)} feature names from train_features_selected.pkl "
                f"(shape: {train_features_selected_df.shape}), "
                f"but model has {len(feature_importances)} importances (model n_features_in_: {getattr(model, 'n_features_in_', 'N/A')})."
            )
            return

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        is_ngram = ~importance_df['Feature'].isin(FeatureExtractor.FEATURE_NAMES)
        ngram_df = importance_df[is_ngram].head(60)

        if ngram_df.empty:
            self.logger.info("No n-gram features found among the selected important features to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 8))
        plt.subplots_adjust(hspace=0.6, wspace=0.4)
        ranges = [(0, 15), (15, 30), (30, 45), (45, 60)]
        titles = ['Top 1-15 N-grams', 'Top 16-30 N-grams', 'Top 31-45 N-grams', 'Top 46-60 N-grams']
        for ax, (start, end), title in zip(axes.flat, ranges, titles):
            subset = ngram_df.iloc[start:end]
            if subset.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=subset,
                y='Feature',
                x='Importance',
                palette='rocket',
                ax=ax
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Importance Score', fontsize=11)
            ax.tick_params(axis='y', labelsize=9)
            for i, (_, row) in enumerate(subset.iterrows()):
                ax.text(
                    row['Importance'] + 0.002,
                    i,
                    f"{row['Importance']:.1f}",
                    va='center',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none')
                )
            if any(len(str(f)) > 8 for f in subset['Feature']):
                ax.tick_params(axis='y', rotation=20)
            ax.grid(axis='x', alpha=0.2, linestyle='')
        plt.suptitle('Top 60 N-gram Features by Importance', fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig("Ngram_Features.png")
        self.logger.info("Saved Ngram_Features.png")
        plt.close()

    def visualize_shap_summary_plot(
        self, model, test_features_scaled_np, selected_feature_indices, all_feature_names_list
    ):
        self.logger.info("Visualizing SHAP summary plot...")
        if not isinstance(test_features_scaled_np, np.ndarray):
            self.logger.error("test_features_scaled must be a NumPy array for SHAP summary.")
            return

        final_selected_names = [
            all_feature_names_list[i] for i in selected_feature_indices if i < len(all_feature_names_list)
        ]
        if len(final_selected_names) != test_features_scaled_np.shape[1]:
            self.logger.error(
                f"Shape mismatch for SHAP summary. Expected {len(final_selected_names)} cols from names, "
                f"got {test_features_scaled_np.shape[1]} in data."
            )
            return

        test_features_df = pd.DataFrame(test_features_scaled_np, columns=final_selected_names)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_features_df)
        except Exception as e:
            self.logger.error(f"Error during SHAP explanation: {e}")
            return

        class1_shap_values = (
            shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
        )
        plt.figure()
        shap.summary_plot(
            class1_shap_values,
            test_features_df,
            plot_type="dot",
            max_display=40,
            show=False
        )
        plt.title("SHAP Feature Importance (Phishing Detection)", fontsize=14)
        plt.xlabel("SHAP Value Impact on Prediction", fontsize=12)
        plt.gcf().set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig("SHAP_Summary.png")
        self.logger.info("Saved SHAP_Summary.png")
        plt.close()

    def visualize_shap_decision_plot(
        self, model, test_features_scaled_np, test_labels_series, selected_feature_indices, all_feature_names_list, instance_idx=0
    ):
        self.logger.info(f"Visualizing SHAP decision plot for instance {instance_idx} (0-indexed)...")
        final_selected_names = [
            all_feature_names_list[i] for i in selected_feature_indices if i < len(all_feature_names_list)
        ]
        if len(final_selected_names) != test_features_scaled_np.shape[1]:
            self.logger.error(
                f"Shape mismatch for SHAP decision. Expected {len(final_selected_names)} cols, got "
                f"{test_features_scaled_np.shape[1]}."
            )
            return
        if instance_idx >= test_features_scaled_np.shape[0]:
            self.logger.error(
                f"Instance index {instance_idx} is out of bounds for test data size {test_features_scaled_np.shape[0]}. "
                "Using index 0."
            )
            instance_idx = 0
        if test_features_scaled_np.shape[0] == 0:
            self.logger.error("Test features array is empty for SHAP decision plot.")
            return

        test_features_df = pd.DataFrame(test_features_scaled_np, columns=final_selected_names)
        test_labels_np = (
            test_labels_series.values if isinstance(test_labels_series, pd.Series) else test_labels_series
        )

        actual_instance_to_plot_idx = instance_idx
        true_label_instance = test_labels_np[actual_instance_to_plot_idx]
        self.logger.info(
            f"Plotting SHAP decision for instance at position {actual_instance_to_plot_idx} "
            f"(True Label: {true_label_instance})"
        )

        try:
            explainer = shap.TreeExplainer(model)
            shap_values_instance_set = explainer.shap_values(
                test_features_df.iloc[[actual_instance_to_plot_idx]]
            )
            expected_value_explainer = explainer.expected_value
        except Exception as e:
            self.logger.error(f"Error during SHAP explanation for decision plot: {e}")
            return

        class1_shap_values_single_instance = (
            shap_values_instance_set[1][0]
            if isinstance(shap_values_instance_set, list) and len(shap_values_instance_set) == 2
            else shap_values_instance_set[0]
        )
        base_value = (
            expected_value_explainer[1]
            if isinstance(expected_value_explainer, (list, np.ndarray)) and len(expected_value_explainer) == 2
            else expected_value_explainer
        )

        plt.figure()
        shap.decision_plot(
            base_value=base_value,
            shap_values=class1_shap_values_single_instance,
            features=test_features_df.iloc[actual_instance_to_plot_idx],
            feature_names=test_features_df.columns.tolist(),
            show=False
        )
        plt.suptitle(
            f"SHAP Decision Plot - Instance {actual_instance_to_plot_idx} "
            f"(True Label: {true_label_instance})",
            fontsize=12
        )
        plt.gcf().set_size_inches(12, 6)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"SHAP_Decision_Instance_{actual_instance_to_plot_idx}.png")
        self.logger.info(f"Saved SHAP_Decision_Instance_{actual_instance_to_plot_idx}.png")
        plt.close()

    def visualize_shap_waterfall_plot(
        self, model, test_features_scaled_np, test_labels_series, predictions_np, selected_feature_indices, all_feature_names_list, instance_idx=1
    ):
        self.logger.info(f"Visualizing SHAP waterfall plot for instance index {instance_idx}...")
        final_selected_names = [
            all_feature_names_list[i] for i in selected_feature_indices if i < len(all_feature_names_list)
        ]
        if len(final_selected_names) != test_features_scaled_np.shape[1]:
            self.logger.error(
                f"Shape mismatch for SHAP waterfall. Expected {len(final_selected_names)} cols, got "
                f"{test_features_scaled_np.shape[1]}."
            )
            return
        if instance_idx >= test_features_scaled_np.shape[0]:
            self.logger.error(
                f"Instance index {instance_idx} out of bounds for test data size {test_features_scaled_np.shape[0]}."
            )
            return

        test_features_df = pd.DataFrame(test_features_scaled_np, columns=final_selected_names)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_single_instance_set = explainer.shap_values(
                test_features_df.iloc[[instance_idx]]
            )
            expected_value_explainer = explainer.expected_value
        except Exception as e:
            self.logger.error(f"Error during SHAP explanation for waterfall plot: {e}")
            return

        class1_shap_values_instance = (
            shap_values_single_instance_set[1][0]
            if isinstance(shap_values_single_instance_set, list) and len(shap_values_single_instance_set) == 2
            else shap_values_single_instance_set[0]
        )
        base_value = (
            expected_value_explainer[1]
            if isinstance(expected_value_explainer, (list, np.ndarray)) and len(expected_value_explainer) == 2
            else expected_value_explainer
        )

        actual_prediction = predictions_np[instance_idx]
        true_label = (
            test_labels_series.iloc[instance_idx]
            if isinstance(test_labels_series, pd.Series) else test_labels_series[instance_idx]
        )

        shap_explanation_obj = shap.Explanation(
            values=class1_shap_values_instance,
            base_values=base_value,
            data=test_features_df.iloc[instance_idx].values,
            feature_names=test_features_df.columns.tolist()
        )
        plt.figure()
        shap.waterfall_plot(shap_explanation_obj, max_display=30, show=False)
        plt.title(
            f"SHAP Waterfall - Instance {instance_idx} (Pred: {actual_prediction}, True: {true_label})",
            fontsize=14
        )
        plt.gcf().set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(f"SHAP_Waterfall_Instance_{instance_idx}.png")
        self.logger.info(f"Saved SHAP_Waterfall_Instance_{instance_idx}.png")
        plt.close()

    def visualize_shap_scatter_plots(
        self, model, test_features_scaled_np, selected_feature_indices, all_feature_names_list, top_n_features=5
    ):
        """
        Plot SHAP scatter for a specific feature index (default: top feature).
        The user can override 'feature_idx' in the __main__ section by providing --instance_idx.
        """
        feature_idx = self.feature_idx_for_scatter if hasattr(self, 'feature_idx_for_scatter') else 0
        final_selected_names = [
            all_feature_names_list[i] for i in selected_feature_indices if i < len(all_feature_names_list)
        ]
        if feature_idx < len(final_selected_names):
            feature_name = final_selected_names[feature_idx]
            self.logger.info(f"Visualizing SHAP scatter plot for feature '{feature_name}' (index {feature_idx})...")
        else:
            self.logger.info(f"Visualizing SHAP scatter plot for feature index {feature_idx}...")

        if len(final_selected_names) != test_features_scaled_np.shape[1]:
            self.logger.error(
                f"Shape mismatch for SHAP scatter. Expected {len(final_selected_names)} cols, got "
                f"{test_features_scaled_np.shape[1]}."
            )
            return

        test_features_df = pd.DataFrame(test_features_scaled_np, columns=final_selected_names)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_all_instances = explainer.shap_values(test_features_df)
            expected_value_explainer = explainer.expected_value
        except Exception as e:
            self.logger.error(f"Error during SHAP explanation for scatter plots: {e}")
            return

        class1_shap_values_all = (
            shap_values_all_instances[1]
            if isinstance(shap_values_all_instances, list) and len(shap_values_all_instances) == 2
            else shap_values_all_instances
        )
        base_value_for_plot = (
            expected_value_explainer[1]
            if isinstance(expected_value_explainer, (list, np.ndarray)) and len(expected_value_explainer) == 2
            else expected_value_explainer
        )

        if feature_idx >= test_features_df.shape[1]:
            self.logger.error(
                f"Feature index {feature_idx} is out of bounds for test data (width {test_features_df.shape[1]})."
            )
            return

        feature_name = test_features_df.columns[feature_idx]

        # Build a 2D Explanation object for the chosen feature
        shap_explanation = shap.Explanation(
            values=class1_shap_values_all[:, feature_idx].reshape(-1, 1),
            base_values=base_value_for_plot,
            data=test_features_df[feature_name].values.reshape(-1, 1),
            feature_names=[feature_name]
        )

        plt.figure(figsize=(12, 6))
        shap.plots.scatter(
            shap_explanation,
            color=shap_explanation,
            show=False
        )
        plt.title(f"SHAP Scatter Plot: {feature_name} Impact", fontsize=14)
        plt.xlabel(f"{feature_name} Value", fontsize=12)
        plt.ylabel("SHAP Value for Phishing Prediction", fontsize=12)
        plt.tight_layout()
        safe_feature_name = "".join(c if c.isalnum() else "_" for c in feature_name)
        plt.savefig(f"SHAP_Scatter_{safe_feature_name}.png")
        self.logger.info(f"Saved SHAP_Scatter_{safe_feature_name}.png")
        plt.close()

if __name__ == "__main__":
    # --- BEGIN DEBUG ---
    print(f"DEBUG: Raw command line arguments received by script: {sys.argv}")
    # --- END DEBUG ---

    parser = argparse.ArgumentParser(description="Visualization for phishing detection pipeline")
    parser.add_argument(
        '--viz_type',
        choices=[
            'handcrafted_feature_importance',
            'character_importance',
            'shap_summary_plot',
            'shap_decision_plot',
            'shap_waterfall_plot',
            'shap_scatter_plots'
        ],
        required=True,
        dest='plot_type',
        help="Type of visualization to generate"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="artifacts",
        help="Directory containing the saved .pkl artifacts (default: artifacts/)"
    )
    parser.add_argument(
        "--instance_idx",
        type=int,
        default=0,
        help="Index of the instance or feature to use for instance-specific plots."
    )
    args = parser.parse_args()
    # --- BEGIN DEBUG ---
    print(f"DEBUG: Parsed arguments: {args}")
    # --- END DEBUG ---

    visualizer = Visualizer()
    data_path = args.data_dir
    try:
        with open(os.path.join(data_path, "model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(data_path, "train_features_selected.pkl"), "rb") as f:
            train_features_selected_df = pickle.load(f)
        with open(os.path.join(data_path, "test_features_scaled.pkl"), "rb") as f:
            test_features_scaled_np = pickle.load(f)
        with open(os.path.join(data_path, "test_labels.pkl"), "rb") as f:
            test_labels_series = pickle.load(f)
        with open(os.path.join(data_path, "predictions.pkl"), "rb") as f:
            predictions_np = pickle.load(f)
        with open(os.path.join(data_path, "selected_features.pkl"), "rb") as f:
            selected_feature_indices = pickle.load(f)
        with open(os.path.join(data_path, "all_feature_names.pkl"), "rb") as f:
            all_feature_names_list = pickle.load(f)
    except FileNotFoundError as e:
        logging.error(
            f"Error loading data artifacts from '{data_path}': {e}. "
            "Please ensure main.py has run and saved artifacts to the correct directory."
        )
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred loading data: {e}")
        exit(1)

    # If plotting SHAP scatter, set the feature_idx inside the visualizer
    if args.plot_type == 'shap_scatter_plots':
        visualizer.feature_idx_for_scatter = args.instance_idx

    if args.plot_type == 'handcrafted_feature_importance':
        visualizer.visualize_handcrafted_feature_importance(
            model, selected_feature_indices, all_feature_names_list
        )
    elif args.plot_type == 'character_importance':
        visualizer.visualize_character_importance(
            model, train_features_selected_df
        )
    elif args.plot_type == 'shap_summary_plot':
        visualizer.visualize_shap_summary_plot(
            model, test_features_scaled_np, selected_feature_indices, all_feature_names_list
        )
    elif args.plot_type == 'shap_decision_plot':
        visualizer.visualize_shap_decision_plot(
            model,
            test_features_scaled_np,
            test_labels_series,
            selected_feature_indices,
            all_feature_names_list,
            instance_idx=args.instance_idx
        )
    elif args.plot_type == 'shap_waterfall_plot':
        visualizer.visualize_shap_waterfall_plot(
            model,
            test_features_scaled_np,
            test_labels_series,
            predictions_np,
            selected_feature_indices,
            all_feature_names_list,
            instance_idx=args.instance_idx
        )
    elif args.plot_type == 'shap_scatter_plots':
        visualizer.visualize_shap_scatter_plots(
            model,
            test_features_scaled_np,
            selected_feature_indices,
            all_feature_names_list
        )
    else:
        logging.error(f"Unknown plot type provided: {args.plot_type}")
