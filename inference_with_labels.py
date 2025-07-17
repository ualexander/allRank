#!/usr/bin/env python3
"""
Inference script that preserves option labels for decision-making
"""

import torch
import os
import shutil
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset_role
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device
from allrank.models import losses
from attr import asdict
from torch.utils.data import DataLoader


def load_trained_model(model_path: str, config_path: str, test_ds):
    """Load trained model and config"""
    config = Config.from_json(config_path)
    n_features = test_ds.shape[-1]

    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))

    device = get_torch_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    return model, config


def predict_with_labels(model, config, test_ds, option_labels=None):
    """
    Get prediction scores while preserving option order/labels

    Args:
        model: Trained model
        config: Model config
        test_ds: Test dataset
        option_labels: List of labels for each option (e.g., ['Option A', 'Option B', 'Option C'])

    Returns:
        List of tuples: [(option_label, original_position, predicted_score, true_relevance)]
    """
    dataloader = DataLoader(
        test_ds, batch_size=config.data.batch_size, num_workers=1, shuffle=False
    )

    device = get_torch_device()
    results = []

    with torch.no_grad():
        for batch_idx, (xb, yb, indices) in enumerate(dataloader):
            X = xb.type(torch.float32).to(device=device)
            y_true = yb.to(device=device)

            # Create input indices and mask
            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = y_true == losses.PADDED_Y_VALUE

            # Get prediction scores (without reordering)
            scores = model.score(X, mask, input_indices)

            # Process each query
            for query_idx in range(X.shape[0]):
                query_scores = scores[query_idx].cpu().numpy()
                query_labels = y_true[query_idx].cpu().numpy()
                query_indices = indices[query_idx].cpu().numpy()

                query_results = []
                for doc_idx in range(len(query_scores)):
                    if mask[query_idx, doc_idx]:  # Skip padded documents
                        continue

                    # Map back to original position using indices
                    original_position = query_indices[doc_idx]
                    option_label = (
                        option_labels[original_position] if option_labels else f"Option_{original_position}"
                    )
                    pred_score = query_scores[doc_idx]
                    true_relevance = query_labels[doc_idx]

                    query_results.append(
                        (option_label, original_position, pred_score, true_relevance)
                    )

                # Sort by predicted score (highest first) to get rankings
                query_results.sort(key=lambda x: x[2], reverse=True)

                # Add rank information
                ranked_results = []
                for rank, (label, orig_pos, score, true_rel) in enumerate(
                    query_results, 1
                ):
                    ranked_results.append(
                        {
                            "option_label": label,
                            "original_position": orig_pos,
                            "predicted_rank": rank,
                            "predicted_score": score,
                            "true_relevance": true_rel,
                        }
                    )

                results.append(ranked_results)

    return results


def run_labeled_inference(
    model_path: str, config_path: str, test_data_path: str, option_labels=None
):
    """Run inference while preserving option labels"""
    print(f"Loading config from: {config_path}")
    config = Config.from_json(config_path)

    # Load test data
    print(f"Loading test data from: {test_data_path}")
    temp_dir = "./temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(test_data_path, os.path.join(temp_dir, "test.txt"))

    test_ds = load_libsvm_dataset_role("test", temp_dir, config.data.slate_length)

    print(f"Loading model from: {model_path}")
    model, config = load_trained_model(model_path, config_path, test_ds)

    # Run inference with labels
    print("Running labeled inference...")
    results = predict_with_labels(model, config, test_ds, option_labels)

    # Print results
    for query_idx, query_results in enumerate(results):
        print(f"\nQuery {query_idx + 1} Results:")
        print(
            "Rank | Option Label | Predicted Score | True Relevance | Original Position"
        )
        print("-" * 75)

        for result in query_results:
            print(
                f"{result['predicted_rank']:4d} | {result['option_label']:12s} | "
                f"{result['predicted_score']:14.4f} | {result['true_relevance']:13.1f} | "
                f"{result['original_position']:17d}"
            )

        # Show recommendation
        best_option = query_results[0]  # Highest ranked
        print(
            f"\n🏆 Recommended choice: {best_option['option_label']} "
            f"(Score: {best_option['predicted_score']:.4f})"
        )

        # Check if model prediction matches ground truth
        true_best = max(query_results, key=lambda x: x["true_relevance"])
        if best_option["option_label"] == true_best["option_label"]:
            print("✅ Model prediction matches ground truth!")
        else:
            print(
                f"❌ Model predicted {best_option['option_label']}, "
                f"but ground truth best is {true_best['option_label']}"
            )

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return results


if __name__ == "__main__":
    # Paths
    model_path = "./test_run/results/test_run/model.pkl"
    config_path = "./test_run/results/test_run/used_config.json"
    test_data_path = "./test_data.txt"

    # Define your option labels
    option_labels = ["Option A", "Option B", "Option C"]

    # Run labeled inference
    results = run_labeled_inference(
        model_path, config_path, test_data_path, option_labels
    )
    print("\nLabeled inference completed!")
