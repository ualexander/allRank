#!/usr/bin/env python3
"""
Inference script that preserves option labels for decision-making
"""

import torch
import os
import shutil
import csv
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


def predict_with_labels(model, config, test_ds):
    """
    Get prediction scores for options

    Args:
        model: Trained model
        config: Model config
        test_ds: Test dataset

    Returns:
        List of results for CSV output
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

                qid = batch_idx * config.data.batch_size + query_idx

                for doc_idx in range(len(query_scores)):
                    if mask[query_idx, doc_idx]:  # Skip padded documents
                        continue

                    # Map back to original position using indices
                    original_position = query_indices[doc_idx]
                    pred_score = query_scores[doc_idx]
                    true_relevance = query_labels[doc_idx]

                    results.append(
                        {
                            "qid": qid,
                            "original_position": original_position,
                            "predicted_score": pred_score,
                            "true_relevance": true_relevance,
                        }
                    )

    return results


def run_labeled_inference(
    model_path: str,
    config_path: str,
    test_data_path: str,
    output_csv: str = "results.csv",
):
    """Run inference and save results to CSV"""
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

    # Run inference
    print("Running inference...")
    results = predict_with_labels(model, config, test_ds)

    # Write to CSV
    print(f"Writing results to {output_csv}")
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["qid", "original_position", "predicted_score", "true_relevance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"Results saved to {output_csv}")
    return results


if __name__ == "__main__":
    # Paths
    model_path = "./test_run/results/test_run/model.pkl"
    config_path = "./test_run/results/test_run/used_config.json"
    test_data_path = "./test_data_5_options.txt"

    # Run inference and save to CSV
    results = run_labeled_inference(
        model_path, config_path, test_data_path, "results.csv"
    )
