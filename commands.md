# Dummy data training (relative path)

python allrank/main.py --config-file-name scripts/local_config.json --run-id test_run --job-dir ./test_run

How the Inference Works

1. Uses rank_slates() from allRank: The script calls rank_slates(datasets, model, config) from
   allrank.inference.inference_utils. This is the official inference function that comes with the allRank
   repository.
2. Model Loading: It loads the trained PyTorch model from model.pkl and reconstructs the model architecture
   using the same config that was used during training.
3. Data Processing: The test data goes through the same preprocessing pipeline as training data (LibSVM format
   parsing, slate padding, etc.).

The X and Y Returns Explained

The rank_slates() function returns (X, y) tuples where:

- X (Features): torch.Tensor of shape [num_queries, slate_length, num_features]
  - Contains the reordered document features
  - Documents are sorted by their predicted relevance scores (highest first)
  - Shape: [1, 3, 20] in your case = 1 query, 3 documents, 20 features each
- y (Labels): torch.Tensor of shape [num_queries, slate_length]
  - Contains the reordered relevance labels
  - Shows the true relevance scores in the new ranked order
  - Your result [3.0, 2.0, 1.0] means the model correctly ranked:
    - Most relevant document (score=3) first
    - Medium relevant document (score=2) second
    - Least relevant document (score=1) last

What Actually Happened

The model took your 3 test documents and reordered them by predicted relevance. The fact that the output shows
[3.0, 2.0, 1.0] means the model successfully learned to rank documents correctly - it put the highest
relevance document first!

This is exactly what a learning-to-rank model should do: take a set of documents for a query and reorder them
by predicted relevance.
