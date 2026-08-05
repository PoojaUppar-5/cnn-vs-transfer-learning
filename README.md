# CNN vs Transfer Learning — Image Classification

Deep Neural Networks (AIMLCZG511, BITS Pilani WILP) — Assignment 2. Compares a custom-built Convolutional Neural Network against a ResNet50-based transfer learning model on the [Cats vs Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) dataset (`tensorflow_datasets`, ~23k images, 2 classes).

## Contents

- [`cnn.ipynb`](cnn.ipynb) — end-to-end notebook covering data loading/exploration, preprocessing, model building, training, evaluation, and comparison.

## Approach

1. **Data pipeline** — load `cats_vs_dogs` via `tfds`, explore class balance and image statistics, resize to 224×224 and normalize.
2. **Custom CNN** — a Sequential Keras model with built-in data augmentation (random flip/rotation/zoom), convolutional blocks, and global average pooling, trained from scratch with early stopping.
3. **Transfer learning** — ResNet50 pretrained on ImageNet, trained in two phases: (1) frozen base with classification head only, (2) fine-tuning.
4. **Evaluation** — accuracy, precision, recall, F1, confusion matrices, training curves, and sample predictions for both models.
5. **Comparison** — side-by-side metrics, parameter counts, and training time.

## Results

| Metric | Custom CNN | Transfer Learning (ResNet50) |
|---|---|---|
| Accuracy | 0.9342 | 0.9918 |
| Precision | 0.9342 | 0.9918 |
| Recall | 0.9342 | 0.9918 |
| F1-score | 0.9342 | 0.9918 |
| Trainable parameters | 390,850 | 266,626 |
| Training time | ~1917s | ~1399s |

ResNet50 outperformed the custom CNN by a uniform 0.0576 across all metrics, benefiting from ImageNet-pretrained features. The custom CNN, trained entirely from scratch, converged more slowly and trained longer despite having more trainable parameters, since ResNet50's frozen base skips backpropagation through most of the network. See Part 5 (Analysis) in the notebook for the full discussion.

## Requirements

- Python 3
- TensorFlow / Keras
- tensorflow-datasets
- numpy, pandas, scikit-learn
- matplotlib, seaborn

## Usage

Open [`cnn.ipynb`](cnn.ipynb) in Jupyter, Google Colab, or the BITS Virtual Lab and run all cells top to bottom. A GPU runtime is recommended for training.
