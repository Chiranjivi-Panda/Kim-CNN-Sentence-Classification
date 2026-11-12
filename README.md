# Kim CNN for Sentence Classification

This project is a PyTorch implementation of the Convolutional Neural Network (CNN) proposed by **Yoon Kim (2014)** for sentence classification.

We reproduce the model on the **SST-2 (Stanford Sentiment Treebank)** dataset to classify movie reviews into two sentiment categories: Positive and Negative.

## 1. Project Overview

The model uses Convolutional Neural Networks to identify meaningful patterns in sentences and predict their sentiment.

The main idea is to:
- Convert words into numerical vectors using word embeddings.
- Apply CNN filters to detect patterns of different lengths.
- Use max-over-time pooling to extract the strongest features.
- Pass the extracted features through a fully connected layer for classification.

## 2. Model Architecture

The model follows these steps:

1. **Word Embedding:** Represents each word as a 300-dimensional vector using pre-trained Word2Vec embeddings.
2. **Convolution:** Uses filters of sizes 3, 4, and 5 to detect patterns across consecutive words.
3. **ReLU Activation:** Introduces non-linearity into the model.
4. **Max Pooling:** Selects the strongest activation from each filter.
5. **Dropout:** Helps prevent overfitting.
6. **Fully Connected Layer:** Produces the final sentiment classification scores.

## 3. Experiments

We evaluated four different embedding configurations:

- **CNN-rand:** Randomly initialized word embeddings.
- **CNN-static:** Pre-trained Word2Vec embeddings kept fixed during training.
- **CNN-non-static:** Pre-trained embeddings fine-tuned during training.
- **CNN-multichannel:** Uses two embedding channels, one fixed and one fine-tuned.

## 4. Results

Our best accuracy was achieved using fine-tuned embeddings.

| Model | Our Accuracy | Kim (2014) |
|---|---:|---:|
| CNN-rand | 75.1% | 82.7% |
| CNN-static | 83.7% | 86.8% |
| CNN-non-static | 85.0% | 87.2% |
| CNN-multichannel | 84.6% | 88.1% |

The results show that pre-trained word embeddings improve classification performance, while fine-tuning provides an additional improvement.

## 5. Ablation Study

We also experimented with different convolution filter configurations to understand their impact on performance.

The experiments compare:
- Width 3 with 100 filters.
- Widths 3, 4, and 5 with 100 filters each.
- Width 3 with 300 filters.

The results suggest that increasing the number of filters also improves performance, and multiple filter widths are not necessarily the only reason for the improvement.

## 6. Technologies Used

- Python
- PyTorch
- NumPy
- Google Word2Vec
- SST-2 Dataset
- Jupyter Notebook

## 7. Reference

Yoon Kim, **"Convolutional Neural Networks for Sentence Classification,"** Proceedings of EMNLP, 2014.

[Read the original paper](https://aclanthology.org/D14-1181/)

---

**Note:** This project is an educational reproduction of the original research. Some implementation and preprocessing differences may contribute to the difference in reported accuracy.
