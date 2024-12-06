# Fake News Detection Model Using an LLM

This project demonstrates how to build, train, and evaluate a fake news detection model using a Large Language Model (LLM) fine-tuned on the [LIAR](https://huggingface.co/datasets/liar) dataset. The model is based on a pre-trained transformer (e.g., BERT) and classifies news statements into several categories, including determining whether they are fake or real.

## Overview

Fake news detection is the task of identifying misinformation and disinformation in written media. With the abundance of online news and social media posts, automatic detection methods are becoming increasingly important. Large Language Models, such as BERT, RoBERTa, and DistilBERT, have shown strong performance on text classification tasks, making them suitable for this domain.

In this project, we:
- Load and preprocess the LIAR dataset.
- Tokenize and prepare the data using a pre-trained BERT tokenizer.
- Fine-tune a pre-trained BERT model for multi-class classification.
- Evaluate the model on a held-out evaluation set.
- Use the trained model to make predictions on new, unseen text.

## Key Features

- **Transformer-based model**: Leverages a pre-trained BERT model for feature extraction.
- **Hugging Face integration**: Uses the `transformers` and `datasets` libraries to simplify data loading, model training, and evaluation.
- **Customizable Training**: Easily adjust epochs, batch sizes, and learning rates via `TrainingArguments`.
- **Scalable Evaluation Metrics**: Utilizes `evaluate` to compute accuracy and can be extended to include other metrics (precision, recall, F1).

## Prerequisites

- Python 3.7 or later
- [Anaconda](https://www.anaconda.com/products/distribution) or [pip](https://pip.pypa.io/en/stable/) for package management
- [PyTorch](https://pytorch.org/) for model training
- Internet access if you plan to download the pre-trained model and dataset on the fly

## Installation

```bash
pip install transformers datasets evaluate torch
```

## Dataset

The [LIAR dataset](https://huggingface.co/datasets/liar) is used as an example in the code. It contains political statements labeled by fact-checkers, providing multiple classes (such as pants-fire, false, barely-true, half-true, mostly-true, and true).

**Important:** The LIAR dataset includes 6 distinct labels, not just binary classification. The code provided has been updated to handle all 6 classes. For a strict fake/real binary classification, you would need to adjust the dataset and labels accordingly.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fake-news-llm.git
   cd fake-news-llm
   ```

2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Fake\ News\ Detection\ Model\ Using\ an\ LLM.ipynb
   ```
   - Make sure you have Jupyter installed (via `conda` or `pip`).
   - Open the notebook and run the cells in order to:
     - Install required libraries.
     - Load and preprocess the dataset.
     - Train the model.
     - Evaluate the model.
     - Make predictions on new text.

3. **Adjusting Parameters**:
   - Modify `num_train_epochs`, `per_device_train_batch_size`, or other hyperparameters in `TrainingArguments` to experiment with training time and model performance.
   - Replace `model_name = "bert-base-uncased"` with another pre-trained model name from Hugging Face (e.g., `roberta-base`) to test different backbones.

4. **Inference**:
   - Use the `predict_news()` function with your own text strings to see how the model classifies them.
   ```python
   text = "Breaking: Scientists have discovered water on Mars!"
   print("Prediction:", predict_news(text))
   ```

## Troubleshooting

- **Symlink Warnings on Windows**: If you encounter warnings related to symlinks, consider enabling Developer Mode in Windows or running Python as an administrator. Alternatively, set the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable to `1`.

- **trust_remote_code Error**: Some datasets require `trust_remote_code=True` in `load_dataset`. If needed, update the dataset loading line accordingly:
  ```python
  dataset = load_dataset("liar", split="train[:80%]", trust_remote_code=True)
  ```

## Contributing

Contributions, bug reports, and suggestions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Note:** The notebook and code provided are for educational purposes. For production-level solutions, additional steps such as hyperparameter tuning, model optimization, proper validation, and domain adaptation should be considered.
