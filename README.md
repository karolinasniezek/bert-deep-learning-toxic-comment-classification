# Toxic Comment Classification with BERT

This project implements a multi-label text classification system for detecting toxic comments using a pretrained BERT model and TensorFlow.

The model predicts six toxicity categories:

toxic
severe_toxic
obscene
threat
insult
identity_hate

The project covers the complete machine learning workflow, including model training, evaluation, threshold optimization, and confusion matrix analysis.

## Technologies

* Python 3.10
* TensorFlow / Keras
* Hugging Face Transformers
* BERT (`bert-base-uncased`)
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

## Features

* Text tokenization with BERT tokenizer
* Multi-label classification (6 classes)
* Fine-tuning pretrained BERT model
* Prediction threshold optimization using F1-score
* Multilabel confusion matrix evaluation
* Model weight saving and loading

## Implemented Tasks

* BERT-based text classification
* Model training and evaluation
* Probability prediction
* Binary classification using configurable thresholds
* Per-class threshold optimization
* Multilabel confusion matrix generation

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd BertDeepLearningToxicCommentClasification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Execute the main script:

```bash
python src/main.py
```

## Output

The project generates:

* model predictions
* optimized classification thresholds
* multilabel confusion matrices
* saved model weights (`toxic_model.weights.h5`)
* confusion matrix report (`macierz_pomylek.txt`)
