# Trump-Tweets-Text-Classification

This project involves optimizing a series of models to predict whether a tweet was written by Donald Trump or not. The optimization process utilized various machine learning and deep learning models to achieve the best possible performance.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

The goal of this project is to classify tweets as either written by Donald Trump or not. We used a variety of machine learning and deep learning models, including logistic regression, SVM, XGBoost, and DistilBERT, to achieve this task.

## Dataset

The dataset contains tweets labeled as either written by Donald Trump or someone else. It includes the following columns:

- `tweet id`
- `user handle`
- `tweet text`
- `time stamp`
- `device`

The dataset is preprocessed to remove noise and irrelevant information, focusing on the `tweet text`.

## Models

We experimented with several models, including:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **XGBoost**
4. **Feedforward Neural Network (FFNN)**
5. **DistilBERT**

Each model was fine-tuned to achieve the best performance.

## Feature Engineering

We used various feature engineering techniques to improve model performance, including:

- **TF-IDF Vectorization**: Transforms text data into numerical features.
- **Additional Features**: Extracted from the tweet text and metadata, such as the number of capitalized words, punctuation, and timestamp features.

## Hyperparameter Optimization

Hyperparameters were optimized using Optuna with cross-validation. This allowed us to test a wide range of hyperparameter values efficiently. The following hyperparameters were optimized:

- `penalties`
- `hidden layer sizes and quantities`
- `number of epochs`
- `maximum depth`
- `number of estimators`

Optuna helped identify the best combinations, significantly improving model performance.

## Results

The best-performing model was XGBoost, achieving an F1 score of 0.89 with the optimized hyperparameters. DistilBERT also showed excellent performance, albeit with higher computational costs. The optimized hyperparameters for each model are provided in the results table.

The use of TF-IDF combined with engineered features yielded the best results for most models. Word2Vec was not as effective due to the small dataset size.

## Conclusion

XGBoost was the most effective model for this task, balancing performance and interpretability. DistilBERT showed promise but required more computational resources. Feature engineering, particularly TF-IDF and additional features, played a crucial role in improving model performance.

## Usage (in the API notebook)

To use the models and run predictions, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/Trump-Tweets-Text-Classification.git
    cd Trump-Tweets-Text-Classification
    ```

2. **Install the dependencies**:

   Take the installs from the top of the ipynb file into a requirements.
    ```sh
    pip install -r requirements.txt
    ```

4. **Prepare your data**: Ensure your dataset is in the correct format as described above.

5. **Run the training pipeline**:
    ```python
    from train import training_pipeline

    # Choose the model (1: Logistic Regression, 2: SVM, 3: XGBoost, 4: FFNN, 5: DistilBERT)
    model = training_pipeline(3, train_fn='path/to/your/train_data.tsv', test=True)
    ```

6. **Make predictions**:
    ```python
    from predict import predict_texts

    predictions, probabilities = predict_texts(model, 'path/to/your/test_data.tsv')
    print(predictions)
    ```
