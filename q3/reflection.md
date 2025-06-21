# Project: Fruit Classifier with a Single Neuron

This project involves building and training a single-neuron logistic regression model from scratch using NumPy to classify fruits based on their characteristics.

## Project Requirements

1.  **Dataset (`fruit.csv`):**
    *   Create a dataset named `fruits.csv`.
    *   It should contain at least 12 rows.
    *   Columns: `length_cm`, `weight_g`, `yellow_score` (from 0 to 1), and `label`.
    *   The `label` should be `0` for an apple and `1` for a banana.

2.  **Model Implementation:**
    *   Implement a single-neuron logistic regression model.
    *   The implementation must use only the NumPy library.

3.  **Training:**
    *   Train the model using batch gradient descent.
    *   The training should run for at least 500 epochs or until the loss is less than 0.05.

4.  **Evaluation:**
    *   Plot the loss over epochs.
    *   Plot the accuracy over epochs.

5.  **Reflection (≤ 300 words):**
    *   Compare the model's initial random predictions with its final results.
    *   Discuss the impact of the learning rate (LR) on the model's convergence.
    *   Relate the training process to the "DJ-knob / child-learning" analogy.
