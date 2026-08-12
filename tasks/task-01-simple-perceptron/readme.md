# Perceptron Learning Algorithm - Implementation Exercise

## Objective

The goal of this task is to implement the Perceptron algorithm from scratch and explore its ability to learn a linear decision boundary for a binary classification problem. Students will complete the missing parts of a Python class representing a simple Perceptron model, following the classic online learning update rule.

## Description

The dataset used for training and testing is synthetically generated using `make_blobs`, creating two overlapping clusters. Labels are transformed to match the expected output range of the Perceptron: -1 and +1.

The exercise includes:

- Weight initialization using local random number generators
- Step activation function
- Linear prediction using bias
- Online weight updates using the Perceptron learning rule
- Visualization of the learned decision boundary

## What Students Must Implement

Students are required to complete the following methods within the `Perceptron` class:

- `_init_weights`: Initialize the weights using a locally seeded random generator.
- `activation`: Implement the step activation function.
- `predict`: Perform a linear combination and apply the activation function.
- `fit`: Apply the Perceptron learning algorithm over multiple epochs.

Each method includes markers (`### START CODE HERE ###` and `### END CODE HERE ###`) and TODO comments to guide students through the implementation.

## How to Run

After completing the required methods, execute the `main()` function to:

- Generate the training and testing data
- Train the Perceptron model
- Evaluate its accuracy on a test set
- Visualize the decision boundary

```bash
python task-01-simple-perceptron.py
