# Task 02 – Perceptron on Halfmoon Data

## Objective

The goal of this task is to **apply the Perceptron algorithm** to a non-linearly separable dataset generated in a "halfmoon" shape. This experiment will show the **limitations of linear classifiers** like the Perceptron when handling data that cannot be separated by a straight line.

Students must:

- Complete the Perceptron implementation.
- Use the `generate_halfmoon_data` function to generate two interleaving halfmoons.
- Train the Perceptron on this data.
- Plot the resulting decision boundary.

## What to Implement

Open the file `task-02-halfmoon-perceptron.py` and complete the missing parts:

- `_init_weights`
- `activation`
- `predict`
- `fit`

## Goal

The Perceptron **should fail** to properly classify the two halfmoons — and that’s **expected**. Use this opportunity to reflect on the **limitations of linear models**, and why non-linear methods (e.g., kernel methods, MLPs) are needed for more complex datasets.

## Running the Code

After completing the code, simply run:

```bash
python task-02-halfmoon-perceptron.py
