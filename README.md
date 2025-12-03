# Deep Learning – Homeworks

Repository for Deep Learning homeworks (IST).

## Structure

- `Homework1/`
  - `homework1 (2).pdf` – Assignment text
  - `hw1-ffn.py` – PyTorch feedforward neural network experiments (Question 2)
  - `hw1-perceptron.py` – Linear models (Question 1)
  - `utils.py` – Data loading and utilities
  - `data/emnist-letters.npz` – EMNIST Letters dataset (ignored by git)

More homework folders (Homework2, Homework3, …) will be added later.

## Setup

From the `DeepLearning/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Homework 1

```bash
cd Homework1
python hw1-ffn.py 
```
