# Implicit Feedback Recommendation System

## 📌 Project Overview
This project implements a high-performance recommendation engine for a live-streaming platform. The goal is to recommend 6 specific **Price points (USD)** and their corresponding **Value for Money (VFM)** scores to users based on their purchase history.

The system uses **Implicit Feedback** (purchase counts) rather than explicit ratings. It employs **Alternating Least Squares (ALS)** for matrix factorization and compares it against a **Popularity Baseline** model.

## 📂 Project Architecture

```
├── recommendation_system.ipynb   # Main Jupyter Notebook (Pipeline, Training, Evaluation)
├── verify_final.py               # Standalone Python script for pipeline verification
├── requirements.txt              # Python dependencies list
├── Pipfile                       # Pipenv definition
└── .env                          # Environment variables configuration
```

## ⚙️ Prerequisites & Installation

### 1. Environment Setup
You can set up the environment using either **Pipenv** (recommended) or **pip**.

#### Option A: Using Pipenv
```bash
# Install dependencies
pipenv install

# Activate shell
pipenv shell

# Start Jupyter
pipenv run jupyter notebook
```

#### Option B: Using Pip
```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Environment Variables (.env)
The project uses a `.env` file to manage system-specific configurations, particularly for library optimizations and hardware acceleration paths.

**Current `.env` configuration:**
```bash
# CUDA Paths (for GPU acceleration support)
CUDA_HOME=/usr/local/.....
LD_LIBRARY_PATH=/usr/local/cuda-**.**/lib64:.......
PATH=/usr/local/cuda-**.**/bin:${PATH}
```
*Note: The notebook also programmatically sets `OPENBLAS_NUM_THREADS=1` to optimize CPU performance for the `implicit` library.*

## 🧠 Methodology

### 1. Data Processing
*   **Library:** `Polars` is used for high-performance data manipulation.
*   **Item Definition:** Items are defined as unique tuples of `(USD, Coins)` to handle different VFM configurations for the same price.
*   **Temporal Split:** A time-based split is strictly enforced to prevent data leakage.
    *   **Training:** First ~77% of data.
    *   **Testing:** Last 7 days (Recency-based).

### 2. Models
*   **Popularity Baseline:**
    *   Recommends the globally most purchased items.
    *   Used as a fallback for **Cold Start** (new users).
*   **ALS (Alternating Least Squares):**
    *   Matrix Factorization algorithm designed for implicit feedback.
    *   Factors the sparse User-Item interaction matrix into latent user and item vectors.
    *   **Sparsity:** The matrix is ~99% sparse, which is ideal for collaborative filtering.

## 📊 Evaluation Metrics
The models are evaluated on a held-out test set using the following metrics:

*   **Precision@6:** What % of recommended items were actually bought?
*   **Recall@6:** What % of the user's actual purchases were recommended?
*   **AUC (Area Under Curve):** Probability that a random purchased item is ranked higher than a random non-purchased item.
*   **MPR (Mean Percentile Rank):** Average rank of purchased items (lower is better).

**Key Results:**
The ALS model achieved an **AUC > 0.9**, significantly outperforming the baseline in Recall and ranking quality, proving the value of personalization.

## 🚀 Usage
1.  Open `recommendation_system.ipynb`.
2.  Run all cells to execute the pipeline:
    *   Load and clean data.
    *   Train models.
    *   View evaluation plots.
    *   Generate recommendations for specific users.
