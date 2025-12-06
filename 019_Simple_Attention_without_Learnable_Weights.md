# Simplified Attention Mechanism

This content outlines the mathematical foundations and Python implementation of a simplified self-attention mechanism without trainable weights.

## Introduction to the Attention Mechanism
The core objective of the attention mechanism is to transform standard **Input Embedding Vectors** into **Context Vectors**. 
* **Input Embeddings:** These are vector representations of words (tokens) in a high-dimensional space (e.g., 3D for demonstration, though LLMs often use 700+ dimensions). They capture the semantic meaning of individual words (e.g., "cat" and "kitten" are close in space).
* **Limitation of Embeddings:** While embeddings capture meaning, they do not inherently contain information about how a word relates to other specific words in a given sentence.
* **Context Vectors:** These can be viewed as **"enriched" embeddings**. **They contain not only the word's semantic meaning but also contextual information regarding its relationship with every other word in the sequence**. This is crucial for predicting the next word in a sequence.

## The Workflow: From Embeddings to Context

The process of converting input vectors ($X$) into context vectors ($Z$) involves three main steps: computing attention scores, normalizing them into weights, and calculating the final context vector.

### Step 1: Computing Attention Scores
The first step is to quantify how much "attention" a specific word (the **Query**) should pay to every other word in the input sequence.

* **The Query:** The token currently being analyzed is termed the "query."
* **Alignment via Dot Product:** To determine importance, the dot product is calculated between the query vector and every input vector.
    * The dot product measures the alignment between two vectors.
    * **High Dot Product:** Vectors are aligned (similar meaning), indicating high attention.
    * **Low Dot Product:** Vectors are orthogonal or unaligned, indicating low attention.
* **Result:** A set of raw attention scores representing the similarity between the query and all inputs.

### Step 2: Normalization (Attention Weights)
Raw attention scores must be normalized to create interpretable **Attention Weights** that sum to 1.

* **Why Normalize?**
    * **Interpretability:** Allows for statements like "Word A pays 20% attention to Word B."
    * **Training Stability:** Crucial for backpropagation and gradient descent in neural networks.
* **The Softmax Function:** While simple summation is possible, the **Softmax** function is preferred.
    * **Handling Extremes:** Softmax pushes extremely high values closer to 1 and low values closer to 0, effectively highlighting the most relevant inputs while suppressing irrelevant ones.
    * **Formula:** $e^{x_i} / \sum e^{x_j}$
* **Numerical Stability (PyTorch Implementation):**
    * In practice (e.g., PyTorch), Softmax is implemented by subtracting the maximum value from the inputs before exponentiation: $e^{x_i - max(x)} / \sum e^{x_j - max(x)}$.
    * This prevents **overflow** (numbers becoming too large) and **underflow** errors during computation without changing the mathematical outcome.

### Step 3: Computing the Context Vector
The final context vector is a weighted sum of all input vectors.

* **Scaling:** Each input embedding vector is multiplied (scaled) by its corresponding attention weight.
    * Vectors with high attention weights retain most of their magnitude.
    * Vectors with low attention weights are scaled down significantly.
* **Summation:** All scaled vectors are added together to form the single context vector for the query.
* **Matrix Multiplication:** Computationally, this can be efficiently performed using matrix multiplication:
    $$\text{Context Vectors} = \text{Attention Weights Matrix} \times \text{Input Matrix}$$
    * The **Attention Weights Matrix** (Size: $N \times N$) contains weights for all pairs.
    * The **Input Matrix** (Size: $N \times D$) contains the $D$-dimensional embeddings for $N$ tokens.
    * The result is an $N \times D$ matrix of context vectors.

## Limitations of Non-Trainable Weights
The simplified model presented relies solely on the dot product of fixed embeddings.

* **Semantic vs. Contextual:** This approach only captures *semantic similarity*. It assumes that words with similar meanings should attend to each other.
* **The Problem:** In many sentences, words that are not semantically similar are still contextually important.
    * *Example:* "The cat sat on the mat because it is warm."
    * If "warm" is the query, it should pay attention to "mat" (because the mat is warm). However, "mat" and "warm" are not semantically similar, so a simple dot product might yield a low score.
* **Solution:** **Trainable Weights** are introduced in advanced models. They allow the model to learn relationships beyond simple meaning, capturing long-range dependencies and complex context (e.g., learning that "warm" often follows "mat" in specific contexts). This leads to the concepts of Key, Query, and Value matrices.

## Summary
* **Objective:** The goal of self-attention is to convert input embeddings into enriched context vectors that capture relationships between words.
* **Mechanism:**
    * **Attention Scores:** Calculated using the dot product between the query vector and input vectors to measure alignment.
    * **Attention Weights:** Scores are normalized using Softmax to ensure they sum to 1 and handle extreme values effectively.
    * **Context Vector:** The weighted sum of input vectors, where weights are determined by the attention mechanism.
* **Implementation:** Efficiently calculated via matrix multiplication: `Attention Weights @ Inputs`.
* **Numerical Stability:** Softmax implementations typically subtract the maximum value to prevent computational overflow.
* **Limitation:** Without trainable weights, attention is based purely on semantic similarity, missing complex contextual relationships where words are related but not similar in meaning.