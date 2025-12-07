# Self-Attention Mechanism with Trainable Weights

## Introduction
The self attention mechanism, often referred to as **"scaled dot-product attention"**, improves upon simplified attention models by introducing optimization through training. The goal is to convert input embedding vectors for tokens into context vectors that encode both semantic meaning and relationship information with respect to other tokens in the sequence.

## Key Concepts: Query, Key, and Value
To calculate context vectors, the input embeddings are first transformed into three distinct vectors using trainable weight matrices. These matrices are optimized during the model training process.

* **Query (Q):** Represents the current token being focused on (analogous to a search query in a database).
* **Key (K):** Represents the label or identifier of input items (analogous to database keys used for matching).
* **Value (V):** Represents the actual content or information of the input items (analogous to the returned data).

## Step-by-Step Implementation

### Step 1: Input Transformation
The process begins with input embeddings (a matrix of size $N \times D_{in}$, where $N$ is the number of tokens).
* Three trainable weight matrices are initialized: $W_Q$, $W_K$, and $W_V$.
* The input matrix is multiplied by these weight matrices to produce the **Query**, **Key**, and **Value** matrices.
* Mathematically, if $X$ is the input:
    * $Q = X \cdot W_Q$
    * $K = X \cdot W_K$
    * $V = X \cdot W_V$
* These resulting matrices project the input into a new dimensional space defined by the output dimension ($D_{out}$) of the weight matrices.
* Hence the above matrix multiplication would be with sizes - N x Din * Din x Dout = N x Dout, effectively projecting from Din -> Dout space.
* In the original paper, Din = Dout but it is not necessary.

### Step 2: Computing Attention Scores
Attention scores determine how much focus a specific query should place on various keys.
* Scores are calculated by taking the dot product between the **Query matrix** and the transpose of the **Key matrix** ($Q \cdot K^T$), ie N x Dout * Dout x N = N x N
* The result is a square matrix (size $N \times N$) where each element represents the affinity or alignment between a specific query and a specific key.
* A high dot product indicates strong alignment, suggesting the model should pay more attention to that relationship.

### Step 3: Normalization and Attention Weights (Scaled Dot-Product)
Raw attention scores are not directly interpretable and can lead to training instability. This step normalizes the scores.

**Scaling Factor:**
* The attention scores are divided by the square root of the key dimension ($\sqrt{d_k}$).
* **Reason for Scaling:**
    * **Variance Stability:** The variance of the dot product increases with the dimensionality of the vectors. Without scaling, the variance grows, leading to unstable learning. Dividing by $\sqrt{d_k}$ keeps the variance close to 1.
    * **Softmax Peakiness:** Large score values result in a "peaky" softmax distribution (one value dominates, others equate to zero), which causes **vanishing gradients**. Scaling keeps values small, ensuring a more distributed and learnable gradient.

**Softmax Application:**
* The scaled scores are passed through a **Softmax** function.
* This ensures that the scores for each query sum to 1.0, converting them into probabilities or **Attention Weights**.
* These weights represent the percentage of attention the current token pays to every other token in the sequence.

### Step 4: Computing Context Vectors
The final step generates the context vectors.
* The **Attention Weights** matrix is multiplied by the **Value (V)** matrix, ie N x N * N x Dout = N x Dout.
* Intuitively, this computes a weighted sum of the Value vectors, where the weights are determined by the attention mechanism.
* The resulting matrix contains the context vectors for all input tokens. Each context vector is an enriched representation containing the token's original meaning plus its contextual relationship with the entire sequence.

## PyTorch Implementation Details
The mechanism can be implemented as a Python class (e.g., `SelfAttention`).

* **Initialization (`__init__`):** Defines the dimensions ($D_{in}$, $D_{out}$) and initializes the weight matrices ($W_Q$, $W_K$, $W_V$).
* **Forward Pass (`forward`):** Executes the four steps described above:
    1.  Compute `keys`, `queries`, `values` using matrix multiplication.
    2.  Compute `attention_scores` via dot product ($Q \cdot K^T$).
    3.  Compute `attention_weights` by scaling scores by $1/\sqrt{d_k}$ and applying Softmax.
    4.  Compute the final `context_vector` by multiplying weights with `values`.

**Optimization with `nn.Linear`:**
Instead of manually initializing random parameters, PyTorch's `nn.Linear` layer (with `bias=False`) is often used. `nn.Linear` includes optimized weight initialization schemes (like Kaiming or Xavier) that lead to more stable and effective model training compared to basic random sampling.

## Summary
* **Trainable Weights:** unlike simplified attention, self-attention uses optimized weight matrices ($W_Q, W_K, W_V$) to project inputs.
* **Projection:** Input embeddings are transformed into Query, Key, and Value matrices.
* **Attention Scores:** Calculated via the dot product of Queries and Keys ($Q \cdot K^T$) to measure token alignment.
* **Scaling is Critical:** Scores are scaled by $\sqrt{d_k}$ to maintain variance near 1 and prevent the softmax function from becoming too peaky or unstable.
* **Context Vectors:** Formed by multiplying normalized attention weights by the Value matrix, creating a weighted sum of information.
* **Implementation:** Can be efficiently coded in PyTorch using matrix operations and `nn.Linear` for better initialization.