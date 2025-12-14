# Multi-Head Attention

## Concept of Multi-Head Attention
Multi-head attention divides the attention mechanism into multiple "heads." Each head operates independently, allowing the model to attend to different parts of the input sequence simultaneously.
* **Extension of Causal Attention:** The mechanism is essentially an aggregation of multiple causal attention layers stacked together.
* **Independent Operations:** Unlike a single attention head that uses one set of Query, Key, and Value matrices, multi-head attention utilizes multiple sets of these trainable matrices.
* **Purpose:** Stacking multiple heads increases computational complexity but significantly enhances the model's ability to recognize complex patterns. Modern models like GPT-3 and GPT-4 utilize a large number of attention heads (e.g., 96 for GPT-3).

## Mechanism and Workflow
The process involves running the single-head attention mechanism multiple times in parallel (conceptually) or sequentially (in this basic implementation) and combining the results.

### 1. Trainable Weight Matrices
* **Single-Head:** Uses one set of trainable weight matrices for Queries ($W_Q$), Keys ($W_K$), and Values ($W_V$).
* **Multi-Head:** If there are $h$ heads, there are $h$ sets of trainable weight matrices. For example, with 2 heads, there are two separate sets of $W_Q$, $W_K$, and $W_V$.

### 2. Computing Context Vectors
* The input embeddings are multiplied by the respective weight matrices for each head to generate specific Queries, Keys, and Values.
* Attention scores and weights are computed independently for each head.
* Each head produces its own Context Vector Matrix.

### 3. Concatenation
* The context vectors from all heads are concatenated to form the final output matrix.
* Concatenation occurs along the column dimension (feature dimension).

## Dimension Analysis
Understanding matrix dimensions is crucial for implementation.
* **Input:** Batch size $\times$ Number of tokens $\times$ Input embedding dimension ($d_{in}$).
* **Single Head Output:** If the output dimension is defined as $d_{out}$, one head produces a matrix of size: Number of tokens $\times$ $d_{out}$.
* **Multi-Head Output:** With $h$ heads, the final output matrix has the dimensions: Number of tokens $\times$ ($d_{out} * h$).
    * *Example:*
        * Batch size: 2
        * Tokens: 6
        * Input Dimension ($d_{in}$): 3
        * Output Dimension per head ($d_{out}$): 2
        * Number of Heads: 2
        * **Final Context Vector Shape:** $2 \times 6 \times 4$

## Summary
* **Foundation:** Multi-head attention extends causal attention by running multiple independent attention mechanisms in parallel.
* **Independent Weights:** Each attention head possesses its own unique set of trainable Query, Key, and Value matrices.
* **Output Aggregation:** The outputs (context vectors) of all heads are concatenated along the feature dimension to produce the final context vector matrix.
* **Dimensionality:** The final column dimension of the output is the product of the single-head output dimension ($d_{out}$) and the number of heads ($h$).