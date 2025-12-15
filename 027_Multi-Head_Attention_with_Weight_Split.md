# Multi-Head Attention with Weight Splits

## Introduction
The implementation of multi-head attention in large language models can be optimized to improve computational efficiency. A standard approach involves creating separate weight matrices for queries, keys, and values for each attention head. However, this method scales poorly; for instance, a model with 96 attention heads would require 96 separate matrix multiplications for each component. "Multi-head attention with weight splits" addresses this inefficiency by performing single large matrix multiplications and then splitting the results, significantly reducing the computational overhead.

## Core Concept: Weight Splits
Instead of maintaining separate classes or matrices for each head, the mechanism utilizes a single class capable of handling multiple heads simultaneously. The process begins with larger weight matrices that already include the dimensions for all heads. The input is multiplied by these larger matrices once, and the resulting tensors are then reshaped (split) to isolate the individual heads. This method effectively integrates the multi-head functionality directly with causal attention operations.

## Step-by-Step Implementation Details

### 1. Input and Output Configuration
The process starts by defining the input tensor dimensions: Batch Size ($B$), Number of Tokens ($T$), and Input Dimension ($D_{in}$).
* **Example Configuration:** A batch size of 1, 3 tokens (e.g., "the cat sleeps"), and an input dimension of 6.
* **Output Dimensions:** The output dimension ($D_{out}$) is typically set equal to the input dimension ($D_{in}$).
* **Head Configuration:** The number of attention heads is determined (e.g., 2 heads), which dictates the "Head Dimension" ($D_{head}$). $D_{head}$ is calculated as $D_{out}$ divided by the number of heads.

### 2. Initialization and Linear Projections
Trainable weight matrices ($W_Q$, $W_K$, $W_V$) are initialized with dimensions $D_{in} \times D_{out}$.
* **Computation:** The input tensor ($B \times T \times D_{in}$) is multiplied by these weight matrices.
* **Result:** This produces three separate matrices for Queries, Keys, and Values, each with the shape $B \times T \times D_{out}$. At this stage, the data for all heads is combined within the $D_{out}$ dimension.

### 3. Reshaping and Transposing (Creating Heads)
To isolate the attention heads, the tensors undergo dimension manipulation:
* **Unrolling:** The last dimension ($D_{out}$) is reshaped into two dimensions: Number of Heads and Head Dimension. The tensor shape transforms from $(B, T, D_{out})$ to $(B, T, \text{Heads}, D_{head})$.
* **Transposing:** To allow parallel computation for each head, the "Tokens" and "Heads" dimensions are swapped. The resulting shape is $(B, \text{Heads}, T, D_{head})$. This grouping ensures that operations can proceed independently for each head.

### 4. Computing Attention Scores and Weights
Attention scores are calculated using the reshaped Queries and Keys.
* **Dot Product:** Queries are multiplied by the transpose of the Keys (transposed along the last two dimensions).
* **Shape:** The resulting score matrix has dimensions $(B, \text{Heads}, T, T)$, representing the relationship between every token and every other token within each specific head.
* **Masking (Causal Attention):** An upper triangular mask is applied (setting values to negative infinity) to ensure tokens only attend to previous positions in the sequence.
* **Scaling and Softmax:** The scores are divided by the square root of the Head Dimension to stabilize gradients, followed by a Softmax operation. This produces the final Attention Weights, where each row sums to 1.

### 5. Context Vector Computation
The Context Vectors are derived by multiplying the Attention Weights by the Values matrix.
* **Operation:** Attention Weights $(B, \text{Heads}, T, T) \times \text{Values} (B, \text{Heads}, T, D_{head})$.
* **Result:** A tensor of shape $(B, \text{Heads}, T, D_{head})$, containing the context information for each token relative to each specific head.

### 6. Final Output Reshaping
The final step involves merging the heads back together to restore the original output structure.
* **Transpose Back:** The dimensions are swapped back to place the "Tokens" dimension before the "Heads" dimension: $(B, T, \text{Heads}, D_{head})$.
* **Flattening:** The last two dimensions (Heads and Head Dimension) are merged (flattened) back into $D_{out}$.
* **Final Shape:** The resulting Context Vector Matrix has the shape $(B, T, D_{out})$, matching the original expected output dimensions. This matrix may optionally pass through a final linear projection layer.

## Mathematical and Code Equivalence
The logical flow corresponds directly to tensor operations in code (e.g., PyTorch). Understanding the manipulation of four-dimensional tensors is critical:
* **3D vs. 4D:** Initial linear layers produce 3D tensors (Batch, Tokens, Feature Dim). The "view" and "transpose" operations convert these into 4D tensors to explicitly handle the Head dimension.
* **Contiguity:** When reshaping, ensuring tensors are stored continuously in memory is often required before flattening dimensions.

## Summary
* **Efficiency:** Weight splits reduce the number of required matrix multiplications from one per head to one per component (Query, Key, Value), regardless of the number of heads.
* **Dimension Logic:** The output dimension ($D_{out}$) is the product of the number of heads and the individual head dimension ($D_{head}$).
* **Dimensions during the Process:**
1. (B x T x Din) * (Din x Dout) -> (B x T x Dout) -> (B x T x Head x Dhead) -> (B x Head x T x Dhead)
2. (B x Head x T x Dhead) * (B x Head x Dhead x T) -> (B x Head x T x T);
3. (B x Head x T x T) * (B x Head x T x Dhead) -> (B x Head x T x Dhead) -> (B x T x Head x Dhead) -> (B x T x Dout)
* **Output Consistency:** The final context vector restores the original dimensionality ($B, T, D_{out}$), effectively concatenating the insights from all attention heads into a single enriched representation for each token.
* **Scalability:** This architecture is the standard foundation for modern Large Language Models (like GPT-3), scaling efficiently to dozens or hundreds of attention heads.