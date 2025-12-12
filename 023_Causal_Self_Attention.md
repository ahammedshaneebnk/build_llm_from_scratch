# Causal Self Attention Mechanism or Masked Self Attention

## Recap of Self-Attention
Before introducing causal masking, the standard self-attention mechanism operates as follows:
1.  **Input Processing:** Words are converted into tokens, then token IDs, and finally into vector embeddings.
2.  **Context Vectors:** The goal is to transform input embeddings (which hold static semantic meaning) into context vectors (which encode relationships with other words in the sequence).
3.  **Mechanism:**
    * Input embeddings are multiplied by trainable weight matrices ($W_q, W_k, W_v$) to generate Query ($Q$), Key ($K$), and Value ($V$) matrices.
    * **Attention Scores:** Calculated by multiplying Queries with the transpose of Keys ($Q \cdot K^T$).
    * **Attention Weights:** Scores are scaled by the square root of the key dimension and passed through a Softmax function to normalize them (rows sum to 1).
    * **Output:** Attention weights are multiplied by the Values ($V$) to produce the final context vectors.

## Causal Attention (Masked Attention)

### Concept and Purpose
Causal attention modifies standard self-attention to restrict the model's view. In a sequence, a specific token should only attend to itself and previous tokens, not future ones.
* **Contrast with Self-Attention:** Standard self-attention allows a token to see the entire sequence (past and future) simultaneously.
* **Restriction:** Causal attention ensures that when predicting the next word, the model effectively masks out any information coming after the current token position. This is crucial for autoregressive tasks like text generation.

### Implementation Strategies

#### The Naive Approach (Data Leakage Risk)
One method involves calculating standard attention weights (using Softmax) and then manually zeroing out the upper triangular elements (future tokens) and re-normalizing.
* **Problem:** This approach suffers from data leakage. Even if the values are zeroed out later, the initial Softmax calculation included the future tokens in its denominator. Therefore, the resulting probabilities are implicitly influenced by the future data.

#### The Efficient Approach (Masking Before Softmax)
The standard and correct implementation applies the mask *before* the Softmax step.
1.  **Upper Triangular Mask:** Create a mask where elements above the diagonal are set to negative infinity ($-\infty$).
2.  **Application:** Add this mask to the raw attention scores.
3.  **Softmax:** When Softmax is applied, $e^{-\infty}$ becomes 0.
4.  **Result:** Future tokens are effectively zeroed out without affecting the probability distribution of the past tokens. The rows naturally sum to 1 without needing re-normalization or risking data leakage.

## Dropout in Attention Mechanisms
Dropout is a regularization technique integrated into the attention mechanism to prevent overfitting and improve generalization.
* **Mechanism:** Randomly switches off (sets to zero) a percentage of neurons or attention weights during training.
* **Lazy Neuron Problem:** Prevents neurons from becoming co-dependent or "lazy" by forcing different pathways to learn robust features.
* **Scaling:** To compensate for the zeroed-out elements, the remaining active elements are scaled up by a factor of ${1/(1-p)}$ (e.g., if dropout is 50%, remaining weights are multiplied by 2) to maintain consistent statistical magnitude.
* **Placement:** In Transformer architectures, dropout is typically applied to the attention weights just after the Softmax calculation and before multiplying with the Value matrix.

## Summary
* **Causal Attention Definition:** A mechanism that restricts a model's attention to only current and past tokens, masking out future tokens to enable valid next-token prediction.
* **Masking Strategy:** The most effective masking technique involves replacing attention scores for future tokens with negative infinity before the Softmax layer to prevent data leakage and ensure correct normalization.
* **Dropout Integration:** Dropout is applied to attention weights to prevent overfitting by randomly deactivating a fraction of the weights and scaling the remainder.