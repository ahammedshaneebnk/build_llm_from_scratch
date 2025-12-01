# What are Token Embeddings?

## Introduction to Token Embeddings
Token embeddings, also referred to as vector embeddings, represent a critical third step in the workflow of building Large Language Models (LLMs), following tokenization and the generation of token IDs. While terms like "word embeddings" are common, "token embeddings" is more accurate as tokens can represent words, sub-words, or characters. The primary function of this step is to convert token IDs into a format that serves as the input for training models like GPT.


## The Need for Token Embeddings
Computers cannot process raw text and require numerical representation. However, simple methods of converting text to numbers have significant limitations:

* **Random Number Assignment:** Assigning arbitrary numbers to words (e.g., Cat = 34, Book = 2.9) fails to capture the relationships between words.
* **One-Hot Encoding:** Creating massive binary vectors for each word is inefficient and, like random assignment, fails to encode semantic meaning. For instance, it cannot indicate that "dog" and "puppy" are related while "dog" and "banana" are not.

### Capturing Semantic Meaning
The goal of embeddings is to exploit the inherent information within text, similar to how Convolutional Neural Networks (CNNs) exploit spatial relationships in images.
* **Semantic Relationships:** Words carry meaning and relate to one another (e.g., "cat" and "kitten" are semantically close).
* **Vector Representation:** Words are encoded as vectors in a multi-dimensional space. The dimensions theoretically represent various features (e.g., "is eatable," "is a pet," "has a tail").
* **Result:** In a well-constructed vector space, words with similar meanings have similar vector representations, while unrelated words are far apart.


## Conceptual Demonstration: Word2Vec
Using pre-trained models like `word2vec` (specifically the Google News 300 dataset), one can demonstrate that vectors effectively capture semantic meaning through arithmetic operations and similarity checks.

* **Vector Arithmetic:** Vectors can be manipulated algebraically to reveal relationships. A classic example is:
    * `Vector(King) + Vector(Woman) - Vector(Man) ≈ Vector(Queen)`
    * This operation subtracts the "masculine" component and adds the "feminine" component, preserving the "royal" attribute.
* **Similarity Scores:** The distance between vectors quantifies the relationship between words.
    * High similarity: Man/Woman, Uncle/Aunt.
    * Low similarity: Paper/Water.

## Creating Token Embeddings for LLMs
Creating embeddings involves constructing a specific data structure known as the Embedding Matrix or Embedding Weights.

### Key Dimensions
Two primary parameters define the structure of the embedding layer:
1.  **Vocabulary Size:** The total number of unique tokens the model recognizes (e.g., GPT-2 has a vocabulary size of 50,257).
2.  **Vector Dimension:** The size of the vector representing each token (e.g., GPT-2 uses 768 dimensions).

### The Embedding Matrix
* **Structure:** The matrix size is defined as `Vocabulary Size x Vector Dimension`.
    * For GPT-2: 50,257 rows by 768 columns.
    * Each row corresponds to a specific token ID and contains the vector weights for that token.
* **Initialization:** The weights in this matrix are initially set to random values.
* **Optimization:** These weights are optimized via backpropagation during the LLM training process. The model learns the correct vector values by training on massive datasets, adjusting the weights to minimize prediction errors and capture semantic relationships.


## Implementation Details
### The Lookup Table
In practice, the embedding layer functions as a lookup table rather than a complex mathematical transformation during inference.
* **Operation:** When the model receives a Token ID (e.g., ID 3), it looks up the corresponding row (e.g., row 4, accounting for zero-indexing) in the embedding matrix and retrieves that specific vector.
* **Efficiency:** This approach is computationally efficient, allowing the model to retrieve embeddings for a sequence of tokens instantaneously.

### Comparison with Linear Layers
Mathematically, an embedding layer performs the same operation as a neural network linear layer applied to a one-hot encoded input.
* **Linear Layer Approach:** `Input (One-Hot) x Weights Transpose`. This involves multiplying a vector of mostly zeros with a weight matrix.
* **Embedding Layer Approach:** Direct indexing/lookup.
* **Why use Embeddings?** While mathematically identical, the linear layer approach involves unnecessary multiplication by zero. The embedding layer's lookup method is significantly more computationally efficient, especially given the massive vocabulary sizes of modern LLMs.

## Summary
* **Definition:** Token embeddings convert token IDs into multi-dimensional vectors that act as inputs for Large Language Models.
* **Purpose:** Unlike random numbers or one-hot encoding, embeddings capture the semantic meaning and relationships between words (e.g., grouping "cat" and "kitten").
* **Structure:** An embedding layer is a matrix with dimensions defined by the Vocabulary Size and the Vector Dimension (e.g., 50,257 x 768 for GPT-2).
* **Training:** Embedding weights are initialized randomly and optimized during the model's training process using backpropagation.
* **Mechanism:** The embedding layer functions as a highly efficient lookup table, retrieving the vector corresponding to a specific token ID.
* **Efficiency:** This lookup method is preferred over standard linear layers to avoid computationally expensive operations involving one-hot encoded vectors.