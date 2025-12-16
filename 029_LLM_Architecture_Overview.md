# Birds Eye View of the LLM Architecture

## Introduction
This section provides a high-level overview of the Large Language Model (LLM) architecture, specifically focusing on the structure used in GPT-2. It contextualizes the architecture within the broader process of building an LLM from scratch and outlines the components that make up the Transformer block.

## LLM Building Stages
The process of building a large language model is divided into three distinct stages:
1.  **Foundations**: Laying the groundwork, which includes data preparation, sampling, tokenization, vector embeddings, positional embeddings, and the attention mechanism.
2.  **Pre-training**: Training the LLM on a large corpus of text to learn general language patterns.
3.  **Fine-tuning**: Adapting the pre-trained model for specific tasks or instructions.

## The LLM Architecture: A Bird's Eye View
The architecture transforms input text into predicted output text through a series of processing steps.

### Input Processing
* **Tokenization**: Input text (e.g., "Every effort moves you") is converted into token IDs using a tokenizer (e.g., Byte Pair Encoder).
* **Token Embeddings**: Each token ID is projected into a vector space. For GPT-2 Small, this is a 768-dimensional vector. These embeddings capture semantic meaning.
* **Positional Embeddings**: To retain the order of tokens, positional embedding vectors are added to the token embedding vectors. The size of the positional embedding matrix corresponds to the context length (e.g., 1024 positions).

### The Transformer Block
The Transformer block is the core of the architecture. It consists of several stacked layers and components:
* **Layer Normalization**: Applied before and after key sub-layers to stabilize training.
* **Masked Multi-Head Attention**: Converts input embeddings into context vectors. This mechanism allows the model to understand how tokens relate to one another within the sequence.
* **Dropout**: Randomly turns off neurons during training to prevent overfitting and improve generalization.
* **Shortcut Connections (Residual Connections)**: Allow gradients to flow more easily during backpropagation by bypassing certain layers.
* **Feed-Forward Neural Network**: A fully connected network that processes the context vectors.
* **GELU Activation**: The Gaussian Error Linear Unit activation function is used within the feed-forward network.

### Output and Prediction
* **Logits**: The output of the Transformer blocks is passed through a final linear layer to produce logits. The output matrix has dimensions corresponding to the batch size, sequence length, and vocabulary size (e.g., 50,257).
* **Next Token Prediction**: For every token in the input sequence, the model predicts the next token. The logits represent the unnormalized probabilities for each potential next token in the vocabulary. The token with the highest probability is selected as the prediction.

## GPT-2 Model Configuration
Here, we utilize the architecture of the **GPT-2 Small** model for implementation.

* **Parameters**: 124 Million
* **Layers (Transformer Blocks)**: 12
* **Embedding Dimension ($d_{model}$)**: 768
* **Attention Heads**: 12
* **Vocabulary Size**: 50,257 (based on Byte Pair Encoding)
* **Context Length**: 1024 tokens
* **Dropout Rate**: 0.1 (typical value)
* **Bias**: Query, Key, Value biases set to `False`

*Note: Larger versions of GPT-2 scale up the number of layers and embedding dimensions (e.g., GPT-2 Extra Large has 48 layers and 1600 embedding dimensions).*

## Implementation Concepts: The Dummy GPT Model
A placeholder "Dummy GPT" class serves as a skeleton for the code implementation.

* **Initialization**: The model initializes the token embedding matrix (`nn.Embedding`) and the positional embedding matrix. Weights are initially random and optimized during pre-training.
* **Forward Pass**:
    1.  Receives a batch of token IDs.
    2.  Retrieves token embeddings and adds positional embeddings.
    3.  Passes the data through the stack of Transformer blocks.
    4.  Applies final layer normalization.
    5.  Projects the output to the vocabulary size to generate logits.
* **Output Shape**: The final tensor shape is `[Batch Size, Sequence Length, Vocabulary Size]`. Each element in the vocabulary dimension corresponds to the probability of that token being the next word.

## Summary
* The LLM architecture connects previous concepts like tokenization and attention into a cohesive trainable system.
* Input text is processed into token embeddings combined with positional embeddings to retain sequence order.
* The Transformer block is a composite structure containing Layer Norm, Multi-Head Attention, Feed-Forward Networks (with GELU), Dropout, and Shortcut connections.
* The specific configuration used corresponds to GPT-2 Small: 124M parameters, 12 layers, 12 heads, and 768 embedding dimensions.
* The model functions by predicting the next token in a sequence, outputting logits that map to the entire vocabulary size (50,257).
* Code implementation involves creating a modular structure where specific blocks (like Layer Norm and Feed Forward) are defined as classes and stacked within the main GPT model.