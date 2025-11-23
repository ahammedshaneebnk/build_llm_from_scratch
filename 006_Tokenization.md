# Building an LLM Tokenizer from Scratch

## Introduction to Large Language Model Development

Building a large language model typically involves three key stages:

1.  **Stage 1: Basic Mechanism & Data Preparation:** This initial stage focuses on data preparation, understanding the attention mechanism, and defining the LLM architecture.
2.  **Stage 2: Pre-training:** This involves the training loop, model evaluation, and building the foundational model.
3.  **Stage 3: Fine-tuning:** Training on smaller, specific datasets to build useful applications like classifiers or personal assistants.

**Tokenization** is the fundamental first step in the **Data Preparation** phase of Stage 1.

***

## The Process of Tokenization

Large Language Models (LLMs) are neural networks that require data to be pre-processed before training. Input text, even billions of documents, cannot be fed directly; it must be tokenized.

Tokenization involves three main conceptual steps:

1.  **Split Text:** Initially split the text into individual word and subword tokens.
2.  **Convert to Token IDs:** Convert these tokens into unique integer IDs.
3.  **Encode to Vectors (Embedding):** Convert the token IDs into vector representations (token embeddings), which are then fed as input data to the LLM.

The first two steps; splitting text and converting to IDs, constitute the core of building a tokenizer.


### Step 1: Tokenizing the Text

Tokenization is implemented in Python, typically utilizing the **regular expression (`re`)** library for efficiency and flexibility.

* **Splitting Logic:** The text is split based on:
    * White spaces.
    * Punctuation marks such as commas, full stops, colons, semicolons, question marks, exclamation marks, quotation marks, and double dashes (`--`).
* **Cleaning:** White space characters are typically removed from the resulting list of tokens. While this reduces memory, keeping spaces may be necessary for training on code or other structure-sensitive data (eg. Python code is sensitive to intendations).

### Step 2: Converting Tokens to Token IDs

For an LLM to process text, each token must be represented numerically.

* **Vocabulary Creation:** A **vocabulary** is constructed from all **unique** tokens found in the training data. This list is sorted alphabetically.
* **ID Assignment:** Each unique token in the sorted vocabulary is mapped to a unique integer, starting from zero. This integer is the **Token ID**.
* **Encoder and Decoder:**
    * The mapping from a **Token (string) to a Token ID (integer)** is the **Encoder** function.
    * The reverse mapping from a **Token ID (integer) to a Token (string)** is the **Decoder** function, which is required to convert the LLM's numerical output back into human-readable text.

## Implementing the Tokenizer Class

A simple `Tokenizer` class is structured in Python to manage the encoding and decoding process:

* **`__init__` Method:** Initializes the two core dictionaries: `string_to_int` (the encoder) and `int_to_string` (the decoder).
* **`encode` Method:** Takes the input text, applies the regular expression splitting and cleaning, and then converts the resulting list of tokens into a list of Token IDs using the vocabulary map.
* **`decode` Method:** Takes a list of Token IDs, converts them back to tokens, joins the tokens into a single string, and performs post-processing (e.g., removing spaces before punctuation) to restore grammatical sentence structure.

***

## Dealing with Unknown Words: Special Context Tokens

A major consideration for any tokenizer is how to handle **out-of-vocabulary (OOV)** words—words not seen in the training data and thus absent from the vocabulary.

* **Unknown Token (`<|unk|>`)**: To prevent the tokenizer from failing, a dedicated `<|unk|>` token is added to the vocabulary. Any unrecognized word in the input text is automatically replaced by this token's ID.
* **End-of-Text Token (`<|endoftext|>`)**: This token is added to the vocabulary to act as a crucial marker, signaling the end of one text source before another begins. This separation is essential for LLMs (like GPT) trained on multiple independent documents to prevent the mixing of context.
* **Other Special Tokens:** Other tokens used in LLM training include:
    * Beginning of Sequence (`<|bos|>`)
    * End of Sequence (`<|eos|>`)
    * Padding (`<|pad|>`)—used to extend shorter text samples in a batch to match the length of the longest text, optimizing parallel processing.

Note that the tokenizer used for GPT models utilizes only the **End-of-Text token** but handles unknown words using **Byte Pair Encoding (BPE)**, which automatically breaks words down into subword units instead of relying on an explicit unknown token.

***

## Summary

* Tokenization is the initial data preparation step for training Large Language Models (LLMs).
* The fundamental process converts text into tokens, which are mapped to unique integer **Token IDs**.
* A **Vocabulary** is a sorted list of all unique tokens from the training data, with each token assigned a sequential ID.
* The **Encoder** converts text to Token IDs, and the **Decoder** converts Token IDs back to text.
* The **regular expression (`re`)** library in Python is used to split text based on white spaces and punctuation.
* **Special Context Tokens** like `<|unk|>` (unknown) and `<|endoftext|>` (end of text) are critical for handling out-of-vocabulary words and separating multiple training documents, respectively.
* The tokenizer used for **GPT** models does not use any of the special context tokens except `<|endoftext|>` but handles unknown words using **BPE**.