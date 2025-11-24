# Byte Pair Encoding (BPE)

## Introduction to Tokenization Schemes

Tokenization is a fundamental concept in training large language models (LLMs). Modern algorithms, such as those used in GPT-2 and GPT-3, primarily rely on **Byte Pair Encoding (BPE)**, a sophisticated subword-based tokenization scheme.

Tokenization algorithms can be categorized into three main types:

1.  **Word-Based Tokenizer**
2.  **Character-Based Tokenizer**
3.  **Subword-Based Tokenizer**

***

## Word-Based Tokenization

In word-based tokenization, every word in a sentence is treated as a single, unique token (e.g., "the fox chased the dog" has five tokens).

### Problems

* **Out-of-Vocabulary (OOV) Words:** It is difficult to deal with words not present in the training vocabulary, often resulting in errors or reliance on an "unknown" token.
* **Large Vocabulary Size:** The English language contains approximately 170,000 to 200,000 words, requiring a vast vocabulary size, which is computationally expensive.
* **Lost Semantic Similarity:** Words with common roots, such as "boy" and "boys," are treated as separate, distinct tokens, failing to capture their inherent similarity.

***

## Character-Based Tokenization

In character-based tokenization, individual characters are considered tokens (e.g., "my hobby" is broken into `m`, `y`, `h`, `o`, `b`, `b`, `y`, etc.).

### Advantages

* **Small Vocabulary Size:** Since every language has a fixed number of characters (e.g., 256 in English), the vocabulary is very small.
* **OOV Problem Solved:** Any new sentence can always be broken down into characters already present in the vocabulary.

### Problems

* **Meaning Loss:** The meaning associated with complete words is lost when they are broken down into individual characters.
* **Longer Sequences:** The tokenized sequence becomes much longer than the initial raw text, increasing the sequence length for the LLM (e.g., the word "dinosaur" is split into eight tokens).

***

## Subword-Based Tokenization

Subword-based tokenization, exemplified by BPE, is a hybrid approach that aims to take the best features of both word and character methods.

### Rules of Subword Tokenization

1.  **Do not split** frequently used words; retain them as single tokens.
2.  **Split** rare words into smaller, meaningful subwords (dropping down to the character level if necessary).

### Advantages

* **Captures Root Words:** It helps the model learn that different words share the same root (e.g., `token`, `tokens`, `tokenizing` share the root `token`).
* **Identifies Suffixes/Prefixes:** It can identify common morphological structures and patterns, such as the suffix `ization` in `tokenization` and `modernization`.
* **Manages Vocabulary:** It provides a vocabulary size that is manageable (smaller than word-based) while retaining important semantic information.

***

## Byte Pair Encoding (BPE) Algorithm

BPE originated in 1994 as a **data compression algorithm**.

### Core Mechanism

The algorithm is iterative:

1.  Scan the data to identify the **most common pair of consecutive bytes/characters**.
2.  Replace that common pair with a single, new, unused byte or variable.
3.  Repeat the process until a stopping criterion is met (e.g., a target vocabulary size or no pair occurs more than once).

### BPE for Large Language Models (LLMs)

When applied to LLMs, BPE ensures the most common words are single tokens and rare words are broken down into two or more subword tokens.

**The End-of-Word Marker ($\mathbf{</w>}$):**
The marker `</w>` is appended to subword tokens to denote the **end of a full word**. This distinction is crucial for:
* **Boundary Definition:** It acts as a clear boundary marker, preventing ambiguity.
* **Reversibility:** It ensures the tokenization is fully reversible, allowing the model to differentiate between a subword that is a suffix (e.g., `ing</w>` in *walk**ing***) and the same subword that is part of a larger word (e.g., `ing` in *s**ing**le*).

**Example of Subword Merging:**
Using a corpus containing `old`, `older`, `finest`, and `lowest`, BPE iteratively identifies and merges frequent character pairs. The token `EST` is merged with the end-of-word marker `</w>` to form **`EST</w>`**, which encodes that the word ends after "est." This process creates subwords that capture the root representation, allowing the LLM to understand the similarity between related terms.

## Practical Implementation in LLMs

OpenAI's models (GPT-2, GPT-3) use a BPE implementation via **`tiktoken`**.

* The `gpt2` encoding uses a vocabulary size of **50,257 tokens**. This size is significantly smaller than a pure word-based vocabulary but retains high expressive power.
* BPE effectively handles unknown or OOV words by breaking them down into existing subword units or individual characters from the vocabulary. For instance, an unfamiliar word is represented as a sequence of known subword tokens, ensuring the tokenizer never produces an error for an unknown word.

***

## Summary

* BPE is a **subword tokenization algorithm** utilized by modern LLMs like GPT-2 and GPT-3.
* It is a hybrid approach, combining the benefits of **Word-Based** (tokenizing frequent words) and **Character-Based** (breaking down rare words) schemes.
* The core BPE mechanism is an iterative data compression technique that **merges the most frequent consecutive pairs** into a single new token.
* The **end-of-word marker $\mathbf{</w>}$** is vital for defining word boundaries and ensuring the tokenization process is unambiguous and reversible.
* BPE significantly **reduces the vocabulary size** compared to a word-based approach, making the model computationally efficient.
* It fundamentally solves the **Out-of-Vocabulary (OOV) problem** by ensuring any input, even random text, can be broken down into existing subwords or characters.
* The resulting subword tokens (e.g., `OLD`, `EST</w>`) **retain morphological and semantic root meanings**, which is crucial for the model's language understanding.