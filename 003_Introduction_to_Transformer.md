# What are Transformers?

The **Transformer architecture** is the core technology behind most modern **Large Language Models (LLMs)**. It is a deep neural network architecture introduced in the 2017 paper, ***"Attention Is All You Need,"*** which pioneered significant breakthroughs in the field.

Originally, the Transformer was developed for **machine translation** tasks, such as converting English text to German or French, rather than for text completion or generation, which LLMs are predominantly known for today.

---

## Simplified Transformer Architecture

The Transformer mechanism for translation can be broken down into an eight-step process involving two main components: the **encoder** (left side) and the **decoder** (right side).


### 1. The Encoder Block (Left Side)

The encoder is responsible for processing the input text and converting it into a meaningful numerical representation.

1.  **Input Text:** The source text to be translated (e.g., an English sentence).
2.  **Pre-processing (Tokenization):** The input sentence is broken down into smaller units called **tokens** (which can be words or sub-words), and each token is assigned a unique numerical **ID**.
3.  **Encoder Processing:** The token IDs are passed to the encoder, which converts them into **Vector Embeddings**.
4.  **Vector Embedding:** This process projects the tokens into a high-dimensional vector space. The vectors are structured so that the **semantic meaning and relationship** between words are captured by their proximity to one another in this space. For example, related words like "King," "Man," and "Woman" would be clustered closer together than "Banana" and "King." 
### 2. The Decoder Block (Right Side)

The decoder is responsible for generating the target sequence, word by word.

5.  **Decoder Input:** The decoder receives the vector embeddings from the encoder, along with the **partial output text** (the words already translated).
6.  **Decoding:** The decoder's task is to predict the **next word** in the target language (e.g., German) based on the input embeddings and the words it has already produced.
7.  **Output Generation:** The decoder iteratively generates the translated sentence one word at a time.
8.  **Final Output:** The process concludes with the complete translated text.

---

## Key Mechanism: Self-Attention

The **Self-Attention Mechanism** is the fundamental innovation and the reason the seminal paper is titled "Attention Is All You Need."

* **Weighing Importance:** Self-attention allows the model to **weigh the importance** of every word in the input sequence relative to every other word when processing a specific word.
* **Long-Range Dependencies:** This mechanism enables the model to capture **long-range dependencies** in the text. This means the model can look far back into the document to maintain context and understand which past words are most relevant for predicting the current or next word, similar to how a human maintains context when reading a story.

---

## Later Transformer Variations

The original encoder-decoder Transformer architecture led to two major variations used in modern models:

| Model | Full Name | Components | Operation | Key Application |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | **B**idirectional **E**ncoder **R**epresentations from **T**ransformers | Encoder only | Predicts **masked (hidden) words** in a sentence. It views the text **bidirectionally** (left-to-right and right-to-left). | Sentiment analysis, understanding word relationships and context. |
| **GPT** | **G**enerative **P**re-trained **T**ransformers | Decoder only | Generates the **next word** in a sequence, operating strictly **left-to-right**. | Text generation, chat, and creative writing. |


---

## Transformers vs. Large Language Models (LLMs)

The terms **Transformers** and **LLMs** are often incorrectly used interchangeably.

* **Not all Transformers are LLMs:** The Transformer architecture can be applied to other domains besides language, such as **computer vision**. **Vision Transformers (ViT)** are used for image classification, image segmentation, and object detection.
* **Not all LLMs are Transformers:** Prior to the Transformer, other architectures were used for text completion and sequence modeling, including **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory networks (LSTMs)**. These models also function as language models by incorporating memory mechanisms.

---

## Summary

* The Transformer architecture, originating from the "Attention Is All You Need" paper, is the foundational technology for modern LLMs.
* It operates using an **encoder** to create vector embeddings that capture semantic meaning and a **decoder** to generate output text one word at a time.
* The crucial **self-attention** mechanism allows the model to weigh the importance of all input words, enabling it to understand long-range context.
* Modern models like **BERT** (encoder-only, bidirectional) and **GPT** (decoder-only, left-to-right) are variations of this architecture.
* It is important to note that the terms Transformer and LLM are not synonymous, as Transformers have applications in vision, and LLMs can be built on pre-Transformer architectures like RNNs and LSTMs.