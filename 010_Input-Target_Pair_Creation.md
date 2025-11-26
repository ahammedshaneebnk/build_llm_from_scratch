# Creating Input-Target Data Pairs for LLMs

## Introduction
This step occurs after tokenization and before generating vector embeddings. While input-output definitions are clear in classification (image vs. label) or regression (area vs. price) tasks, LLMs require a specific technique to generate these pairs from raw text for self-supervised learning.

## Conceptual Framework

### Input-Target Pairs
* LLMs learn to predict one word at a time based on the preceding context.
* The text is divided into input blocks and target blocks.
* **Mechanism:**
    * In a sequence, the input is a specific set of tokens.
    * The target is the immediate next token.
    * All tokens past the target are masked during the training iteration.

### Auto-regressive Learning
* The model follows an auto-regressive pattern where the output of one iteration becomes the input for the next.
* **Example:**
    * *Iteration 1:* Input = "LLMs", Target = "learn"
    * *Iteration 2:* Input = "LLMs learn", Target = "to"
    * *Iteration 3:* Input = "LLMs learn to", Target = "predict"
* This approach is a form of self-supervised (or unsupervised) learning because the labels are inherent in the sentence structure itself; no manual labeling is required.

### Context Length
* **Definition:** The number of tokens given as input to the model to make a prediction.
* If the context size is 4, the model looks at the preceding 4 words to predict the 5th word.
* In one input-output pair defined by a context size of $N$, there are effectively $N$ prediction tasks occurring (predicting the 2nd token from the 1st, the 3rd from the 1st and 2nd, etc.).

## Implementation Strategy

### Simple Sliding Window
A basic intuitive method involves creating two arrays, $X$ (input) and $Y$ (target). $Y$ is essentially the $X$ array shifted by one position.
* **Input (X):** Sequence of tokens $[t_1, t_2, t_3, t_4]$
* **Target (Y):** Sequence of tokens $[t_2, t_3, t_4, t_5]$
* This structure reinforces that if the input is $[t_1]$, output is $t_2$; if input is $[t_1, t_2]$, output is $t_3$, and so on.

### PyTorch Data Loader
For efficient training and parallel processing, raw arrays are converted into tensors using PyTorch's `Dataset` and `DataLoader` classes.
* **Dataset Class:** Defines how individual rows (input chunks) are fetched. It creates tensors where each row represents one input context.
* **DataLoader Class:** Handles batch processing, shuffling, and parallel loading using multiple CPU threads.

## Key Definitions and Parameters

### Context Size (Max Length)
* Determines the window of text the model pays attention to at one time.
* Common context sizes in large models (like GPT-3) can be 256, 1024, or larger.

### Stride
* Dictates how many positions the window slides to create the next batch.
* **Stride = 1:** High overlap between batches. Input 1 is "A B C D", Input 2 is "B C D E". This can lead to overfitting due to repetitive data exposure.
* **Stride = Context Length:** No overlap. Input 1 is "A B C D", Input 2 is "E F G H". This is often preferred to utilize the dataset fully without redundancy.

### Batch Size
* The number of data samples processed simultaneously before the model updates its parameters.
* **Small Batch Size:** Faster updates but potentially noisy gradients.
* **Large Batch Size:** More stable updates but requires more memory and takes longer per update.

### Number of Workers
* A parameter in the `DataLoader` that specifies how many CPU subprocesses (threads) are used for data loading. This enables parallel processing to speed up training.

## Code Implementation Steps

1.  **Tokenization:**
    * The raw text (e.g., "The Verdict" short story) is encoded into token IDs using a Byte Pair Encoding (BPE) tokenizer (e.g., `tiktoken`).

2.  **Custom Dataset Class (`GPTDatasetV1`):**
    * Inherits from `torch.utils.data.Dataset`.
    * Takes arguments: text, tokenizer, max_length, and stride.
    * Implements the `__getitem__` method to return a specific row of input and target tensors based on an index.
    * Input tensor contains a chunk of tokens of length `max_length`.
    * Target tensor contains the same chunk shifted by one token.

3.  **Data Loader Function (`create_dataloader`):**
    * Instantiates the `GPTDatasetV1`.
    * Passes the dataset to `torch.utils.data.DataLoader`.
    * Configures batch size, shuffle options, and `drop_last` (to prevent size mismatch errors during training).
    * Returns an iterator that yields batches of input and target tensors ready for the model.

## Summary
* **Data Pre-processing:** Creating input-target pairs is a critical step connecting tokenization to model training.
* **Next-Word Prediction:** The core task is predicting the next token, where targets are simply inputs shifted by one position.
* **Sliding Window:** Data is processed using a sliding window approach, controlled by `context_size` and `stride`.
* **PyTorch Integration:** Using `Dataset` and `DataLoader` classes allows for efficient, batched, and parallel data processing essential for training large models.
* **Hyperparameters:** Key configuration options include batch size (trade-off between noise and speed) and stride (trade-off between data coverage and overfitting risk).