# Pretraining LLMs vs Finetuning LLMs

## 1. Pre-training (The Foundational Model)

Pre-training is the first stage where an LLM is trained on a **large and diverse dataset**. This stage is responsible for the LLM's fundamental ability to interact effectively and answer general questions.

### Data Corpus
LLMs are trained on massive amounts of data, often in the hundreds of billions of words/tokens.

| Data Source | GPT-3 Data Size (Approximate Tokens) |
| :--- | :--- |
| **Common Crawl** | 410 Billion |
| **WebText2** | 20 Billion |
| **Books (2 sets)** | 67 Billion |
| **Wikipedia** | 3 Billion |

### Task and Capabilities
* **Core Task:** Initially, LLMs are trained for a simple task called **word completion** or **predicting the next word** (i.e., unsupervised learning).
* **Emergent Capabilities:** A surprising discovery was that training an LLM for this simple task resulted in it gaining a wide range of other capabilities, known as **emergent properties**. These include translation, summarization, sentiment detection, and question answering, without ever being explicitly trained for them.
* **Cost:** The computational cost for pre-training is enormous. For example, the total pre-training cost for **GPT-3 was approximately $4.6 million**.

---

## 2. Fine-tuning

Fine-tuning is the second stage, where the already pre-trained model (the **foundational model**) is further refined on a **much narrower data set** for a specific application or domain.

### Purpose and Need
* Fine-tuning is essential for companies and industries that need an LLM to answer questions **specific to their private data** or a particular domain, where the general pre-trained model might lack the required knowledge.
* **Examples:**
    * **SK Telecom:** Fine-tuned GPT-4 for customer service interactions specific to telecom conversations in Korean, leading to significant performance improvements.
    * **Harvey:** An AI legal tool for attorneys that was fine-tuned on **legal case history** and other specific legal knowledge that foundational models often lack.
    * **JP Morgan Chase:** Unveiled its own LLM suite, fine-tuned on their proprietary banking data for employee use.

### Data Type
* Pre-training is typically done on **unlabeled data** (raw text).
* Fine-tuning is mostly done on **labeled text data** for supervised learning.
* **Categories of Fine-tuning:**
    * **Instruction Fine-tuning:** Using instruction-answer pairs as a labeled data set (e.g., translation, customer support).
    * **Classification Tasks:** Using labeled data to classify inputs (e.g., classifying emails as spam or no-spam).

---

## 3. Summary: Stages of Building an LLM 

1.  **Data:** Collect a massive, diverse, and raw (**unlabeled**) text corpus from the internet, books, etc.
2.  **Pre-training (Foundational Model):** Train the model on the huge, **unlabeled** data set to create the **base/foundational model**, which has broad general capabilities.
3.  **Fine-tuning (Specific Application):** Train the foundational model further on a smaller, task-specific, and **labeled** data set to develop specific applications (e.g., a chatbot, a summarization assistant).