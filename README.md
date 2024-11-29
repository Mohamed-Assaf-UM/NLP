# NLP
### 1. **What is NLP?**
   - NLP (Natural Language Processing) is a branch of machine learning and deep learning that helps machines understand and process human language, like text or speech.  
   - Examples: Detecting spam emails, voice commands (e.g., Alexa, Google Home).

---

### 2. **Why NLP is Important?**
   - Machines can't directly understand human languages like English, Chinese, or others. 
   - NLP techniques convert text into numeric formats (vectors) that machines can interpret.

---

### 3. **Preparing for NLP**
   - Start with **Python programming**—a must-have skill for NLP.  
   - Libraries like **NLTK** (Natural Language Toolkit) and **Spacy** will simplify tasks.

---

### 4. **NLP Roadmap**
The roadmap is divided into layers of increasing complexity:

#### **Step 1: Text Preprocessing (Part 1)**  
   - Clean and prepare text data using techniques like:  
     - **Tokenization**: Breaking text into smaller units (words, sentences).  
       *Example*:  
       - Input: *"I love cricket."*  
       - Output: ["I", "love", "cricket", "."]  
     - **Lemmatization/Stemming**: Reducing words to their root form.  
       *Example*:  
       - "running" → "run."  
     - **Stop words removal**: Removing common words (e.g., "is," "the," "and").  

#### **Step 2: Text Preprocessing (Part 2)**  
   - Convert cleaned text into **vectors** using:  
     - **Bag of Words (BoW)**: Counts word occurrences.  
     - **TF-IDF**: Weighs words based on importance in a document.  
     - **N-grams (unigrams, bigrams)**: Captures word sequences (e.g., "New York").  

#### **Step 3: Text Preprocessing (Part 3)**  
   - Use more advanced methods like:  
     - **Word2Vec**: Converts words into meaningful numeric vectors.  
     - **Average Word2Vec**: Averages word vectors for a sentence.  

#### **Step 4: Deep Learning for NLP**  
   - Techniques include:  
     - **RNN (Recurrent Neural Network)**, **LSTM (Long Short-Term Memory)**: Handle sequential data like text.  
     - **GRU (Gated Recurrent Unit)**: Simplifies RNN.  
     - Applications: Sentiment analysis, spam detection, text summarization.  

#### **Step 5: Advanced NLP**  
   - Use **transformers (BERT)** for high accuracy in NLP tasks.  
   - These are more powerful but require greater computational resources.

---

### 5. **Tools & Libraries**
   - **For Machine Learning NLP**: NLTK, Spacy.  
   - **For Deep Learning NLP**: TensorFlow (by Google), PyTorch (by Facebook).

---

### Simplified Summary
The main goal of NLP is to:
1. **Clean text** (remove noise, tokenize, etc.).
2. **Convert text into numbers** (vectors) that machines can process.
3. Use **machine learning or deep learning models** to solve tasks like spam detection, text summarization, or sentiment analysis.

---
Here’s a simplified explanation of the use cases of NLP described in the transcript, presented in a structured and concise manner:

---

### **Common NLP Use Cases in Daily Life**

1. **Spell Checking and Autocorrection (Gmail)**
   - Automatically corrects spelling errors while composing an email.
   - Suggests text completions or automated replies based on the content.

2. **Automated Replies (LinkedIn)**
   - Provides quick reply suggestions for messages or comments, saving time.

3. **Language Translation (Google Translate)**
   - Converts text from one language to another seamlessly.
   - Example: Translating "How are you?" to Arabic or Hindi in real-time.
   - Available as a feature in platforms like LinkedIn (e.g., "See Translation" for posts).

4. **Search and Recognition (Google Search)**
   - **Text to Image/Video**: Detects relevant images or videos based on textual input.
   - Identifies contextually related media and organizes search results.

5. **Advanced NLP Platforms (Hugging Face)**
   - Provides pre-trained models for tasks like:
     - **Question Answering**
     - **Text Summarization**
     - **Text Classification**
     - **Translation**
   - Companies like Google AI, Microsoft, and Grammarly leverage these models.

6. **Voice Assistants (Alexa, Google Assistant)**
   - Performs tasks through voice commands, such as controlling appliances or checking appointments.
   - Example: Asking, "Do I have any doctor appointments tomorrow?" retrieves calendar data.

---

### **Key Takeaways**
- NLP is integrated into daily tools to enhance productivity, simplify communication, and improve convenience.
- The course will teach **how these systems work**, covering tasks like text preprocessing, model training, and real-world applications.
---
### **Natural Language Processing (NLP) – Basic Terminologies and Tokenization**

#### **Key Terminologies in NLP**

1. **Corpus**:
   - A collection of texts or paragraphs used in NLP.
   - Example: *"My name is Krish. I have an interest in teaching ML, NLP, and DL."* This entire paragraph is considered a **corpus**.

2. **Documents**:
   - Individual sentences or smaller pieces derived from a corpus.
   - Example: In the paragraph above:
     - Document 1: *"My name is Krish."*
     - Document 2: *"I have an interest in teaching ML, NLP, and DL."*

3. **Vocabulary**:
   - A set of unique words from the corpus.
   - Example: For the corpus *"I like to drink apple juice. My friend likes mango juice."*:
     - Vocabulary: {"I", "like", "to", "drink", "apple", "juice", "My", "friend", "likes", "mango"}.
     - Note: Unique words only (repeated words are counted once).

4. **Words**:
   - All individual words in the corpus, including repetitions.
   - Example: In the above paragraph, the total words = 11, as "juice" is repeated.

---

#### **What is Tokenization?**

Tokenization is the process of breaking down a corpus into smaller units called tokens. Tokens can be:
   - **Sentences**: Breaking a paragraph into sentences.
   - **Words**: Breaking sentences into individual words.

---

#### **Examples of Tokenization**

1. **Paragraph → Sentences (Sentence Tokenization)**:
   - Input Corpus:  
     *"My name is Krish. I am also a YouTuber."*
   - Tokens (Sentences):  
     - *"My name is Krish."*  
     - *"I am also a YouTuber."*

2. **Sentence → Words (Word Tokenization)**:
   - Input Sentence:  
     *"My name is Krish."*
   - Tokens (Words):  
     - ["My", "name", "is", "Krish"]

---

#### **Why is Tokenization Important?**

1. **Preprocessing for NLP**: 
   - Tokenization is a fundamental step for preparing text data for further processing (e.g., cleaning, vectorization).

2. **Feature Extraction**:
   - Tokens (words) are often converted into numerical vectors for machine learning models.

3. **Facilitates Analysis**:
   - Allows NLP systems to analyze sentences or words individually for tasks like sentiment analysis, text summarization, etc.

---

#### **Understanding Vocabulary with Example**

**Example Paragraph**:
   - *"I like to drink apple juice. My friend likes mango juice."*

1. **Tokenize into Sentences**:  
   - Sentence 1: *"I like to drink apple juice."*  
   - Sentence 2: *"My friend likes mango juice."*

2. **Count Words (Total)**:  
   - All words = ["I", "like", "to", "drink", "apple", "juice", "My", "friend", "likes", "mango", "juice"]  
   - Total = 11 words.

3. **Find Unique Words (Vocabulary)**:  
   - Unique words = {"I", "like", "to", "drink", "apple", "juice", "My", "friend", "likes", "mango"}  
   - Vocabulary Size = 10.

---

#### **Key Takeaways**

- **Corpus**: Collection of texts or paragraphs.
- **Documents**: Sentences or subsets of the corpus.
- **Vocabulary**: Unique words in the corpus.
- **Tokenization**: Splitting corpus into sentences or words.
   - **Sentence Tokenization**: Paragraph → Sentences.
   - **Word Tokenization**: Sentence → Words.
---
Let's break down each block of this code into simple and clear explanations, covering the details of what is happening in each step and the differences between the tokenization techniques.

---

### **Step 1: Preparing the Corpus**

```python
corpus = """Hello Welcome,to Krish Naik's NLP Tutorials.
Please do watch the entire course! to become expert in NLP.
"""
print(corpus)
```

- **What is happening?**
  - The `corpus` variable contains a block of text (a paragraph) that we will use for tokenization.
  - This is simply printed to check the content.
  
- **Output:**
  ```
  Hello Welcome,to Krish Naik's NLP Tutorials.
  Please do watch the entire course! to become expert in NLP.
  ```

---

### **Step 2: Sentence Tokenization**

#### **Using `sent_tokenize`**

```python
from nltk.tokenize import sent_tokenize
documents = sent_tokenize(corpus)
type(documents)
for sentence in documents:
    print(sentence)
```

- **What is happening?**
  - `sent_tokenize`: Splits the corpus into sentences. 
  - Here, it breaks the text wherever it detects sentence boundaries (like a period `.` or exclamation mark `!`).
  - `documents` will be a list where each sentence is a separate element.

- **Output:**
  ```
  Hello Welcome,to Krish Naik's NLP Tutorials.
  Please do watch the entire course!
  to become expert in NLP.
  ```

- **Key Point:**
  - This method focuses on splitting the **paragraph into sentences**.
  - The output is a list where each sentence is a token.

---

### **Step 3: Word Tokenization**

#### **Using `word_tokenize`**

```python
from nltk.tokenize import word_tokenize
word_tokenize(corpus)
```

- **What is happening?**
  - `word_tokenize`: Breaks the corpus into individual words (or tokens).
  - It also includes punctuation as separate tokens.

- **Output:**
  ```
  ['Hello', 'Welcome', ',', 'to', 'Krish', 'Naik', "'s", 'NLP', 'Tutorials', '.', 'Please', 'do', 'watch', 'the', 'entire', 'course', '!', 'to', 'become', 'expert', 'in', 'NLP', '.']
  ```

- **Key Point:**
  - The tokens include words **and** symbols like `,`, `'`, and `.`.
  - This method splits **both sentences and words**, making every word and punctuation a separate token.

---

#### **Tokenizing Each Sentence**

```python
for sentence in documents:
    print(word_tokenize(sentence))
```

- **What is happening?**
  - Here, instead of tokenizing the entire corpus at once, we tokenize each sentence from `documents` (generated by `sent_tokenize`).
  
- **Output:**
  ```
  ['Hello', 'Welcome', ',', 'to', 'Krish', 'Naik', "'s", 'NLP', 'Tutorials', '.']
  ['Please', 'do', 'watch', 'the', 'entire', 'course', '!']
  ['to', 'become', 'expert', 'in', 'NLP', '.']
  ```

- **Key Point:**
  - This helps tokenize sentence by sentence, which can be useful for advanced NLP tasks.

---

### **Step 4: WordPunct Tokenizer**

#### **Using `wordpunct_tokenize`**

```python
from nltk.tokenize import wordpunct_tokenize
wordpunct_tokenize(corpus)
```

- **What is happening?**
  - `wordpunct_tokenize`: Splits the text into words and punctuation **separately**.
  - Unlike `word_tokenize`, it splits even apostrophes (`'`) and periods (`.`) as separate tokens.

- **Output:**
  ```
  ['Hello', 'Welcome', ',', 'to', 'Krish', 'Naik', "'", 's', 'NLP', 'Tutorials', '.', 'Please', 'do', 'watch', 'the', 'entire', 'course', '!', 'to', 'become', 'expert', 'in', 'NLP', '.']
  ```

- **Key Point:**
  - The difference is that `'` in `Krish's` is treated as a separate token (`Krish` and `s` are split).
  - Use this tokenizer when you want **precise separation of words and symbols.**

---

### **Step 5: Treebank Word Tokenizer**

#### **Using `TreebankWordTokenizer`**

```python
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(corpus)
```

- **What is happening?**
  - `TreebankWordTokenizer`: Splits text based on specific rules inspired by the Penn Treebank dataset.
  - It follows more structured rules for splitting contractions like `"don't"` into `"do"` and `"n't"`, and treating some symbols differently.

- **Output:**
  ```
  ['Hello', 'Welcome', ',', 'to', 'Krish', 'Naik', "'s", 'NLP', 'Tutorials.', 'Please', 'do', 'watch', 'the', 'entire', 'course', '!', 'to', 'become', 'expert', 'in', 'NLP', '.']
  ```

- **Key Point:**
  - This tokenizer keeps some punctuation attached to words (`Tutorials.` remains a single token).
  - It is often used in **syntactic parsing** tasks because of its rules-based nature.

---

### **Comparison of Tokenizers**

| **Tokenizer**          | **Key Feature**                                                                 | **Use Case**                                      |
|-------------------------|---------------------------------------------------------------------------------|--------------------------------------------------|
| `sent_tokenize`         | Splits the text into sentences.                                                 | When working with sentence-level processing.     |
| `word_tokenize`         | Splits into words and includes punctuation as tokens.                           | General word-level processing.                   |
| `wordpunct_tokenize`    | Separates words and punctuation strictly.                                       | When precise punctuation separation is needed.   |
| `TreebankWordTokenizer` | Follows specific linguistic rules for splitting contractions and symbols.       | Tasks requiring structured splitting like syntax.|

---

### **Final Note**

- **Sentence Tokenization** is used to break text into sentences, while **Word Tokenization** is used to break sentences/paragraphs into words or smaller tokens.
- Different tokenizers serve different needs based on the task. For example:
  - **Punctuation-sensitive analysis**: Use `wordpunct_tokenize`.
  - **General tokenization**: Use `word_tokenize`.
  - **Structured NLP tasks**: Use `TreebankWordTokenizer`.

---
