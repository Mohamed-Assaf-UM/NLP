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
