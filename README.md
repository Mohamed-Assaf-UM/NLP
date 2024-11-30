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
### **What is Stemming?**

Stemming is the process of reducing a word to its root or base form by removing suffixes or prefixes. It is widely used in **Natural Language Processing (NLP)** tasks like text classification, sentiment analysis, and information retrieval.

For example:
- Words like **"eating," "eats," "eaten"** are reduced to **"eat."**
- Words like **"going," "gone," "goes"** are reduced to **"go."**

The root word produced by stemming is often **not a valid word in the language** but serves as a representative for the word group.

---

### **Practical Example: Stemming in Action**

The given example uses three stemming algorithms: **PorterStemmer**, **RegexpStemmer**, and **SnowballStemmer**. Let’s break them down:

---

### **1. Porter Stemmer**

**Code:**
```python
from nltk.stem import PorterStemmer
stemming = PorterStemmer()

words = ["eating", "eats", "eaten", "writing", "writes", "programming", "programs", "history", "finally", "finalized"]

for word in words:
    print(word + " ----> " + stemming.stem(word))
```

**Output:**
```
eating ----> eat
eats ----> eat
eaten ----> eaten
writing ----> write
writes ----> write
programming ----> program
programs ----> program
history ----> histori
finally ----> final
finalized ----> final
```

#### **Explanation:**
- The **Porter Stemmer** applies a series of rules to remove common suffixes like **-ing**, **-es**, and **-ed**.
- It also converts plural words to their singular form.
- However, it doesn’t always produce readable or valid words (e.g., `"history"` → `"histori"`).
- It is widely used for its simplicity and effectiveness in text pre-processing.

---

### **2. Regexp Stemmer**

**Code:**
```python
from nltk.stem import RegexpStemmer
reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)

print(reg_stemmer.stem('eating'))    # eat
print(reg_stemmer.stem('ingeating')) # ingeat
```

**Output:**
```
eating ----> eat
ingeating ----> ingeat
```

#### **Explanation:**
- The **RegexpStemmer** uses regular expressions to identify and strip specific patterns.
- The regex pattern `'ing$|s$|e$|able$'` removes:
  - Words ending with **-ing** (e.g., `"eating"` → `"eat"`),
  - Words ending with **-s** or **-e**, or
  - Words ending with **-able** (e.g., `"comfortable"` → `"comfort"`).
- The `min=4` ensures that words shorter than 4 characters are not stemmed.
- This is a customizable and lightweight stemming method.

---

### **3. Snowball Stemmer**

**Code:**
```python
from nltk.stem import SnowballStemmer
snowballsstemmer = SnowballStemmer('english')

for word in words:
    print(word + " ----> " + snowballsstemmer.stem(word))
```

**Output:**
```
eating ----> eat
eats ----> eat
eaten ----> eaten
writing ----> write
writes ----> write
programming ----> program
programs ----> program
history ----> histori
finally ----> final
finalized ----> final
```

**Additional Examples:**
```python
print(snowballsstemmer.stem("fairly"))       # fair
print(snowballsstemmer.stem("sportingly"))  # sport
print(snowballsstemmer.stem("goes"))        # goe
```

#### **Explanation:**
- The **Snowball Stemmer** (or Porter2 Stemmer) is an improvement over the original Porter Stemmer.
- It resolves some of Porter’s issues with over-stemming or under-stemming.
  - Example: `"fairly"` → `"fair"` and `"sportingly"` → `"sport"`.
- It supports multiple languages.
- **Why better than Porter?**
  - It has better-defined rules and handles exceptions more robustly.

---

### **Key Differences Between Stemmers**

| **Stemmer**         | **Method**                                                                                       | **Strengths**                                           | **Weaknesses**                                             |
|----------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| **Porter Stemmer**   | Rule-based stemming that removes common suffixes.                                               | Simple, fast, widely used.                            | May produce non-readable words (e.g., `"history"` → `"histori"`). |
| **Regexp Stemmer**   | Uses regular expressions to define patterns for stemming.                                       | Fully customizable.                                   | Requires manual regex design, limited to specific patterns. |
| **Snowball Stemmer** | An advanced version of Porter with better handling of exceptions and language support.          | More accurate and flexible than Porter.              | Slightly slower than Porter.                              |

---

### **When to Use Stemming?**

1. **Text Classification Problems** (e.g., spam detection, sentiment analysis).
2. **Information Retrieval** (e.g., search engines).
3. **When preprocessing large corpora** to reduce word variations.
4. **When accuracy is not heavily dependent on word structure.**

---

### **What is Lemmatization?**

Lemmatization is the process of converting a word to its base or dictionary form, known as a **lemma**, while ensuring that the resulting word is a valid word with proper meaning in the language. Unlike stemming, lemmatization considers the **context** and **part of speech (POS)** to produce accurate base forms of words.

For example:
- Words like **"eating," "eats," "eaten"** are converted to **"eat"** (verb form).
- Words like **"finally"** and **"finalized"** are converted to **"finalize"** or remain unchanged based on their meaning and context.

---

### **Why Use Lemmatization Over Stemming?**
- **Lemmatization** produces words that are valid in the language (e.g., `"eaten"` → `"eat"` instead of `"eaten"`).
- It uses linguistic knowledge and considers the word's **part of speech (POS)**.
- It is more precise but computationally heavier than stemming.

---

### **WordNetLemmatizer**

#### **Code:**
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Example 1: Lemmatizing a single word with its part of speech (POS)
print(lemmatizer.lemmatize("going", pos="v"))  # Output: go
```

#### **Output:**
```
go
```

#### **Explanation:**
- The word **"going"** is a verb (`pos='v'`), so the lemmatizer returns **"go"** as its base form.

---

### **Working with a List of Words**

#### **Code:**
```python
words = ["eating", "eats", "eaten", "writing", "writes", "programming", 
         "programs", "history", "finally", "finalized"]

for word in words:
    print(word + " ----> " + lemmatizer.lemmatize(word, pos='v'))
```

#### **Output:**
```
eating ----> eat
eats ----> eat
eaten ----> eat
writing ----> write
writes ----> write
programming ----> program
programs ----> program
history ----> history
finally ----> finally
finalized ----> finalize
```

#### **Explanation:**
- The lemmatizer uses the part of speech (`pos='v'`) to convert verbs to their base form:
  - `"eating"` → `"eat"`
  - `"writing"` → `"write"`
  - `"programming"` → `"program"`
- For non-verbs, the lemmatizer keeps the word unchanged (e.g., `"history"`, `"finally"`).

---

### **Additional Examples**

#### **Code:**
```python
# Lemmatizing with different parts of speech
print(lemmatizer.lemmatize("goes", pos='v'))            # go
print(lemmatizer.lemmatize("fairly", pos='v'))          # fairly
print(lemmatizer.lemmatize("sportingly"))               # sportingly
```

#### **Output:**
```
go
fairly
sportingly
```

#### **Explanation:**
- `"goes"` is a verb, so it is reduced to **"go."**
- `"fairly"` and `"sportingly"` are adverbs (`pos='r'` by default in WordNet), and no transformation occurs unless a specific rule is applied.

---

### **Key Points about Lemmatization:**

1. **Part of Speech (POS):**
   - The accuracy of lemmatization depends heavily on providing the correct POS tag.
   - Common POS tags used:
     - `n` → Noun
     - `v` → Verb
     - `a` → Adjective
     - `r` → Adverb

2. **Comparison with Stemming:**
   - Lemmatization produces more **linguistically meaningful results.**
   - Example:
     - Stemming: `"history"` → `"histori"`
     - Lemmatization: `"history"` → `"history"`

3. **Applications in NLP:**
   - Lemmatization is useful in tasks where preserving linguistic meaning is important, such as:
     - **Q&A Systems**
     - **Chatbots**
     - **Text Summarization**

---

### **When to Use Lemmatization?**

1. **When linguistic accuracy is critical.**
   - Example: Analyzing texts for grammatical structure.
2. **For applications requiring valid dictionary words.**
   - Example: Query expansion in search engines.

---

### **Stopwords in NLP**  
Stopwords are common words in a language that often do not carry significant meaning in text analysis or machine learning tasks. Examples of stopwords in English include "is," "and," "the," "in," "on," etc. These words are frequently used to structure sentences but usually do not contribute to the semantic meaning of a document.  
   
Removing stopwords during text preprocessing helps reduce noise and computational overhead, making it easier for algorithms to focus on meaningful words. Stopwords lists are available for many languages and can be customized as per the task's requirements.  

---

### **Code Explanation with Stopwords**  
Here’s how the provided code integrates stopwords processing:

```python
from nltk.corpus import stopwords
import nltk

# Download the stopwords if not already available
nltk.download('stopwords')

# Fetch English stopwords
stop_words = stopwords.words('english')
print(stop_words)
```

1. **`stopwords.words('english')`**: Returns a predefined list of stopwords in English from the NLTK library. These are common words like "I," "me," "the," and "and."
2. **Why Remove Stopwords?**: Words like "I have three visions for India" include stopwords like "I," "have," and "for." These words don't add much meaning to understanding the text's context and are removed to simplify data.
3. **Using Arabic Stopwords**: Similarly, you can fetch Arabic stopwords with `stopwords.words('arabic')` if you're processing Arabic text.

---

### **Processing the Speech of Dr. APJ Abdul Kalam**  

To preprocess the paragraph by removing stopwords:  
```python
# Import required modules
from nltk.tokenize import word_tokenize

# Paragraph provided
paragraph = """I have three visions for India. In 3000 years of our history... (trimmed for brevity)"""

# Tokenize paragraph into words
words = word_tokenize(paragraph.lower())  # Convert text to lowercase and tokenize

# Remove stopwords
filtered_words = [word for word in words if word not in stop_words and word.isalpha()]  # Keep only meaningful words
print(filtered_words)
```

**Key Steps in the Code**:
1. **`word_tokenize(paragraph.lower())`**: Splits the paragraph into individual words while converting all to lowercase for uniformity.
2. **Filtering Stopwords**: A list comprehension removes words present in the `stop_words` list and keeps only alphabetic tokens (`word.isalpha()` ensures no numbers or special characters are included).
3. **Output**: A cleaner version of the speech with significant words retained, reducing the text's dimensionality for further NLP tasks like stemming, lemmatization, or vectorization.  

---
### **1. Stemming**
**What it does:**
- Stemming reduces a word to its root or base form. The resulting stem may not be a valid word (e.g., "playing" becomes "play", but "university" might become "univers").

**How it works internally:**
- **Rule-based approach**: Stemmers use predefined rules to strip prefixes or suffixes.
  - For example, in the **Porter Stemmer**, rules like:
    - If a word ends with "ing", remove "ing" (e.g., "playing" → "play").
    - If a word ends with "ed", remove "ed" (e.g., "hoped" → "hope").
- **Exceptions and heuristics**: Some algorithms include specific rules to handle common exceptions (e.g., "flies" → "fli" may be kept as is if no better match is found).

Popular stemmers in NLTK:
- **Porter Stemmer**: Rule-based and widely used for simplicity.
- **Lancaster Stemmer**: A more aggressive rule-based stemmer.

---

### **2. Lemmatization**
**What it does:**
- Lemmatization reduces a word to its **dictionary form** (lemma), ensuring the output is an actual word (e.g., "am", "are", "is" → "be").

**How it works internally:**
- **Dictionary Lookup**:
  - The lemmatizer maps words to their dictionary form using a lexicon (a predefined vocabulary database).
  - E.g., "better" → "good" (using WordNet in NLTK).
- **Part of Speech (POS) tagging**:
  - Lemmatizers require POS information to determine the correct lemma.
  - Example: "running" → "run" (verb), but "better" → "good" (adjective).
  - The POS tagger identifies whether the word is a verb, noun, etc., to choose the right lemma.

Popular lemmatizers in NLTK:
- **WordNet Lemmatizer**: Uses WordNet lexical database.
- **SpaCy Lemmatizer** (if you use SpaCy): Advanced with modern optimizations.

---

### **3. Stop Words**
**What it does:**
- Stop words are common words (e.g., "is", "and", "the") that are often removed from text since they don’t add significant meaning in many NLP tasks.

**How it works internally:**
- **Predefined List**:
  - Libraries like NLTK have a list of stop words (from various corpora like Penn Treebank or similar sources).
  - Example: Checking a word against this list during text preprocessing.
- **Customization**:
  - You can modify the stop words list based on your needs.

---

### **Why Everything Is in NLTK**
1. **Ease of Use**:
   - NLTK (Natural Language Toolkit) provides an integrated set of tools for NLP, so you don’t have to manually implement stemming, lemmatization, or stop-word filtering.

2. **Reusability**:
   - Algorithms like Porter Stemmer or WordNet Lemmatizer are complex to write from scratch, requiring rules, lexicons, and exceptions. NLTK provides these implementations ready-to-use.

3. **Modular Design**:
   - NLTK includes:
     - Predefined corpora like WordNet.
     - Algorithms for tokenization, stemming, lemmatization, etc.
   - It’s designed to support rapid development and experimentation in NLP.

4. **Open Source**:
   - NLTK is a community-supported library that compiles years of linguistic and computational research into a single library.

---

### **How Libraries Like NLTK Compare**
- **NLTK** is a pioneer in NLP and is more academic and educational.
- **SpaCy** is another library that is faster and better for production use (e.g., it uses a different pipeline for lemmatization and stop-word filtering).
- **CoreNLP** and **TextBlob** also provide similar functionalities but with unique focuses.

---

Let's break down how **Named Entity Recognition (NER)** works using the provided example and code snippet.

---

### **What is Named Entity Recognition (NER)?**
- **NER** is a technique in NLP to identify and classify named entities in text into predefined categories such as:
  - **Person**: Names of people (e.g., "Gustave Eiffel").
  - **Organization**: Company or institution names (e.g., "Eiffel's company").
  - **Location**: Geographical locations (e.g., "The Eiffel Tower").
  - **Date/Time**: Temporal expressions (e.g., "1887 to 1889").
  - **Other categories**: Such as monetary values, percentages, etc.

---

### **Steps in Your Code**

#### **1. Tokenizing the Sentence**
```python
words = nltk.word_tokenize(sentence)
```
- Tokenization breaks the sentence into words:
  ```
  ['The', 'Eiffel', 'Tower', 'was', 'built', 'from', '1887', 'to', '1889', 'by', 'Gustave', 'Eiffel', ',', 'whose', 'company', 'specialized', 'in', 'building', 'metal', 'frameworks', 'and', 'structures', '.']
  ```

---

#### **2. POS Tagging**
```python
tag_elements = nltk.pos_tag(words)
```
- Part-of-Speech tagging assigns grammatical labels to words:
  ```
  [('The', 'DT'), ('Eiffel', 'NNP'), ('Tower', 'NNP'), ('was', 'VBD'), ('built', 'VBN'),
  ('from', 'IN'), ('1887', 'CD'), ('to', 'TO'), ('1889', 'CD'), ('by', 'IN'),
  ('Gustave', 'NNP'), ('Eiffel', 'NNP'), (',', ','), ('whose', 'WP$'), 
  ('company', 'NN'), ('specialized', 'VBD'), ('in', 'IN'), ('building', 'VBG'),
  ('metal', 'JJ'), ('frameworks', 'NNS'), ('and', 'CC'), ('structures', 'NNS'), ('.', '.')]
  ```
  Here:
  - `NNP` = Proper noun (e.g., "Eiffel", "Tower").
  - `CD` = Cardinal number (e.g., "1887").
  - `VBD` = Past tense verb (e.g., "was").

---

#### **3. Named Entity Recognition (NER)**
```python
nltk.ne_chunk(tag_elements).draw()
```
- **What happens internally:**
  1. **POS-based Grouping**: The chunker groups adjacent tokens based on their POS tags. For example:
     - `"Eiffel Tower"` → A named entity of type `Location`.
     - `"Gustave Eiffel"` → A named entity of type `Person`.
  2. **Pre-trained Classifier**: The `maxent_ne_chunker` uses a pre-trained model based on the **Maximum Entropy (MaxEnt)** algorithm, which identifies and classifies named entities by analyzing their context.
  3. **Hierarchical Tree Structure**: The `ne_chunk()` function returns a tree representation where recognized named entities are labeled.

---

### **Visualization Using `.draw()**
When `.draw()` is called, a graphical representation of the parse tree appears. You’ll see:
- `PERSON` for names like "Gustave Eiffel".
- `LOCATION` for places like "Eiffel Tower".
- `DATE` for time periods like "1887 to 1889".

---

### **Output from NER**
Based on the sentence:
```plaintext
The Eiffel Tower was built from 1887 to 1889 by Gustave Eiffel, whose company specialized in building metal frameworks and structures.
```

The named entities recognized would be:
1. **LOCATION**: `Eiffel Tower`.
2. **DATE**: `1887 to 1889`.
3. **PERSON**: `Gustave Eiffel`.

These entities will be grouped in the tree, making it easy to extract structured information.

---

### **Why NLTK for NER?**
- NLTK offers pre-trained models and tools, like the `maxent_ne_chunker`, for educational and experimental purposes.
- It's a library focused on simplicity and extensibility, allowing easy tokenization, POS tagging, and named entity recognition in a single pipeline.

---

### Summary of the Video: Introduction to NLP and Sentiment Analysis  

**1. Overview and Recap**  
- NLP topics discussed previously: Stemming, Lemmatization, Stopwords (using NLTK and Python).  
- Objective: To understand how these concepts fit into the lifecycle of an NLP project.  

**2. Problem Statement: Sentiment Analysis**  
- The task: Sentiment analysis (classifying text as positive, negative, or neutral).  
- Key terms explained:  
  - **Corpus**: A collection of documents or sentences.  
  - **Vocabulary**: Unique words in the corpus.  

**3. Text Pre-Processing**  
- **Text Pre-processing (Part 1):**  
  1. **Tokenization**: Splitting a paragraph into sentences or sentences into words.  
  2. **Lowercasing**: Converting all words to lowercase to avoid treating “The” and “the” as different words.  
  3. **Regular Expressions**: Cleaning text by removing special characters or unwanted patterns.  

- **Text Pre-processing (Part 2):**  
  1. **Stemming**: Reducing words to their root form (e.g., "playing" → "play").  
  2. **Lemmatization**: Reducing words to their dictionary form while maintaining meaning.  
  3. **Stopwords**: Removing common words like "is," "the," and "and" that do not contribute to meaning.  

**4. Converting Text into Vectors**  
- After cleaning, text is transformed into numerical data for machine learning.  
- Techniques:  
  1. **One-Hot Encoding**: Represents words as binary vectors (e.g., "food" → `[0, 1, 0, 0]`).  
  2. **Bag of Words (BoW)**: Counts the frequency of words in a text.  
  3. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighs words based on importance in the document and corpus.  
  4. **Word2Vec**: Creates meaningful vector representations of words based on context.  
  5. **Average Word2Vec**: Aggregates word vectors for entire sentences or documents.  

**5. Training Machine Learning Models**  
- Steps:  
  1. Convert text into vectors.  
  2. Use numerical vectors as input to machine learning models.  
  3. Train models to classify sentiments using algorithms (ML or DL).  

**6. Future Topics**  
- Deep Learning Techniques: Word Embeddings, Transformers, and BERT.  
- Libraries: **Gensim** for implementing Word2Vec.  

### Flow of Sentiment Analysis Workflow  
1. **Data Preparation**: Collect and preprocess text.  
2. **Text Pre-processing**: Tokenization, lowercasing, stemming, lemmatization, removing stopwords, cleaning.  
3. **Feature Extraction**: Convert text into vectors using methods like BoW, TF-IDF, or Word2Vec.  
4. **Model Training**: Train ML algorithms with vectorized text.  
5. **Prediction and Evaluation**: Use the model to classify sentiments and evaluate accuracy.  

---

### **Continuing the Discussion on NLP**

After **text preprocessing**, where we performed tasks like:
1. **Stemming**: Reducing words to their root form (e.g., "running" → "run").
2. **Lemmatization**: Converting words to their base dictionary form (e.g., "better" → "good").
3. **Removing stop words**: Eliminating common but less meaningful words (e.g., "is", "the", "and").

The next step is to convert **text into vectors** (numerical representations), a critical step because machine learning models only work with numbers, not text.

---

### **Understanding One-Hot Encoding**
One-hot encoding is the simplest method to convert text into vectors.

---

#### **Steps in One-Hot Encoding**

1. **Find Unique Words (Vocabulary)**
   - Combine all sentences or documents into a single **corpus**.
   - Extract the **unique words** (vocabulary) from the corpus.

   **Example Corpus**:
   ```
   D1: "The food is good"
   D2: "The food is bad"
   D3: "Pizza is amazing"
   ```
   Combine the documents:  
   **"The food is good. The food is bad. Pizza is amazing."**

   **Vocabulary**:
   ```
   ['the', 'food', 'is', 'good', 'bad', 'pizza', 'amazing']
   ```
   Total unique words: **7**.

---

2. **Create One-Hot Vectors**
   - Each word in the vocabulary is assigned a unique position in a vector of length equal to the vocabulary size.
   - For a word, the position corresponding to it is marked as `1`, and all others are `0`.

   **Example Representation**:
   - "the": `[1, 0, 0, 0, 0, 0, 0]`
   - "food": `[0, 1, 0, 0, 0, 0, 0]`
   - "is": `[0, 0, 1, 0, 0, 0, 0]`
   - "good": `[0, 0, 0, 1, 0, 0, 0]`
   - "bad": `[0, 0, 0, 0, 1, 0, 0]`
   - "pizza": `[0, 0, 0, 0, 0, 1, 0]`
   - "amazing": `[0, 0, 0, 0, 0, 0, 1]`

---

3. **Encode Entire Sentences**
   - For each sentence, represent each word using its one-hot vector.
   - Combine the vectors for all words in the sentence.

   **Example for Sentence D1 ("The food is good")**:
   - "the" → `[1, 0, 0, 0, 0, 0, 0]`
   - "food" → `[0, 1, 0, 0, 0, 0, 0]`
   - "is" → `[0, 0, 1, 0, 0, 0, 0]`
   - "good" → `[0, 0, 0, 1, 0, 0, 0]`

   Combined Representation:
   ```
   [
     [1, 0, 0, 0, 0, 0, 0],  # "the"
     [0, 1, 0, 0, 0, 0, 0],  # "food"
     [0, 0, 1, 0, 0, 0, 0],  # "is"
     [0, 0, 0, 1, 0, 0, 0]   # "good"
   ]
   ```
   **Shape**: `4 x 7` (4 words × 7 unique vocabulary).

   **Example for D2 ("The food is bad")**:
   ```
   [
     [1, 0, 0, 0, 0, 0, 0],  # "the"
     [0, 1, 0, 0, 0, 0, 0],  # "food"
     [0, 0, 1, 0, 0, 0, 0],  # "is"
     [0, 0, 0, 0, 1, 0, 0]   # "bad"
   ]
   ```
   **Shape**: `4 x 7`.

---

### **Key Points**
1. **Vector Dimensions**: The vector size is equal to the number of unique words (vocabulary size).
2. **Sparse Representation**: Most elements in the vector are `0`, leading to inefficient storage.

---

### **Limitations of One-Hot Encoding**
1. **No Semantic Understanding**: Words like "good" and "amazing" are treated as completely unrelated, even though they are similar in meaning.
2. **High Dimensionality**: For large vocabularies, the vector size becomes enormous, making computations expensive.
3. **Context Ignorance**: Words are encoded independently, ignoring their context in a sentence.

---

### **What’s Next?**
- More advanced techniques like:
  1. **Bag of Words (BoW)**.
  2. **TF-IDF** (Term Frequency-Inverse Document Frequency).
  3. **Word Embeddings** (e.g., Word2Vec, GloVe).
  
  These methods address the limitations of one-hot encoding by considering word frequencies, importance, and context.

---
