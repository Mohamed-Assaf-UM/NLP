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

Here's a simplified explanation of **advantages and disadvantages of one-hot encoding** in NLP, summarizing the details:

---

### **Advantages of One-Hot Encoding**
1. **Easy to Implement**:
   - Libraries like `sklearn` provide `OneHotEncoder`.
   - Pandas offers `pd.get_dummies` for simple one-hot encoding.

2. **Transforms Words into Numerical Form**:
   - Converts text data into vectors that machine learning algorithms can process.

---

### **Disadvantages of One-Hot Encoding**
1. **Sparse Matrix**:
   - Results in arrays/matrices with mostly zeros and a few ones.
   - Sparse matrices consume more memory and computational resources.
   - They often lead to **overfitting** in machine learning models.

2. **No Fixed Input Size**:
   - The vector size depends on the vocabulary, making it inconsistent for machine learning models that require fixed-sized input.

3. **No Semantic Meaning Captured**:
   - Words like *food* and *pizza*, which are similar in meaning, are treated as unrelated.
   - Distance metrics (like cosine similarity) show equal distances between vectors, ignoring the relationships between words.

4. **Out-of-Vocabulary (OOV) Issue**:
   - If a new word (not in the training vocabulary) appears in test data, it cannot be represented in the one-hot encoding format.

5. **Scalability Problems**:
   - For large vocabularies (e.g., 50,000 unique words), the size of the matrix grows significantly, leading to inefficiency and increased sparsity.

---

### **Conclusion**
One-hot encoding is simple and useful for small datasets but has significant limitations for real-world NLP tasks. Advanced methods like **Bag of Words**, **TF-IDF**, or **Word Embeddings** (e.g., Word2Vec, GloVe) are used to overcome these drawbacks.

---
Here's a simplified breakdown of the explanation of **Bag of Words (BoW)** from your transcript, formatted for easy note-taking and understanding:

---

### **Bag of Words (BoW) Explanation**
#### **Purpose:**
- Converts text data into numerical vectors for machine learning tasks like text classification (e.g., spam detection, sentiment analysis).

#### **Steps to Implement Bag of Words:**
1. **Input Data:**
   - Example sentences:
     - Sentence 1: "He is a good boy."
     - Sentence 2: "She is a good girl."
     - Sentence 3: "Boy and girl are good."
   - All sentences are labeled as **positive (1)** for supervised learning.

2. **Preprocessing:**
   - **Lowercase Words:** 
     - Convert all text to lowercase to treat words like "Boy" and "boy" as the same.
   - **Remove Stopwords:**
     - Eliminate common words like "he," "is," "a," "and," etc., which do not contribute to meaning.
     - Result after preprocessing:
       - Sentence 1 → "good boy"
       - Sentence 2 → "good girl"
       - Sentence 3 → "boy girl good"

3. **Build Vocabulary:**
   - Extract unique words from all sentences:
     - Vocabulary = ["good", "boy", "girl"]
   - Count word frequencies across all sentences:
     - "good" → 3 times
     - "boy" → 2 times
     - "girl" → 2 times

4. **Vectorization (Feature Representation):**
   - Represent each sentence as a vector of word occurrences (binary or frequency):
     - Vocabulary Order: ["good", "boy", "girl"]
     - Sentence 1 → [1, 1, 0] ("good" and "boy" are present)
     - Sentence 2 → [1, 0, 1] ("good" and "girl" are present)
     - Sentence 3 → [1, 1, 1] ("good," "boy," and "girl" are present)

5. **Binary vs. Frequency BoW:**
   - **Binary BoW:** 
     - Words are marked as `1` if present, `0` if absent, irrespective of frequency.
   - **Frequency BoW:** 
     - Words are represented by their frequency in the sentence.

#### **Applications:**
- Sentiment Analysis
- Spam Detection
- General Text Classification Tasks

#### **Key Notes:**
- BoW creates a fixed-length feature vector for each sentence based on the vocabulary size.
- Uncommon words with low frequency may be excluded to reduce dimensionality.
- These vectors are fed into machine learning models for classification or other tasks.

---
### **Bag of Words (BoW): Advantages and Disadvantages**  

#### **Advantages of Bag of Words**
1. **Simple and Intuitive**:  
   - BoW is straightforward to implement and easy to understand.  
   - Converting text into fixed-size vectors is systematic.  

2. **Fixed-Size Input for ML Algorithms**:  
   - Regardless of sentence length, the vector representation is based on vocabulary size.  
   - This uniformity is essential for machine learning algorithms that require fixed-size inputs.  

---

#### **Disadvantages of Bag of Words**
1. **Sparse Matrix Problem**:  
   - Large vocabulary results in vectors with many zeros.  
   - This sparsity can increase storage and computational requirements, leading to potential overfitting.  

2. **Ignores Word Order (No Contextual Meaning)**:  
   - Word sequences are disregarded, leading to loss of sentence semantics.  
   - Example: "The food is good" and "The food is not good" might appear similar despite opposite meanings.  

3. **Out-of-Vocabulary (OOV) Issues**:  
   - New words not in the training vocabulary are ignored during testing.  
   - This exclusion can negatively impact predictions if the new word is significant.  

4. **Limited Semantic Capture**:  
   - BoW focuses only on word presence or absence without understanding context or relationships between words.  
   - Example: Synonyms like "happy" and "joyful" are treated as entirely unrelated.  

5. **Misleading Similarity**:  
   - Similarity measures (e.g., cosine similarity) may show unrelated or opposite sentences as close.  
   - Example: "The food is good" vs. "The food is not good" may appear similar due to shared words despite opposite meanings.  

---

#### **Key Takeaways**
- **Strengths**: Simple, easy to implement, and creates fixed-size vectors suitable for ML models.  
- **Weaknesses**: Lacks the ability to handle sparse data, word order, OOV words, and semantic understanding effectively.  

---

#### **Next Steps**
- Explore advanced techniques like **Word2Vec**, **TF-IDF**, and **Word Embeddings** that address BoW's shortcomings.  
- These techniques focus on improving semantic representation, reducing sparsity, and handling OOV challenges.  

---

### Step 1: **Importing the Dataset**
```python
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])
```
- **Purpose**: Load the SMS dataset into a DataFrame.  
- **Parameters**:
  - `sep='\t'`: Specifies that the file uses tabs (`\t`) to separate the columns.
  - `names=["label", "message"]`: Assigns column names to the loaded dataset.

#### Example Dataset:
| **label** | **message**                     |
|-----------|---------------------------------|
| ham       | Hello, how are you?            |
| spam      | Win a free iPhone now!         |
| ham       | Meet me at 5 PM.               |

---

### Step 2: **Data Cleaning and Preprocessing**
```python
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
```
- **Purpose**: Prepare the text data by:
  - Removing unnecessary characters (e.g., punctuation, numbers).
  - Converting words to lowercase.
  - Removing common words like "the," "is," etc. (stop words).
  - Stemming words to reduce them to their base/root form (e.g., "playing" → "play").

---

#### Step 2.1: **Iterate through each message**
```python
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])  # Remove non-alphabetic characters.
    review = review.lower()  # Convert to lowercase.
    review = review.split()  # Split into individual words.
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # Remove stop words and apply stemming.
    review = ' '.join(review)  # Join the words back into a sentence.
    corpus.append(review)  # Add the cleaned message to the corpus.
```

#### Simplified Example:
- Original: **"Hello, how are you?"**
  - After `re.sub`: "Hello how are you"
  - After `.lower()`: "hello how are you"
  - After `split()`: `['hello', 'how', 'are', 'you']`
  - After stopword removal and stemming: `['hello']`
  - Result: `"hello"`

**Corpus after processing all messages**:
```python
corpus = ["hello", "win free iphon", "meet pm"]
```

---

### Step 3: **Create the Bag of Words Model**
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=100, binary=True)
X = cv.fit_transform(corpus).toarray()
```
- **Purpose**: Convert the `corpus` into a numerical format (Bag of Words) for use in machine learning models.
  - `max_features=100`: Only keep the top 100 most frequent words.
  - `binary=True`: Represent words as 1s (present) and 0s (absent), instead of their frequency.

#### Vocabulary:
The `CountVectorizer` creates a vocabulary based on the words in the corpus:
```python
['free', 'hello', 'iphon', 'meet', 'pm', 'win']
```

#### Bag of Words Matrix (X):
Each row corresponds to a message, and each column corresponds to a word from the vocabulary:
| **free** | **hello** | **iphon** | **meet** | **pm** | **win** |
|----------|-----------|-----------|----------|--------|---------|
| 0        | 1         | 0         | 0        | 0      | 0       |
| 1        | 0         | 1         | 0        | 0      | 1       |
| 0        | 0         | 0         | 1        | 1      | 0       |

---

### Final Explanation:
1. **Data Cleaning**:
   - Removed unnecessary characters and stopwords.
   - Converted text to lowercase.
   - Stemmed words to their base form.

2. **Bag of Words Model**:
   - Created a fixed vocabulary from the corpus.
   - Converted each message into a numerical vector based on the presence/absence of words.

---

### Simplified Explanation of N-grams

N-grams are a technique in **Natural Language Processing (NLP)** used to represent text in a way that captures the context of words by considering their combinations in a sequence.

---

### **Key Concepts from the Explanation**

1. **The Problem with Bag of Words**:
   - The **Bag of Words (BoW)** model treats sentences as collections of individual words, ignoring their sequence.
   - Example:
     - Sentence 1: "The food is good."
     - Sentence 2: "The food is not good."
   - BoW will consider both sentences almost identical except for the word "not," even though they convey opposite meanings.

2. **How N-grams Solve This**:
   - N-grams capture sequences of *n* consecutive words in a text.
   - By including word combinations, the context of the sentence improves.
     - **Bigram (n=2)**: Looks at pairs of consecutive words like *"food is"* or *"not good."*
     - **Trigram (n=3)**: Looks at triplets like *"food is good"* or *"food not good."*

---

### **Example Walkthrough**
#### Sentences:
1. "The food is good."
2. "The food is not good."

#### Vocabulary from Words (Unigram):
After stopword removal (`the`, `is` are removed):
- Vocabulary: `['food', 'not', 'good']`

BoW representation:
| **food** | **not** | **good** |
|----------|----------|----------|
| 1        | 0        | 1        | *(Sentence 1)*  
| 1        | 1        | 1        | *(Sentence 2)*  

**Problem**: The vectors are almost the same except for "not," failing to capture the opposing sentiment.

---

#### Vocabulary from Bigrams (n=2):
Bigram vocabulary:  
`['food good', 'food not', 'not good']`

Representation:
| **food good** | **food not** | **not good** |
|---------------|--------------|--------------|
| 1             | 0            | 0            | *(Sentence 1)*  
| 0             | 1            | 1            | *(Sentence 2)*  

**Advantage**: The vectors now clearly distinguish the two sentences.

---

#### Vocabulary from Trigrams (n=3):
Trigram vocabulary:
`['food is good', 'food is not', 'is not good']`

Representation:
| **food is good** | **food is not** | **is not good** |
|------------------|-----------------|-----------------|
| 1                | 0               | 0               | *(Sentence 1)*  
| 0                | 1               | 1               | *(Sentence 2)*  

**Advantage**: Even more context is captured compared to bigrams.

---

### **Practical N-gram Use in Scikit-learn**
In Scikit-learn's `CountVectorizer`:
- `ngram_range=(1, 1)`: Only unigram (single words).
- `ngram_range=(1, 2)`: Combination of unigram and bigram.
- `ngram_range=(2, 3)`: Combination of bigram and trigram.

Example code:
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The food is good", "The food is not good"]

# Unigram and Bigram
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())  # Display vocabulary
print(X.toarray())  # Display vectorized representation
```

---

### **Summary**
1. **N-grams** enhance context by capturing word sequences.
2. **Applications**:
   - Sentiment analysis.
   - Text classification.
   - Contextual understanding of sentences.
3. N-grams are a powerful way to improve upon traditional BoW models by considering the relationships between words in a sentence.
### Practical Example: Using N-grams with `ngram_range=(2,3)`

In this example, we extend the **Bag of Words (BoW)** model by including **bigrams (n=2)** and **trigrams (n=3)**. Here's how it works:

---

### **Key Steps in the Code**

#### 1. **Setup the CountVectorizer**
- **`ngram_range=(2, 3)`**:
  - Includes **bigrams** (combinations of 2 consecutive words) and **trigrams** (combinations of 3 consecutive words).
- **`max_features=100`**:
  - Limits the number of features (vocabulary) to 100 most frequent terms.
- **`binary=True`**:
  - Creates a binary vector where each value is 1 if the n-gram is present, 0 otherwise.

#### 2. **Transforming the Corpus**
The `fit_transform()` method:
- Extracts bigrams and trigrams from the `corpus`.
- Converts the text into a vectorized representation.

---

### **Simplified Example**

#### Corpus:
```python
corpus = [
    "The food is good",
    "The food is not good",
    "Food quality is not bad"
]
```

#### **N-grams with `ngram_range=(2,3)`**

1. **Bigrams (n=2)**:
   - Example: `"The food"`, `"food is"`, `"is good"`, `"food not"`, `"not bad"`

2. **Trigrams (n=3)**:
   - Example: `"The food is"`, `"food is good"`, `"is not bad"`

---

#### **Code Implementation**:
```python
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer with bigrams and trigrams
cv = CountVectorizer(max_features=100, binary=True, ngram_range=(2, 3))
X = cv.fit_transform(corpus).toarray()

# Vocabulary (n-grams extracted)
print("Vocabulary:\n", cv.vocabulary_)

# Vectorized representation of the corpus
print("\nVectorized Corpus:\n", X)
```

---

#### **Output Explanation**

**Vocabulary (Extracted N-grams):**
- The `cv.vocabulary_` gives the mapping of n-grams to their index in the vector:
  ```
  {
    'food is': 0,
    'food is good': 1,
    'food is not': 2,
    'is not': 3,
    'is not bad': 4,
    'not bad': 5,
    'not good': 6
  }
  ```

**Vectorized Corpus:**
Each row corresponds to a sentence, and each column corresponds to an n-gram:
- **Matrix (X):**
  ```
  [[1 1 0 0 0 0 0]   # "The food is good"
   [1 0 1 0 0 0 1]   # "The food is not good"
   [0 0 1 1 1 1 0]]  # "Food quality is not bad"
  ```

---

### **How N-grams Improve Context**
1. **Capturing Relationships**:
   - With bigrams and trigrams, the model learns relationships between consecutive words (e.g., "not good" vs. "is not").
2. **Better Differentiation**:
   - The n-gram representation distinguishes "The food is good" from "The food is not good" by identifying meaningful word sequences like "not good."

---

### **Summary**
- **`ngram_range=(2,3)`** extracts bigrams and trigrams to improve the contextual understanding of the corpus.
- Bigrams capture two-word combinations, and trigrams capture three-word combinations.
- This approach addresses limitations of simple Bag of Words by incorporating word order and relationships into the feature set.
---
  
