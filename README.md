# RNN-Based Sentiment Analysis Using PyTorch

## 📌 Project Overview

This project implements a **Recurrent Neural Network (RNN)** using PyTorch to perform sentiment analysis on IMDB movie reviews.

The model classifies movie reviews into two sentiment categories:

- **Positive**
- **Negative**

The project uses Natural Language Processing (NLP) techniques to clean and preprocess textual data, convert reviews into numerical representations using TF-IDF, and train an RNN model for binary sentiment classification.

The objective is to understand how recurrent neural networks can be applied to text classification problems.

## 🎯 Project Objectives

- Load and explore the IMDB movie review dataset.
- Remove duplicate reviews.
- Perform text preprocessing and normalization.
- Remove URLs, HTML tags, and punctuation.
- Apply tokenization and stemming.
- Encode sentiment labels into numerical values.
- Convert text into numerical features using TF-IDF.
- Build an RNN model using PyTorch.
- Train the model using backpropagation and Adam optimization.
- Evaluate performance using test accuracy.

## 🛠️ Technologies Used

- **Python** — Programming language
- **PyTorch** — Deep learning framework
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **NLTK** — Natural Language Processing
- **Scikit-learn** — TF-IDF, label encoding, and dataset splitting
- **Matplotlib** — Visualization
- **Jupyter Notebook** — Interactive development environment

## 📂 Project Structure

```text
RNN-Sentiment-Analysis/
│
├── RNN_for_sentimentanalysis.ipynb
├── IMDB Dataset.csv
└── README.md
```

## 📊 Dataset Description

The project uses the **IMDB Movie Reviews Dataset**, containing movie reviews and their corresponding sentiment labels.

| Dataset Property | Value |
|---|---:|
| Original records | 50,000 |
| Original columns | 2 |
| Columns | `review`, `sentiment` |
| Positive reviews | 25,000 |
| Negative reviews | 25,000 |
| Missing values | 0 |
| Records after duplicate removal | 49,582 |

### Dataset Features

| Column | Description |
|---|---|
| `review` | Text containing the movie review |
| `sentiment` | Sentiment category: positive or negative |

The dataset is balanced, with equal numbers of positive and negative reviews before duplicate removal.

## ⚙️ Project Workflow

### 1. Import Libraries and Load Dataset

The notebook uses Pandas to load the IMDB dataset.

```python
import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

print(df.shape)
print(df.head())
print(df.isnull().sum())
```

### 2. Remove Duplicate Reviews

Duplicate records are removed to reduce repeated observations.

```python
df.drop_duplicates(inplace=True)
```

After removing duplicates, the dataset contains 49,582 records.

### 3. Text Preprocessing

The notebook applies several text-cleaning operations.

**Convert text to lowercase**

```python
df["review"] = df["review"].str.lower()
```

**Remove URLs**

```python
import re

def remove_urls(text):
    return re.sub(r"http\S+", "", text)

df["review"] = df["review"].apply(remove_urls)
```

**Remove HTML tags**

```python
def remove_html(text):
    return re.sub(r"<.*?>", "", text)

df["review"] = df["review"].apply(remove_html)
```

**Remove punctuation**

```python
def remove_punctuations(text):
    return re.sub(r"[^\w\s]", "", text)

df["review"] = df["review"].apply(remove_punctuations)
```

### 4. Tokenization and Stopword Processing

The notebook uses NLTK for tokenization and English stopword processing.

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```

The notebook defines a stopword-removal function and applies it to the reviews.

**Implementation note:** The current function calls `text.replace(word, "")` without assigning the returned string back to `text`. Therefore, stopwords are not actually removed by this function. This can be corrected in a future improvement.

### 5. Stemming

Porter Stemmer is used to reduce words to their root forms.

```python
from nltk.stem import PorterStemmer

def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []

    tokens = word_tokenize(text)

    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)

    return " ".join(stemmed_words)

df["review"] = df["review"].apply(stemming)
```

For example:
- playing → play
- studies → studi
- connected → connect

### 6. Encode Sentiment Labels

The categorical sentiment values are converted into numerical labels using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["sentiment"] = le.fit_transform(df["sentiment"])
```

The resulting labels are:

| Sentiment | Encoded Value |
|---|---:|
| Negative | 0 |
| Positive | 1 |

### 7. Convert Text into Numerical Features Using TF-IDF

The notebook uses `TfidfVectorizer` to convert processed reviews into numerical feature vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(max_features=5000)

X = tf.fit_transform(df["review"])
```

**TF-IDF (Term Frequency–Inverse Document Frequency)** represents the importance of words in documents.

Configuration:
- Maximum features: 5,000
- Input: Processed movie reviews
- Output: Numerical TF-IDF feature vectors

### 8. Split the Dataset

The notebook divides the dataset into training and testing sets using an 80:20 ratio.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

| Dataset | Samples |
|---|---:|
| Training | 39,665 |
| Testing | 9,917 |
| Total | 49,582 |

### 9. Convert Data into PyTorch Tensors

The sparse TF-IDF matrices are converted into dense arrays and then PyTorch tensors.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train = X_train.toarray()
X_test = X_test.toarray()

train_set = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float()
)

test_set = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float()
)
```

### 10. Create DataLoaders

The notebook uses a batch size of 64.

```python
train_loader = DataLoader(
    train_set,
    shuffle=True,
    batch_size=64
)

test_loader = DataLoader(
    test_set,
    shuffle=True,
    batch_size=64
)
```

### 11. Build the RNN Architecture

The notebook implements a basic Recurrent Neural Network using PyTorch.

```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out
```

**Model architecture:**

| Layer | Configuration |
|---|---|
| Input | 5,000 TF-IDF features |
| RNN layer | 128 hidden units |
| Number of recurrent layers | 1 |
| Fully connected layer | 128 → 1 |
| Output | Binary sentiment score |

**Important architectural detail:** The notebook adds a singleton sequence dimension using `unsqueeze(1)`. Consequently, each 5,000-dimensional TF-IDF vector is passed to the RNN as a sequence of length one. This is a basic RNN implementation rather than a word-by-word sequential text model.

### 12. Configure Loss Function and Optimizer

The notebook uses Binary Cross Entropy Loss and the Adam optimizer.

```python
model = RNN(input_size=5000)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())
```

- **Loss function:** Binary Cross Entropy
- **Optimizer:** Adam
- **Task:** Binary sentiment classification

### 13. Train the RNN Model

The model is trained for 10 epochs.

```python
epochs = 10

for epoch in range(epochs):
    model.train()

    for Xb, yb in train_loader:
        optimizer.zero_grad()

        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)

        outputs = torch.sigmoid(outputs.squeeze())

        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()

    print(
        f"Epoch = {epoch + 1}/{epochs} "
        f"and loss = {loss.item()}"
    )
```

Training involves:
1. Forward propagation.
2. Calculating binary classification loss.
3. Backpropagation.
4. Updating model parameters using Adam.

### 14. Evaluate Model Accuracy

The notebook evaluates the model on the test dataset using a threshold of 0.5.

```python
model.eval()

with torch.no_grad():
    correct_vals = 0
    tot_vals = 0

    for Xb, yb in test_loader:
        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)

        predicted = (
            torch.sigmoid(outputs.squeeze()) > 0.5
        ).float()

        tot_vals += yb.size(0)

        correct_vals += (
            predicted == yb
        ).sum().item()

    accuracy = correct_vals / tot_vals * 100

    print(f"Accuracy = {accuracy}")
```

## 📈 Results

<badge color="success">86.78% accuracy</badge>

The notebook's recorded test accuracy is:

| Evaluation Metric | Result |
|---|---:|
| Training samples | 39,665 |
| Testing samples | 9,917 |
| Input features | 5,000 |
| Hidden size | 128 |
| RNN layers | 1 |
| Training epochs | 10 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss function | Binary Cross Entropy |
| **Test accuracy** | **86.78%** |

The model correctly classified approximately 86.78% of the test reviews.

## 🔍 Key Concepts

| Concept | Description |
|---|---|
| NLP | Techniques for processing and analyzing human language. |
| Tokenization | Splits text into individual tokens or words. |
| Stopwords | Common words that may be removed during text preprocessing. |
| Stemming | Reduces words to their root forms. |
| TF-IDF | Converts text into numerical features based on word importance. |
| RNN | Neural network designed to process sequential information. |
| Hidden State | Stores information passed through recurrent steps. |
| Sigmoid | Converts model output into a probability between 0 and 1. |
| BCELoss | Measures binary classification prediction error. |
| Adam | Adaptive optimizer used to train neural networks. |

## 💻 Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd RNN-Sentiment-Analysis
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Install Dependencies

```bash
pip install numpy pandas nltk scikit-learn torch jupyter
```

### 3. Verify Dataset Location

Ensure `IMDB Dataset.csv` is located in the same directory as `RNN_for_sentimentanalysis.ipynb`.

### 4. Run the Notebook

```bash
jupyter notebook RNN_for_sentimentanalysis.ipynb
```

Run the cells sequentially to preprocess reviews, generate TF-IDF features, train the RNN, and evaluate sentiment classification accuracy.

## 🚀 Applications

Sentiment analysis can be used for:

- Movie review classification
- Customer feedback analysis
- Product review analysis
- Social media sentiment monitoring
- Brand reputation analysis
- Opinion mining

## 🔮 Future Improvements

- Correct the stopword-removal function.
- Fit the TF-IDF vectorizer only on training data to avoid data leakage.
- Use a word-level token sequence instead of a single-step TF-IDF input.
- Experiment with LSTM and GRU architectures.
- Add a validation dataset and monitor validation loss.
- Use `BCEWithLogitsLoss` to combine sigmoid and binary cross-entropy more numerically stably.
- Add precision, recall, F1-score, and a confusion matrix.
- Add dropout and regularization to reduce overfitting.
- Save the trained model and fitted TF-IDF vectorizer for future predictions.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (Artificial Intelligence and Machine Learning)

## 📄 License

This project is intended for educational and learning purposes. Add a license file if you plan to distribute it under a specific open-source license.
