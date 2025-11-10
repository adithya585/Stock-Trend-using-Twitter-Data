import re
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Optional: Tweepy setup (not used when training from CSV dataset below)
try:
    import tweepy as tw
    api_key = "vgU87NCZEPxg3lr6bNvf2xlHA"
    api_secret = "bUgGuPRupiystGHpntGYfa6F7VcDmzubrzVB1grQuAw34dmEkb"
    auth = tw.OAuthHandler(api_key, api_secret)
    twitter_api = tw.API(auth, wait_on_rate_limit=True)
except Exception:
    twitter_api = None


def preprocess_and_tokenize(train_data: pd.DataFrame):
    nltk.download('stopwords', quiet=True)
    stop_words = nltk.corpus.stopwords.words('english')

    pre_data = train_data["Text"].tolist()
    filtered_statements = []
    for statement in pre_data:
        words = statement.split()
        words = [word.lower() for word in words]
        filtered_words = [word for word in words if word not in stop_words]
        filtered_statement = ' '.join(filtered_words)
        filtered_statement = re.sub('', '', filtered_statement)
        filtered_statement = re.sub('https://.*', '', filtered_statement)
        filtered_statement = re.sub(r'[#$(@]\w+', '', filtered_statement)
        filtered_statements.append(filtered_statement)

    nltk.download('wordnet', quiet=True)
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        st = ""
        for w in w_tokenizer.tokenize(text):
            st = st + lemmatizer.lemmatize(w) + " "
        return st.strip()

    filterdata = pd.DataFrame(filtered_statements, columns=["tweet"])
    filterdata["tweet"] = filterdata.tweet.apply(lemmatize_text)

    filtered = filterdata['tweet'].values
    labels = train_data['Sentiment'].values
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(filtered, encoded_labels, test_size=0.7)

    vocab_size = 3000
    oov_tok = ''
    embedding_dim = 100
    max_length = 200
    padding_type = 'post'
    trunc_type = 'post'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index

    train_seq = tokenizer.texts_to_sequences(x_train)
    train_padded = pad_sequences(train_seq, padding=padding_type, maxlen=max_length)

    test_seq = tokenizer.texts_to_sequences(x_test)
    test_padded = pad_sequences(test_seq, padding=padding_type, maxlen=max_length)

    return tokenizer, train_padded, test_padded, y_train, y_test, max_length, vocab_size, embedding_dim


def build_model(vocab_size: int, embedding_dim: int, max_length: int):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate_model(model, test_padded, y_test):
    prediction = model.predict(test_padded)
    pred_labels = [1 if p >= 0.5 else 0 for p in prediction]
    acc = accuracy_score(y_test, pred_labels)
    cm = confusion_matrix(y_test, pred_labels)
    print("Accuracy of prediction on test set:", acc)
    print("Confusion matrix:\n", cm)


def try_unseen_data_inference(tokenizer, max_length, model):
    try:
        data = pd.read_csv("data/unseen_data.csv")
    except Exception:
        return

    tweets = data["Tweet"]
    labels = data["Sentiment"].values

    seq = tokenizer.texts_to_sequences(tweets.tolist())
    test_padded = pad_sequences(seq, padding='post', maxlen=max_length)

    prediction = model.predict(test_padded)
    pred_labels = [1 if p >= 0.5 else -1 for p in prediction]
    try:
        acc = accuracy_score(labels, pred_labels)
        print("Accuracy of prediction on unseen set:", acc)
        print("Confusion matrix (unseen):\n", confusion_matrix(labels, pred_labels))
    except Exception:
        pass


def main():
    # Load dataset (from local repo data folder)
    train_data = pd.read_csv("data/stock_data.csv")

    tokenizer, train_padded, test_padded, y_train, y_test, max_length, vocab_size, embedding_dim = preprocess_and_tokenize(train_data)

    model = build_model(vocab_size, embedding_dim, max_length)
    model.summary()

    num_epochs = 10
    history = model.fit(train_padded, y_train, epochs=num_epochs, verbose=1, validation_split=0.1)

    evaluate_model(model, test_padded, y_test)

    # Optional: run on unseen data if available
    try_unseen_data_inference(tokenizer, max_length, model)


if __name__ == "__main__":
    main()

