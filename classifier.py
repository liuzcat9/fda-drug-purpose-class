# Generate the classifier model from prepared .pkl drug label data
import numpy as np
import pandas as pd
import pickle
from joblib import dump, load

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def NB_classifier(train_text, y):
    nb_pipe = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('mnb', MultinomialNB(alpha=.001))
    ])

    # split into training and validation
    X_train, X_test, y_train, y_test = train_test_split(train_text, y, test_size=.25)

    nb_pipe.fit(train_text, y)

    print(nb_pipe.predict(X_train[200:202]))
    print(cross_val_score(nb_pipe, train_text, y, cv=5))

    # fit full data
    nb_pipe.fit(train_text, y)
    return nb_pipe

def one_vs_rest(train_text, y):
    # note: takes much too long
    ovr_pipe = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('ovr', OneVsRestClassifier(SVC()))
    ])

    print(cross_val_score(ovr_pipe, train_text, y, cv=5))
    return ovr_pipe

def LinearSVC_classifier(train_text, y):
    svc_pipe = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('svc', LinearSVC(multi_class="ovr", penalty="l2", max_iter=10000, C=1.0))
    ])

    print(cross_val_score(svc_pipe, train_text, y, cv=5, scoring="accuracy"))

    svc_pipe.fit(train_text, y)
    return svc_pipe

def RF_classifier(train_text, y):
    nb_pipe = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('rf', RandomForestClassifier(n_estimators=10000, max_depth=3, n_jobs=4))
    ])

    print(cross_val_score(nb_pipe, train_text, y, cv=5, scoring="accuracy"))

    # fit full data
    nb_pipe.fit(train_text, y)
    return nb_pipe

def neural_network(train_text, y):
    # process all data into tensorflow-friendly input
    max_vocab, X_train = tokenize_nn(train_text)
    one_hot_y = label_nn(y)

    # build NN model
    embedding_dim = 100
    model = create_nn_model(max_vocab, embedding_dim)

    # fit model to data
    history = model.fit(X_train, one_hot_y,
                        validation_split=0.2, epochs=5, verbose=True, batch_size=20)

    return model

# helper function to token words into keras-friendly format
def tokenize_nn(train_text):
    # tokenize training text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_text)
    max_vocab = len(tokenizer.word_index) + 1  # reserve 0 spot
    print("Tokenized max vocab: ", str(max_vocab))

    # save tokenizer
    pickle.dump(tokenizer, open("nn_purpose_model/tokenizer.pkl", 'wb'))

    X_train = tokenizer.texts_to_sequences(train_text)

    # pad sequences
    X_train = pad_sequences(X_train, padding='post', maxlen=200)

    return max_vocab, X_train

# helper function
def label_nn(y):
    # encode y output using sklearn to integers
    label = LabelEncoder()
    y_ints = label.fit_transform(y)

    # save y labels
    pickle.dump(label, open("nn_purpose_model/labels.pkl", 'wb'))

    one_hot_y = tf.one_hot(y_ints, 64)

    return one_hot_y

# helper function to create layers of nn model
def create_nn_model(max_vocab, embedding_dim):
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(max_vocab, embedding_dim, input_length=200))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(192, activation='sigmoid'))
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(73, activation='relu'))
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.Dense(64, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# function to test loaded model and label prediction
def run_nn_model(test_text, y_test):
    # load tokenizer
    tokenizer = pickle.load(open("nn_purpose_model/tokenizer.pkl", 'rb'))
    X_test = tokenizer.texts_to_sequences(test_text)

    # pad sequences
    X_test = pad_sequences(X_test, padding='post', maxlen=200)

    model = keras.models.load_model("nn_purpose_model")

    # convert y portion
    y_label = pickle.load(open("nn_purpose_model/labels.pkl", 'rb'))
    y_ints = y_label.transform(y_test)
    y_one_hot = tf.one_hot(y_ints, 64)

    # print("Predicting for: ")
    # print(test_text)
    # print(y_test, y_ints)
    print(model.evaluate(X_test, y_one_hot))
    # print(np.argmax(model.predict(X_test), axis=-1))
    y_str = y_label.inverse_transform(np.argmax(model.predict(X_test), axis=-1))

    # print(y_str)

def save_nn_model(model, filename):
    tf.keras.models.save_model(model, filename)

def pickle_model(model, filename):
    dump(model, filename + ".joblib")
    print("Dumped model " + filename)

# read processed drug dataframe and take out invalid ingredient rows
def read_and_process_valid_df(df_pkl):
    drug_df = pd.read_pickle(df_pkl, compression="zip")

    print("Full DF with purpose:")
    print(drug_df)  # 95315 rows

    print("DF with all valid ingredients:")
    print(drug_df.dropna(subset=["active_ingredient", "inactive_ingredient"]))  # 94777 rows

    valid_df = drug_df.dropna(subset=["active_ingredient", "inactive_ingredient"])

    return valid_df

# obtain purpose class labels > 200 count in valid dataset
def find_purpose_classes(valid_df):
    # find frequency of each purpose la)bel
    purpose_freq = valid_df["purpose"].value_counts()
    print(len(purpose_freq))

    # set purpose class cutoff to 200 -> 64 purposes
    often_purpose_freq = purpose_freq[purpose_freq > 200]
    print(often_purpose_freq)

    return often_purpose_freq

# obtain training ingredient text and labels
def get_ingredient_training_data(valid_df, often_purpose_freq):
    # use only training data of those purposes
    train_df = valid_df[valid_df["purpose"].isin(often_purpose_freq.index)]
    y = train_df["purpose"]
    print("DF with purpose frequency > 200:")
    print(train_df)  # 44207 rows

    # vectorize the text of active and inactive ingredients
    active_train_text = train_df["active_ingredient"]
    inactive_train_text = train_df["inactive_ingredient"]
    train_text = active_train_text + " " + inactive_train_text

    return train_text, y

if __name__ == "__main__":
    valid_df = read_and_process_valid_df("drug_df.pkl")
    often_purpose_freq = find_purpose_classes(valid_df)

    train_text, y = get_ingredient_training_data(valid_df, often_purpose_freq)

    # model = NB_classifier(train_text, y)
    # model = RF_classifier(train_text, y)
    # 90% accuracy
    model = LinearSVC_classifier(train_text, y)
    # pickle_model(model, "purpose_model")

    # neural network
    # model = neural_network(train_text, y)
    # save_nn_model(model, "nn_purpose_model")

    # test loaded model
    run_nn_model(train_text[0:1000], y[0:1000])
    # print("Actual purposes: ")
    # print(train_df.iloc[0:5]["purpose"])