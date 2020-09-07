# Generate the classifier model from prepared .pkl drug label data
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def NB_classifier(drug_df):
    # vectorize the text of active ingredient
    active_train_text = drug_df["active_ingredient"]
    inactive_train_text = drug_df["inactive_ingredient"]
    train_text = active_train_text + " " + inactive_train_text

    train_X = CountVectorizer().fit_transform(train_text)

    train_y = drug_df["purpose"]

    mnb = MultinomialNB(alpha=.5)
    mnb.fit(train_X, train_y)

    print(train_text[200:202])
    print(mnb.predict(train_X[200:202]))
    print(mnb.score(train_X, train_y))
    return 0

if __name__ == "__main__":
    # read processed drug dataframe and take out invalid ingredient rows
    drug_df = pd.read_pickle("drug_df.pkl", compression="zip")

    print("Full DF with purpose:")
    print(drug_df) # 95315 rows

    print("DF with all valid ingredients:")
    print(drug_df.dropna(subset=["active_ingredient", "inactive_ingredient"])) # 94777 rows

    valid_df = drug_df.dropna(subset=["active_ingredient", "inactive_ingredient"])

    # find frequency of each purpose label
    purpose_freq = valid_df["purpose"].value_counts()
    print(len(purpose_freq))

    # set purpose class cutoff to 200 -> 64 purposes
    often_purpose_freq = purpose_freq[purpose_freq > 200]
    print(often_purpose_freq)

    # use only training data of those purposes
    train_df = valid_df[valid_df["purpose"].isin(often_purpose_freq.index)]
    print("DF with purpose frequency > 200:")
    print(train_df) # 44207 rows

    model = NB_classifier(train_df)