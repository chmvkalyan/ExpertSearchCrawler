import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import matplotlib.pyplot as plt
import pandas
from os import path
from tqdm.auto import tqdm
import string
import urllib
import tensorflow as tf
import keras
import nltk
from bs4 import BeautifulSoup
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse
from os import path
from tqdm.auto import tqdm
from nltk.stem.snowball import SnowballStemmer
import spacy
from tinydb import TinyDB

tf.get_logger().setLevel('INFO')

# Seed for numpy randomizer
SEED = 1337

# For model consistency and reproducibility
np.random.seed(SEED)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 11500000


def create_negative_training_set(db_path, pos_urls_path, neg_urls_path, neg_texts_path, verbose=True):
    """
    Creates negative datasets from the pages crawler downloaded
    :return:
    """
    db = TinyDB(db_path)
    with open(pos_urls_path, 'r') as b:
        bio_urls = b.readlines()
    bio_urls = [u.strip() for u in bio_urls]

    with open(neg_texts_path, "w") as pages, \
            open(neg_urls_path, 'w') as urls:
        for entry in tqdm(db):
            u = urlparse(entry["file"])
            if path.exists(u.path) and entry["url"] not in bio_urls:
                urls.write("{}\n".format(entry["url"]))
                with open(u.path, "r") as f:
                    content = f.readlines()
                pages.write("{}\n".format(" ".join(content)))
            else:
                if verbose:
                    print("Found {} in positive urls".format(entry["url"]))


class TextProcessor(object):
    def __init__(self):
        self.stemmer = SnowballStemmer(language="english")
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def process_text(self, text):
        tokens = word_tokenize(text)
        cleaned_tokens = [self.stemmer.stem(t) for t in tokens if
                          t not in string.punctuation and t not in self.stop_words]
        lemmas = [t.lemma_.lower() for t in nlp(" ".join(cleaned_tokens), disable=['ner', 'parser']) if
                  2 < len(t.text) < 20 and t.lemma_ != '-PRON-']
        return " ".join(lemmas)


def load_training_file(filename, processor, content_filter=None):
    """
    Loads a data file where each line is a document
    :param filename: path to the data file
    :param processor: text processor
    :param content_filter: filters file content
    :return: a list where each item is a line
    """
    print("Loading {}".format(filename))
    with open(filename, 'r') as f:
        content = []
        for line in tqdm(f.readlines()):
            content.append(processor.process_text(line.strip().lower()))

    if content_filter:
        filtered = list(filter(content_filter, content))
    else:
        filtered = content
    print("Accepted: {}/Discarded: {}".format(len(filtered), len(content) - len(filtered)))
    return content


def create_training_data(exported_dataframe_filename, pos_filename, neg_filename):
    """
    Creates training data frame from positive and negative labels
    :param exported_dataframe_filename: exported dataframe
    :param pos_filename: the path to positive text training data
    :param neg_filename: the path to negative text training data
    :return: a panda data frame
    """
    if path.exists(exported_dataframe_filename):
        return pandas.read_pickle(exported_dataframe_filename)

    processor = TextProcessor()
    pos = TextData(text_path=pos_filename, label=1, text_processor=processor)
    neg = TextData(text_path=neg_filename, label=0, text_processor=processor)
    df = pandas.concat([pos.to_df(), neg.to_df()], ignore_index=True).sample(frac=1)
    pandas.to_pickle(df, exported_dataframe_filename)
    return df


class TextData(object):
    """
    TextData represents a group of text data that have the same label
    """

    def __init__(self, text_path, label, text_processor, max_content_length=1000):
        self._df = pandas.DataFrame()
        self._df['text'] = load_training_file(
            text_path, processor=text_processor, content_filter=lambda sen: len(sen) < max_content_length)
        self._df['label'] = [label] * len(self._df)

    def size(self):
        return len(self._df)

    def sample(self, num_samples):
        return self._df[:num_samples]

    def to_df(self):
        return self._df


def plot_history(history, plot_filename=None):
    """
    Plots model training history (model accuracy and model loss)
    :param plot_filename: if provided, the plot will be saved
    :param history: training history
    :return:
    """
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Model Accuracy/Loss Plots')
    # Plots model accuracy
    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].title.set_text('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'test'], loc='lower right')

    # Plots model loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].title.set_text('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'test'], loc='upper right')

    fig.tight_layout()
    plt.show()

    if plot_filename:
        fig.savefig(plot_filename)


def print_tensor_env():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


def create_tensor_dataset(exported_dataframe_filename, pos_filename, neg_filename,
                          train_rate=0.7, val_rate=0.15, test_rate=0.15):
    """
    Creates tensor dataset from positive and negative training texts
    :param exported_dataframe_filename: exported dataframe
    :param pos_filename: the path to positive text training data
    :param neg_filename: the path to negative text training data
    :param train_rate: train data split ratio
    :param val_rate: validation data split ratio
    :param test_rate: test data split ratio
    :return: training_dataset (training data), val_dataset (validation data), test_dataset (test data)
    """

    df = create_training_data(
        exported_dataframe_filename=exported_dataframe_filename,
        pos_filename=pos_filename,
        neg_filename=neg_filename)

    dataset = tf.data.Dataset.from_tensor_slices((df.text.values, df.label.values))
    size = dataset.cardinality().numpy()
    train_size = int(train_rate * size)
    val_size = int(val_rate * size)
    test_size = int(test_rate * size)

    full_dataset = dataset.shuffle(buffer_size=len(df.label.values))
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset, val_dataset, test_dataset


class BioIdentifier(object):
    def __init__(self, model, threshold=0):
        self._model = tf.keras.models.load_model(model)
        self._processor = TextProcessor()
        self._threshold = threshold

    def is_bio_text(self, text):
        processed = self._processor.process_text(text)
        #print("sentence: {}".format(processed))
        return self._model.predict([processed])[0][0] > self._threshold

    def is_bio_html_content(self, content):
        text = BeautifulSoup(content, 'html.parser').get_text(separator=" ")
        return self.is_bio_text(text)

    def is_bio_url(self, url):
        with urllib.request.urlopen(url) as response:
            html = response.read()
        return self.is_bio_html_content(html)
