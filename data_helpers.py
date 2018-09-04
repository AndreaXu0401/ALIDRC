import numpy as np
import re
import itertools
from collections import Counter
from codecs import open
import gensim
import json
import pickle
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding = "UTF-8").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r",encoding = "UTF-8").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data_and_labels_discourse(data_file):
    all_examples = list(open(data_file, "r", encoding="UTF-8").readlines())
    x_text_arg1 = []
    x_text_arg2 = []
    labels = []
    for example in all_examples:
        segments = example.split('###')
        labels.append(segments[0])
        x_text_arg1.append(segments[1].replace("[","").replace("]",""))
        x_text_arg2.append(segments[2].replace("[","").replace("]",""))

    labels = [ [1, 0] if "Not" in l else [0, 1] for l in labels]

    x_arg1 = [sent for sent in x_text_arg1]
    x_arg2 = [sent for sent in x_text_arg2]

    x_arg1 = [s.split(", ") for s in x_arg1]
    x_arg2 = [s.split(", ")for s in x_arg2]





    return [x_arg1, x_arg2, labels]







def build_input_data(sentences,vocab,max_document_length,padding_word = "<PAD/>"):
    padded_sentences = [x[:max_document_length - 1] + [padding_word] * max(max_document_length - len(x), 1) for x in sentences]

    #print(padded_sentences)

    x_arg1 = np.array([[vocab.index(word) for word in sentence] for sentence in padded_sentences])

    return x_arg1



def load_pos_pkl(filename):

    file_pkl = open(filename, "rb")
    x_text_arg1, x_text_arg2, labels = pickle.load(file_pkl)
    file_pkl.close()
    labels_num = []

    for l in labels:
        if "Temporal" in l:
            labels_num.append([1, 0, 0, 0])

        if "Comparison" in l:
            labels_num.append([0, 1, 0, 0])

        if "Contingency" in l:
            labels_num.append([0, 0, 1, 0])

        if "Expansion" in l:
            labels_num.append([0, 0, 0, 1])
    return x_text_arg1,x_text_arg2,labels

def load_word_pkl(filename):
    file_pkl = open(filename, "rb")
    x_text_arg1, x_text_arg2, labels = pickle.load(file_pkl)
    file_pkl.close()

    labels_num = []

    for l in labels:
        if "Temporal" in l:
            labels_num.append([1,0,0,0])

        if "Comparison" in l:
            labels_num.append([0,1,0,0])

        if "Contingency" in l:
            labels_num.append([0,0,1,0])

        if "Expansion" in l:
            labels_num.append([0,0,0,1])

    return x_text_arg1, x_text_arg2, labels_num



def build_pos_vocab_embd(vocab_file_name):
    vocab = []
    vocab_file = open(vocab_file_name, "rb")
    vocab = pickle.load(vocab_file)
    vocab_file.close()



    embd =[]


    for i in range(len(vocab)):
        embd.append(np.random.uniform(-1,1,50).tolist())

    vocab.insert(0, "<PAD/>")
    embd.insert(0,[0.0] * 50)


    return vocab,embd


def build_word_vocab_embd(vocab_file_name):
    model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",
                                                            unicode_errors="ignore", binary=True)


    vocab_file = open(vocab_file_name, "rb")
    vocab = pickle.load(vocab_file)
    vocab_file.close()
    #vocab = list(set(vocab))

    embd =[]
    for word in vocab:
        try:
            embd.append(model[word])
        except:
            embd.append(np.random.uniform(-1, 1, 300).tolist())

    vocab.insert(0, "<PAD/>")
    embd.insert(0, [0] * 300)
    vocab.insert(1, "<UNK/>")
    embd.insert(1, np.random.uniform(-1, 1, 300).tolist())


    return vocab,embd




# if __name__ == "__main__":
#     train_file = open("./aaaaaa.pkl", "rb")
#     y_train, x_text_train_arg1, x_text_train_arg2 = pickle.load(train_file)
#     train_file.close()
#
#     print(x_text_train_arg1)
#
#     vocab_file = open('./pos_list.pkl', "rb")
#     vocab = pickle.load(vocab_file)
#     vocab_file.close()
#
#     vocab.insert(0, "<PAD/>")
#
#     x_train_arg1 = build_input_data(x_text_train_arg1, vocab, 80)
#
#     print(x_train_arg1)


