# In this section, we load all the requisite libaries.
import os
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import collections
import random
import pickle

import gensim

# nlp = spacy.load('en')
import en_core_web_sm
nlp = en_core_web_sm.load()


## Submission.py for COMP6714-Project2
###################################################################################################################

VOCABULARY_SIZE = 15000
BATCH_SIZE = 64      # Size of mini-batch for skip-gram model.
SKIP_WINDOW = 2       # How many words to consider left and right of the target word.
NUM_SKIPS = 4         # How many times to reuse an input to generate a label.
NUM_SAMPLED= 800      # Sample size for negative examples.
LEARNING_RATE = 0.003

data_index = 0
# the variable is abused in this implementation.
# Outside the sample generation loop, it is the position of the sliding window: from data_index to data_index + span
# Inside the sample generation loop, it is the next word to be added to a size-limited buffer.

input_dir = './BBC_Data.zip'

## Output file name to store the final trained embeddings.
embedding_file_name = 'adjective_embeddings.txt'

## Fixed parameters
num_steps = 100001
# num_steps = 10
embedding_dim = 200

model_file = 'adjective_embeddings.txt'

def generate_batch(data, count, dictionary, reverse_dictionary, adjective, batch_size, num_samples, skip_window):
    global data_index

    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span]) # initial buffer content = first sliding window


    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words) # now we obtain a random list of context words
        for j in range(num_samples): # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word] # buffer[context_word] is a random context word

        # slide the window to the next position
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index]) # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1


    # end-of-for
    data_index = (data_index + len(data) - span) % len(data) # move data_index back by `span`
    return batch, labels

def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    assert os.path.exists(data_file)
    if os.path.exists(embedding_file_name):
        return

    with open(data_file, 'rb') as f:
        data, count, dictionary, reverse_dictionary, adjective = pickle.loads(f.read())

    vocabulary_size = VOCABULARY_SIZE

    batch_size = BATCH_SIZE      # Size of mini-batch for skip-gram model.
    embedding_size = embedding_dim # Dimension of the embedding vector.
    skip_window = SKIP_WINDOW       # How many words to consider left and right of the target word.
    num_samples = NUM_SKIPS         # How many times to reuse an input to generate a label.
    num_sampled = NUM_SAMPLED      # Sample size for negative examples.

    # Specification of test Sample:
    sample_size = 20       # Random sample of words to evaluate similarity.
    sample_window = 100    # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False) # Randomly pick a sample of size 16

    ## Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
                softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

            # Construct the Gradient Descent optimizer using a learning rate of 0.01.
            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm

            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()


            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()


    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)

        print('Initializing the model')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(data, count, dictionary, reverse_dictionary, adjective, batch_size, num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)

            average_loss += loss_val

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000

                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval() #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    top_k = 10  # Look for top-10 neighbours for words in sample set.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    #print(log_str)
                #print()

        final_embeddings = normalized_embeddings.eval()


    with open(embeddings_file_name, 'w') as f:
            adj_words = []
            for x in adjective:
                if x in dictionary:
                    adj_words.append(x)

            adj_num = len(adj_words)
            print(len(adj_words), embedding_dim, file = f)
            for i in range(adj_num):
                print(adj_words[i], end=' ', file=f)
                for j in range(embedding_dim):
                    print(final_embeddings[dictionary[adj_words[i]]][j], end = ' ', file=f)
                print(file=f)



def tokenize(raw_doc, adjective):
    document = nlp(raw_doc)
    for word in document:
        if word.pos_ == 'ADJ':
            adjective.add(word.text.lower())

    tokens = []
    for word in document:
        if word.text.isalpha():
            k = word.text.lower()
            tokens.append(k)
    return tokens

    


def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def process_data(input_data):
    if os.path.exists('data_file'):
        return 'data_file'
    tokens = []
    adjective = set()
    with zipfile.ZipFile(input_data) as f:
        for file in f.namelist():
            raw_doc = tf.compat.as_str(f.read(file)).strip()
            if raw_doc:
                tokens.extend(tokenize(raw_doc, adjective))

    data, count, dictionary, reverse_dictionary = build_dataset(tokens, VOCABULARY_SIZE)

    with open('data_file', 'wb') as f:
        f.write(pickle.dumps((data, count, dictionary, reverse_dictionary, adjective)))
    return 'data_file'



def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)

    return [x[0] for x in model.most_similar(input_adjective, topn=top_k)]



# data_file = process_data(input_dir)
# adjective_embeddings(data_file, embedding_file_name, num_steps, embedding_dim)
# model_file = 'adjective_embeddings.txt'
# input_adjective = 'bad'
# top_k = 100
# output = Compute_topk(model_file, input_adjective, top_k)
# print(output)








