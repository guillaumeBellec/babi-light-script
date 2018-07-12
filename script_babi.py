"""Simplistic implementation of memory networks.
Author: Guillaume Bellec
Date: 16th of December 2016
Version: Tested for python 3.4, and tensorflow 0.11 and 0.12

This is a simplistic code to implement a end-to-end memory networks [1].
This code is written for an educational purpose.
The dataset processing was copied from [2] and the algorithm code was inspired from [2].

There is a code available at [2] for implementation with multi 'hops' (mutli-layer in the context of memory networks).
Find this in the MemN2N object at the '_inference()' method.

[1] Original paper: http://papers.nips.cc/paper/5846-end-to-end-memory-networks
[2] Github: https://github.com/domluna/memn2n
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf

from data_utils import load_data

# If needed control the random seed
seed = rd.randint(20)
rd.seed(seed)
print('Seed set to {}'.format(seed))

# PARAMETERS
d_emb = 20  # Embedding size: matrix of size (n_vocab x d_emb to pre reduce the dimension of the data)

batch_size = 32  # Number of stories/question per batch
l_rate = 1e-2  # Learning rate
decay_every = 200  # Decay learning rate every k iterations
lr_decay = .5  # Decay rate of the learning rate
n_epoch = 300  # Number of training epochs
print_every = 10  # Report errors every k steps
gd_noise = .000  # STD of the noise added to each variable at every training epoch to get out of local minima

# Construct the data
trainS, testS, trainQ, testQ, trainA, testA, text_train, text_test, n_vocab = load_data(
    dataset_dir='../../datasets/BABI/tasks_1-20_v1-2/', dataset_suffix='en')
N, n_sentence, n_word = trainS.shape
n_out = n_vocab

# Tensorflow place holders
a_stories = tf.placeholder(dtype=tf.int32, shape=(None, n_sentence, n_word))
a_query = tf.placeholder(dtype=tf.int32, shape=(None, n_word))
a_target = tf.placeholder(dtype=tf.int32, shape=(None,))
learning_rate = tf.placeholder(dtype=tf.float32)

# Variables
word_bag = tf.clip_by_value(tf.Variable(initial_value=np.ones((1, 1, n_word, 1)), dtype=tf.float32), 0,
                            1)  # Embedding matrix, to reduce the dimension of one hot encoded words
emb = tf.Variable(initial_value=rd.randn(n_vocab, d_emb) * .1,
                  dtype=tf.float32)  # Embedding matrix, to reduce the dimension of one hot encoded words
w = tf.Variable(initial_value=rd.randn(d_emb, n_out) * .1, dtype=tf.float32)  # Output weight of the last layer
m_key = tf.Variable(initial_value=rd.randn(n_sentence, d_emb) * .1,
                    dtype=tf.float32)  # Absolute access key for each memory cell

# Process the story and query to get a larger binary one hot encoding
# (Note that for large dataset tensorflow provides directly tf.nn.embedding_lookup() to perform this and 1) in one line)
x_stories = tf.one_hot(a_stories, n_vocab)
x_query = tf.expand_dims(tf.one_hot(a_query, n_vocab), 1)

with tf.name_scope('InputProcessing'):
    # 1) Process the input with bag of word embedding
    x = tf.reduce_sum(word_bag * x_stories, axis=2)
    q = tf.reduce_sum(word_bag * x_query, axis=2)

    x = m_key + tf.einsum('bij,jk->bik', x, emb)
    q = tf.einsum('bij,jk->bik', q, emb)

with tf.name_scope('MemN2N'):
    # 2) Compute similarity between each sentence and the query, here similarity is just a dot product
    dotted = tf.reduce_sum(x * q, axis=2)

    # 3) Calculate with softmax of the most similar memory and feed it to the memory content to the next layer
    probs = tf.nn.softmax(dotted)
    probs = tf.expand_dims(probs, 2)
    m_read = tf.reduce_sum(probs * x, axis=1)

    # 4) Last softmax layer to output the right word
    a_z = tf.matmul(q[:, 0, :] + m_read, w)
    z = tf.nn.softmax(a_z)

# Compute error
Y = tf.one_hot(a_target, n_vocab)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=Y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Create tf ops to add noise
gd_step_list = []
for v in tf.trainable_variables():
    var_shp = tf.shape(v)
    step = tf.assign(v, v + tf.random_normal(var_shp, stddev=gd_noise))
    gd_step_list.append(step)

correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Store results
train_error_list = []
test_error_list = []
train_acc_list = []
test_acc_list = []

for k_epoch in range(n_epoch):

    # Compute errors for monitoring
    train_acc, train_err = sess.run([accuracy, cross_entropy],
                                    feed_dict={a_stories: trainS, a_query: trainQ, a_target: trainA[:, 0]})
    test_acc, test_err = sess.run([accuracy, cross_entropy],
                                  feed_dict={a_stories: testS, a_query: testQ, a_target: testA[:, 0]})

    if np.mod(k_epoch, print_every) == 0:
        print('Epoch {}'.format(k_epoch))
        print('Acc \t Train {:.3g} \t Test {:.3g}'.format(train_acc, test_acc))
        print('Err \t Train {:.3g} \t Test {:.3g}'.format(train_err, test_err))

    # Add errors to monitoring lists
    train_error_list.append(train_err)
    test_error_list.append(test_err)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    # Learning step
    batch_list = np.array_split(rd.permutation(N), N // batch_size)
    for batch in batch_list:
        sess.run(train_step, feed_dict={a_stories: trainS[batch], a_query: trainQ[batch], a_target: trainA[batch, 0],
                                        learning_rate: l_rate})
    sess.run(gd_step_list, feed_dict={a_stories: trainS[batch], a_query: trainQ[batch], a_target: trainA[batch, 0],
                                      learning_rate: l_rate})

    if np.mod(k_epoch, decay_every) == 0:
        l_rate *= lr_decay

fig, ax_list = plt.subplots(2)
ax_list[0].plot(np.arange(n_epoch), train_error_list, lw=2, color='blue')
ax_list[0].plot(np.arange(n_epoch), test_error_list, lw=2, color='green')
ax_list[1].plot(np.arange(n_epoch), train_acc_list, lw=2, color='blue')
ax_list[1].plot(np.arange(n_epoch), test_acc_list, lw=2, color='green')
ax_list[0].set_xlabel('Epoch')
ax_list[1].set_xlabel('Epoch')
ax_list[0].set_ylabel('Error')
ax_list[1].set_ylabel('Accuracy')

plt.legend()
plt.show()
