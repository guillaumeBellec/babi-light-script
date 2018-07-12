from __future__ import absolute_import

import os
import re
import numpy as np
from itertools import chain
from functools import reduce
from sklearn import cross_validation, metrics
import tensorflow as tf

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.
    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data

def print_data(text,answer_given=0,word_idx=None):
    story, query, answer = text
    print('\n - STORY')
    for sentence in story:
        print(' '.join(sentence))
    print('- QUERY')
    print(' '.join(query))

    print('- ANSWER')
    if word_idx is None:
        answer_given = 'N.A.'
    else:
        answer_given = word_idx[answer_given]
    print(answer[0] + ' (given {})'.format(answer_given))


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        #y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        #for a in answer:
        #    y[word_idx[a]] = 1
        aa = [word_idx[a] for a in answer]

        S.append(ss)
        Q.append(q)
        A.append(aa)

    return np.array(S), np.array(Q), np.array(A)


def vectorize_stacked_data(data, word_idx, sentence_size, corpus_size, query_repeat=1):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    IS_Q = []
    A = []
    for story, query, answer in data:
        ss = []
        is_q = []

        # pad to memory_size before reading the sentences
        lm = max(0, corpus_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
            is_q.append(False)

        # pad to memory_size before reading the sentences
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)
            is_q.append(False)

        # add query at the end
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq
        for i in range(query_repeat):
            ss.append(q)
            is_q.append(True)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:corpus_size][::-1]
        is_q = is_q[::-1][:corpus_size][::-1]

        aa = [word_idx[a] for a in answer]

        S.append(ss)
        Q.append(q)
        IS_Q.append(is_q)
        A.append(aa)

    return np.array(S), np.array(Q), np.array(IS_Q), np.array(A)

def load_data(task_id=1, memory_size=10, dataset_dir='../datasets/BABI/tasks_1-20_v1-2/', dataset_suffix='en'):

    print("Started Task:", task_id)

    # task data
    train, test = load_task(dataset_dir + dataset_suffix + '/', task_id)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position

    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    # train/validation/test sets
    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

    return S, testS, Q, testQ, A, testA, train, test, vocab_size

def load_stacked_data(task_id=1, memory_size=10, dataset_dir='../datasets/BABI/tasks_1-20_v1-2/', dataset_suffix='en', query_repeat=0):

    print("Started Task:", task_id)

    # task data
    train, test = load_task(dataset_dir + dataset_suffix + '/', task_id)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position

    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)
    print("Vobabulary size", vocab_size)

    # train/validation/test sets
    memory_size += query_repeat
    S, Q, is_Q, A = vectorize_stacked_data(train, word_idx, sentence_size, memory_size, query_repeat=query_repeat)
    testS, testQ, test_is_Q, testA = vectorize_stacked_data(test, word_idx, sentence_size, memory_size, query_repeat=query_repeat)

    return S, testS, Q, testQ, is_Q, test_is_Q, A, testA, train, test, vocab_size, word_idx

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

if __name__ == '__main__':
    S, testS, Q, testQ, is_Q, test_is_Q, A, testA, train, test, vocab_size, word_idx = load_stacked_data(
        dataset_dir='../datasets/BABI/tasks_1-20_v1-2/')

    print('\nText:')
    print(print_data(train[0]))

    print('\nStory:')
    print(S.shape)
    print(S[0])

    print('\nQuery:')
    print(Q.shape)
    print(Q[0])

    print('\nIs query (only relevant to identify the query when given sequentially with the text):')
    print(is_Q.shape)
    print(is_Q[0])

    print('\nAnswer:')
    print(A.shape)
    print(A[0])
