from babi_light_script.data_utils import load_stacked_data,print_data


S, testS, Q, testQ, is_Q, test_is_Q, A, testA, train, test, vocab_size, word_idx = load_stacked_data(dataset_dir='../../datasets/BABI/tasks_1-20_v1-2/')

print('text')
print(print_data(train[0]))

print('story')
print(S.shape)
print(S[0])

print('query')
print(Q.shape)
print(Q[0])

print('is query')
print(is_Q.shape)
print(is_Q[0])

print('answer')
print(A.shape)
print(A[0])
