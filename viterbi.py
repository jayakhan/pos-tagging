from numpy.core.fromnumeric import argmax
import numpy as np
import pandas as pd
import nltk
from pos import build_hmm_components

def viterbi(obs, pi, A, B):
    viterbi = np.ones((len(A), len(obs)), dtype='float32')
    viterbi_df = pd.DataFrame(viterbi)
    backpointer = np.ones((len(A), len(obs)), dtype='object')
    # Fill values in first column of viterbi matrix
    for s in range(0, len(A)):
        viterbi[s, 0] = pi[s] * B[s, 0]
        backpointer[s, 0] = 0
    # Fill values in other columns of viterbi matrix
    for t in range(1, len(obs)):
        for s in range(0, len(A)): 
            viterbi[s, t] = max(np.log(viterbi[m, t-1] * A[m, s] * B[s, t]) for m in range(0, len(A)))
            backpointer[s, t] = argmax(viterbi_df.iloc[:,t])
    list_states = []
    best_prob = max(viterbi_df.iloc[: , -1])
    best_path_pointer = argmax(viterbi_df.iloc[: , -1])
    # Back trace backpointer matrix
    for i in pd.DataFrame(backpointer).iloc[-1,:]:
        list_states.append(i)
    return list_states


def find_index(obs, obs_df):
    list_ints = []
    words_labels_list = [i for element in obs for i in element]
    words_tokens = sorted(set([vocab[0] for vocab in words_labels_list]))
    for words in words_tokens:
        if words in [columns for columns in obs_df.columns]: 
            list_ints.append(obs_df.columns.get_loc(words))
    return list_ints


if __name__ == "__main__":
    # Load corpus from nltk
    training_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
    trans_mat, obs_mat, obs_df, pi_mat = build_hmm_components(training_corpus)
    obs = nltk.corpus.brown.tagged_sents(tagset='universal')[10150:10153]
    obs_int = find_index(obs, obs_df)
    viterbi(obs_int, pi_mat, trans_mat, obs_mat)