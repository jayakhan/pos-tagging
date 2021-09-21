import numpy as np
import pandas as pd


# Calculate Observation Distribution
def emission(word, lable, words_labels_list):
    pair_list = [pair for pair in words_labels_list if pair[1] == lable]
    tag_count = len(pair_list)  
    word_given_tag_list = [pair[0] for pair in pair_list if pair[0] == word]  
    word_given_tag_count = len(word_given_tag_list)    
    return word_given_tag_count, tag_count

# Calculate Transition Distribution
def transition(lable2, lable1, words_labels_list, vocab):
    tags = []
    tags = [pair[1] for pair in words_labels_list]
    t1_tags_list = [tag for tag in tags if tag == lable1]
    t1_tags_count = len(t1_tags_list)
    tags = [tags[index+1] for index in range(len(tags)-1) if tags[index] == lable1 and tags[index+1] == lable2]
    tags_len = len(tags)
    return tags_len, t1_tags_count

# Calculate Initial State Distribution
def state(word, label, words_labels_list):
    total_start_label = []
    start_label = []
    for i in range(len(words_labels_list)-1):
        if words_labels_list[i][0] == '.' and words_labels_list[i][1] == '.':
            if words_labels_list[i+1][1] != '':
                total_start_label.append(words_labels_list[i+1][1])
                if words_labels_list[i+1][0] == word:
                    start_label.append(label)
    return len(start_label), len(total_start_label)

# Create Transition Matrix
def create_transition_matrix(words_labels_list, words_tokens, labels_tokens, len_labels):
    transition_matrix = np.ones((len_labels, len_labels), dtype='float32')
    for i, t1 in enumerate(list(labels_tokens)):
        for j, t2 in enumerate(list(labels_tokens)): 
            transition_matrix[i, j] = np.log((transition(t2, t1, words_labels_list, words_tokens)[0] + 1)/(transition(t2, t1, words_labels_list, words_tokens)[1] + len(words_tokens)))
    #transition_df = pd.DataFrame(transition_matrix, columns = list(labels_tokens), index=list(labels_tokens)) 
    #print(transition_df.sum(axis=1))
    return transition_matrix

# Create Observation Matrix
def create_observation_matrix(words_labels_list, words_tokens, labels_tokens, len_labels, len_words):
    emission_matrix = np.ones((len_words, len_labels), dtype='float32')
    for i, t1 in enumerate(list(words_tokens)):
        for j, t2 in enumerate(list(labels_tokens)):
            emission_matrix[i, j] = np.log((emission(t1, t2, words_labels_list)[0] + 1)/(emission(t1, t2, words_labels_list)[1] + len_words))
    emission_df = pd.DataFrame(emission_matrix, columns = list(labels_tokens), index=list(words_tokens)) 
    return emission_matrix.transpose(), emission_df.transpose()

# Create Initial State Matrix
def create_state(words_labels_list, words_tokens, labels_tokens, len_labels, len_words):
    state_matrix = np.ones((len_words, len_labels), dtype='float32')
    for i, t1 in enumerate(list(words_tokens)):
        for j, t2 in enumerate(list(labels_tokens)):
            state_matrix[i, j] = np.log((state(t1, t2, words_labels_list)[0] + 1)/(state(t1, t2, words_labels_list)[1] + len_words))
    state_df = pd.DataFrame(state_matrix, columns = list(labels_tokens), index=list(words_tokens)) 
    return state_df.transpose().values.tolist()[0]

def build_hmm_components(corpus):
    words_labels_list = [i for element in corpus for i in element]
    words_tokens = set([vocab[0] for vocab in words_labels_list])
    labels_tokens = set([label[1] for label in words_labels_list])
    len_labels = len(labels_tokens)
    len_words = len(words_tokens)
    trans = create_transition_matrix(words_labels_list, words_tokens, labels_tokens, len_labels)
    obs, obs_df = create_observation_matrix(words_labels_list, words_tokens, labels_tokens, len_labels, len_words)
    initial = create_state(words_labels_list, words_tokens, labels_tokens, len_labels, len_words)
    return trans, obs, obs_df, initial
