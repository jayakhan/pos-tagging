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
def create_state(corpus):
    pi = []
    counter = []
    corp = [ tup for sent in corpus for tup in sent ]
    tags = {tag for word,tag in corp}

    for i in range(len(corpus)):
        pi.append(corpus[i][0])

    pi_list = [pair[1] for pair in pi] #fetch all tags
    
    for tag in tags:
        count = 0
        for i in pi_list:
            if i == tag:
                count += 1
        counter.append(count)

    pi_matrix = np.array(counter)
    pi_matrix = pi_matrix/10000
    pi_df=pd.Series(pi_matrix, index = list(tags))
    return pi_df

def build_hmm_components(corpus):
    words_labels_list = [i for element in sorted(corpus) for i in element]
    words_tokens = sorted(set([vocab[0] for vocab in words_labels_list]))
    labels_tokens = sorted(set([label[1] for label in words_labels_list]))

    len_labels = len(labels_tokens)
    len_words = len(words_tokens)

    trans = create_transition_matrix(words_labels_list, words_tokens, labels_tokens, len_labels)
    obs, obs_df = create_observation_matrix(words_labels_list, words_tokens, labels_tokens, len_labels, len_words)
    initial_df = create_state(corpus)

    return trans, obs, obs_df, initial_df
