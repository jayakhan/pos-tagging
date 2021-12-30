#submitted by:
#Surabhi Trivedi
#Jaya Khan

import nltk
import numpy as np
import pandas as pd

######making corpus
corpus = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]

corp = [ tup for sent in corpus for tup in sent ]
tags = {tag for word,tag in corp}
vocab = {word for word,tag in corp}

######creating obs
corpus_check = nltk.corpus.brown.tagged_sents(tagset='universal')
check = [10150, 10151, 10152]
 
# list of 3 sents on which we test the model
test_run = [corpus_check[i] for i in check]
 
# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]
 
# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]

######list of tags
s = set([pair[1] for pair in corp])
t = []
for i in s:
    t.append(i)

######emmision/ observation matrix
def word_given_tag(word, tag, train_bag = corp):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list) #total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list) + 1 # 1 is being added for smoothing
    return (count_w_given_tag, count_tag)


x = set([pair for pair in test_tagged_words])
y = []
for i in x:
    y.append(i)

e_matrix = np.zeros((len(y), len(t)), dtype='float32')

for i, t1 in enumerate(y):
    for j, tag in enumerate(t): 
        e_matrix[i, j] = word_given_tag(y[i], tag)[0]/word_given_tag(y[i], tag)[1]

######transition matrix
def t2_given_t1(t2, t1, train_bag = corp):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 1       #starting with 1 for smoothing
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))

######initial state distribution
pi = []
for i in range(len(corpus)):
    pi.append(corpus[i][0])

pi_list = [pair[1] for pair in pi] #fetch all tags

counter = []

for tag in tags:
    count = 0
    for i in pi_list:
        if i == tag:
            count += 1
            pass
        pass
    counter.append(count)

#pi_matrix = np.array(counter)
#pi_matrix = pi_matrix/10000
#pi_df=pd.Series(pi_matrix, index = list(tags))


######Viterbi
def viterbi(obs, pi, A, B):
    state = []
    #list of tags
    s = set([pair[1] for pair in corp])
    t = []
    for i in s:
        t.append(i)
    
    #emmision df
    x = set([pair for pair in obs])
    y = []
    for i in x:
        y.append(i)
    e_df = pd.DataFrame(A, columns = t, index=y)
    #print(e_df)

    #transition df
    tags_df = pd.DataFrame(B, columns = list(tags), index=list(tags))

    #pi df
    pi_matrix = np.array(pi)
    pi_matrix = pi_matrix/10000
    pi_df=pd.Series(pi_matrix, index = list(tags))
    print(pi_df)
    
    for key, word in enumerate(obs):
        #print(word)
        p = []
        for tag in t:
            #print(tag)
            if (key == 0 or obs[key-1] == "."):
                transition_p = pi_df[tag] #pick from counter
            else:
                transition_p = tags_df.loc[state[-1], tag] #pick from transition B

            #calculate state probability:
            emission_p = e_df.loc[word,tag]
            #print(emission_p)
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            #print(p)
        
        pmax = max(p)
        print(pmax)
        #state_max = '<unknown>'
        #print(pmax)
        # getting state for which probability is maximum
        if(pmax==0): 
            state_max = '<unknown>'
        else:
            state_max = t[p.index(pmax)] 
        state.append(state_max)
    return list(zip(test_tagged_words, state))

tagged_seq=viterbi(test_tagged_words, counter, e_matrix, tags_matrix)
print(tagged_seq)
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
 
accuracy = len(check)/len(tagged_seq)
print(accuracy)

