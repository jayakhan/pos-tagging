# Part of Speech Tagging

### We have used the first 10k tagged sentences from the Brown corpus to generate the components of a part-of-speech hidden markov model: the transition matrix, observation matrix, and initial state distribution.

```
nltk.corpus.brown.tagged_sents(tagset=’universal’)[:10000]
```

### Implement a function viterbi() that takes arguments: obs - the observations [list of ints], pi - the initial state probabilities [list of floats], A - the state transition probability matrix [2D numpy array], B - the observation probability matrix [2D numpy array] and returns states - the inferred state sequence [list of ints]
