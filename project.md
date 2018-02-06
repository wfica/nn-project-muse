# Neural Network class project: ICLR 2018 Reproducibility Challenge
The aim of this report is to discuss results presented in ['Word translation without parallel data'](https://arxiv.org/pdf/1710.04087.pdf) by A. Conneau, G. Lample, M. Ranzato, L. Denoyer and H. Jegou. This notice is a part of a neural network class project. The report consists of two parts. First, we answer question raised by our teacher. Then, we give a summary of our efforts to reproduce results claimed by the authors.

### Thank you, Google!
We are thankful that our University faculty was granted Google Cloud Platform Credits, because of that we were able to reproduce the results from the paper.

## Part 1: The Teacher's Questions

### 1. Does the code does what it should? In particular, are there any hacks not mentioned in the paper?
The code looks as if it does what is should.
- We have not found any hidden constants that might have been used as seeds to random number generator.
- The model's parameters are initialized by some default method from pythorch.

Reservations:
- In the paper is it written that a ```smoothing coefficient s = 0.2``` is applied for discriminator's predictions, while in the parameters of ```unsupervised.py``` the default value of ```dis_smooth``` is equal to `0.1`.
- In the paper, the generator's loss takes account of discriminator's accuracy on embeddings of the target language (i.e. the worst the accuracy, the lower the loss). We do not understand why it is so. On the one hand, the accuracy does not depend on generator's parameters, on the other such a loss does not spoil anything, since it does not affect the gradient.
- Moreover, the authors cite paper  _Adversarial Training for Unsupervised Bilingual Lexicon Induction_, in which the generator's loss does not count the discriminator's accuracy on embeddings of the target language.

### 2. What is the validation method? How many words are used for validation?
The validation method is as follows:
- k is chosen from {1, 5, 10},
- Let W consist of 1500 words that are arbitrarily chosen to be translated,
- Let D consist of 200k most frequent words from the target language,
- For each word of W: k best translations belonging to D are chosen,
- The percentage of correct translations is calculated.

### 3. How does the criterion from chapter 3.5 work?
This criterion tells how to choose the best model for the adversarial part. The criterion works as follows: using the embeddings, a dictionary of 10k most frequent words from both languages is created. Then the average cosine similarity of those translations is computed.

### 4. How many most frequent words are used during each stage of the algorithm?

- 200k most frequent words are used for the validation (see 1.2.)
- 50k most frequent words are used for training the discriminator
- The Procrustes step considers as many pairs as the algorithm produces. That is, a pair (Wx, y) is considered to be a translation if y is a kNN of Wx and Wx is a kNN of y. (The constant k = 10 is used for kNN. The authors claim that they did experiments with k = 5, 10 or 50 and the results were comparable.)
- In the validation step 10k most frequent words are considered. CSLC is used to find translations of those, and then the average cosine similarity between the words and their translations is computed.
- In the end, the result of the algorithm is a matrix W, and it is used for translating.

### 5. Are the high-quality dictionaries truly unbiased?

We looked closely at, in the paper called ground-truth, English-German dictionary.

The dictionary contains 951 (95.1%) words out of 1000 most frequent English words. The following are missing:
```
a ago American among another as at be by Congress Democrat do establish go he herself himself I if in into it me Mr Mrs my no n't of oh ok on onto or PM relate Republican so than themselves those throughout to toward TV up upon us we
```
The dictionary contains 2857 (95.2%) words out of 3000 most frequent English words. The following are missing:
```
a ad African African-American ago ah AIDS AM American among another anymore appreciate Arab as Asian at be Bible British by Canadian Catholic CEO Chinese Christian Christmas conclude Congress congressional consist Democrat depending differently distinct DNA do elect e-mail emerge English entirely essentially establish European facility French furthermore German go God he heavily herself hi himself I ie if in incorporate Indian Internet into Iraqi Irish Islamic Israeli it Italian Japanese Jew Jewish Latin long-term manner me Mexican mm-hmm moreover Mr Mrs Ms Muslim my newly no nod notion n't obtain of oh ok Olympic on onto or ought ourselves Palestinian pant PC PM portion pursue rapidly regard regarding relate rely Republican Russian Senate shortly shrug so so-called Soviet Spanish suppose Supreme tablespoon tale than themselves those throughout to toward towards TV undergo United unless unlike up upon us vs we whenever whereas
```

The dictionary consists of 101931 pairs of translations, some of which are repeated. There are 74655 unique words.
It seems that there would be no benefit from enlarging the dictionary as it is already of a low quality.


## Part 2: The Reproduction of the Results

### The results of running 5 epochs for EN-PL
```
INFO - 01/26/18 16:33:16 - 0:25:15 - ====================================================================
INFO - 01/26/18 16:33:16 - 0:25:15 -                        Dataset      Found     Not found          Rho
INFO - 01/26/18 16:33:16 - 0:25:15 - ====================================================================
INFO - 01/26/18 16:33:16 - 0:25:15 -                       EN_MC-30         30             0       0.8535
INFO - 01/26/18 16:33:16 - 0:25:16 -                   EN_MTurk-771        771             0       0.6305
INFO - 01/26/18 16:33:16 - 0:25:16 -                      EN_YP-130        130             0       0.5013
INFO - 01/26/18 16:33:16 - 0:25:16 -                 EN_RW-STANFORD       1323           711       0.5249
INFO - 01/26/18 16:33:16 - 0:25:16 -                   EN_MTurk-287        286             1       0.6321
INFO - 01/26/18 16:33:16 - 0:25:16 -                  EN_SIMLEX-999        998             1       0.3629
INFO - 01/26/18 16:33:16 - 0:25:16 -                  EN_WS-353-ALL        353             0       0.7274
INFO - 01/26/18 16:33:16 - 0:25:16 -                   EN_MEN-TR-3k       3000             0       0.7325
INFO - 01/26/18 16:33:16 - 0:25:16 -                  EN_WS-353-SIM        203             0       0.7642
INFO - 01/26/18 16:33:16 - 0:25:16 -                       EN_RG-65         65             0       0.7923
INFO - 01/26/18 16:33:16 - 0:25:16 -                    EN_VERB-143        144             0       0.4042
INFO - 01/26/18 16:33:16 - 0:25:16 -                  EN_WS-353-REL        252             0       0.6595
INFO - 01/26/18 16:33:16 - 0:25:16 -                   EN_SEMEVAL17        379             9       0.7067
INFO - 01/26/18 16:33:16 - 0:25:16 - ====================================================================
INFO - 01/26/18 16:33:16 - 0:25:16 - Monolingual source word similarity score average: 0.63786
INFO - 01/26/18 16:33:17 - 0:25:16 - Found 2745 pairs of words in the dictionary (1500 unique). 0 other pairs contained at least one unknown word (0 in lang1, 0 in lang2)
INFO - 01/26/18 16:33:17 - 0:25:17 - 1500 source words - nn - Precision at k = 1: 0.000000
INFO - 01/26/18 16:33:17 - 0:25:17 - 1500 source words - nn - Precision at k = 5: 0.000000
INFO - 01/26/18 16:33:18 - 0:25:17 - 1500 source words - nn - Precision at k = 10: 0.000000
INFO - 01/26/18 16:33:18 - 0:25:17 - Found 2745 pairs of words in the dictionary (1500 unique). 0 other pairs contained at least one unknown word (0 in lang1, 0 in lang2)
INFO - 01/26/18 16:33:45 - 0:25:44 - 1500 source words - csls_knn_10 - Precision at k = 1: 0.000000
INFO - 01/26/18 16:33:45 - 0:25:44 - 1500 source words - csls_knn_10 - Precision at k = 5: 0.000000
INFO - 01/26/18 16:33:45 - 0:25:45 - 1500 source words - csls_knn_10 - Precision at k = 10: 0.000000
INFO - 01/26/18 16:34:06 - 0:26:06 - Building the train dictionary ...
INFO - 01/26/18 16:34:06 - 0:26:06 - New train dictionary of 2563 pairs.
INFO - 01/26/18 16:34:06 - 0:26:06 - Mean cosine (nn method, S2T build, 10000 max size): 0.50096
INFO - 01/26/18 16:35:30 - 0:27:29 - Building the train dictionary ...
INFO - 01/26/18 16:35:30 - 0:27:29 - New train dictionary of 1911 pairs.
INFO - 01/26/18 16:35:30 - 0:27:29 - Mean cosine (csls_knn_10 method, S2T build, 10000 max size): 0.47842
INFO - 01/26/18 16:35:34 - 0:27:34 - Discriminator source / target predictions: 0.88776 / 0.16134
INFO - 01/26/18 16:35:34 - 0:27:34 - Discriminator source / target / global accuracy: 0.99906 / 0.95717 / 0.97812
```

### The results of running 5 epochs for EN-PL

```
INFO - 01/30/18 00:48:52 - 0:26:11 - ====================================================================
INFO - 01/30/18 00:48:52 - 0:26:11 -                        Dataset      Found     Not found          Rho
INFO - 01/30/18 00:48:52 - 0:26:11 - ====================================================================
INFO - 01/30/18 00:48:52 - 0:26:11 -                       EN_MC-30         30             0       0.8612
INFO - 01/30/18 00:48:52 - 0:26:11 -                   EN_MTurk-771        771             0       0.6524
INFO - 01/30/18 00:48:52 - 0:26:11 -                      EN_YP-130        130             0       0.4754
INFO - 01/30/18 00:48:52 - 0:26:11 -                 EN_RW-STANFORD       1323           711       0.5011
INFO - 01/30/18 00:48:52 - 0:26:11 -                   EN_MTurk-287        286             1       0.6530
INFO - 01/30/18 00:48:52 - 0:26:11 -                  EN_SIMLEX-999        998             1       0.3705
INFO - 01/30/18 00:48:52 - 0:26:11 -                  EN_WS-353-ALL        353             0       0.7191
INFO - 01/30/18 00:48:52 - 0:26:11 -                   EN_MEN-TR-3k       3000             0       0.7547
INFO - 01/30/18 00:48:52 - 0:26:11 -                  EN_WS-353-SIM        203             0       0.7747
INFO - 01/30/18 00:48:52 - 0:26:11 -                       EN_RG-65         65             0       0.7938
INFO - 01/30/18 00:48:52 - 0:26:11 -                    EN_VERB-143        144             0       0.3577
INFO - 01/30/18 00:48:52 - 0:26:11 -                  EN_WS-353-REL        252             0       0.6588
INFO - 01/30/18 00:48:52 - 0:26:11 -                   EN_SEMEVAL17        379             9       0.7060
INFO - 01/30/18 00:48:52 - 0:26:11 - ====================================================================
INFO - 01/30/18 00:48:52 - 0:26:11 - ====================================================================
INFO - 01/30/18 00:48:52 - 0:26:11 -                        Dataset      Found     Not found          Rho
INFO - 01/30/18 00:48:52 - 0:26:11 - ====================================================================
INFO - 01/30/18 00:48:52 - 0:26:11 -                   ES_SEMEVAL17        368            10       0.7392
INFO - 01/30/18 00:48:52 - 0:26:11 -                       ES_MC-30         27             3       0.7475
INFO - 01/30/18 00:48:52 - 0:26:11 -                       ES_RG-65         65             0       0.8794
INFO - 01/30/18 00:48:52 - 0:26:11 -                      ES_WS-353        319            33       0.6126
INFO - 01/30/18 00:48:52 - 0:26:11 - ====================================================================
INFO - 01/30/18 00:48:52 - 0:26:11 - Monolingual source word similarity score average: 0.63680
INFO - 01/30/18 00:48:52 - 0:26:11 - Monolingual target word similarity score average: 0.74466
INFO - 01/30/18 00:48:52 - 0:26:11 - Monolingual word similarity score average: 0.69073
INFO - 01/30/18 00:48:53 - 0:26:12 - ====================================================================
INFO - 01/30/18 00:48:53 - 0:26:12 -                        Dataset      Found     Not found          Rho
INFO - 01/30/18 00:48:53 - 0:26:12 - ====================================================================
INFO - 01/30/18 00:48:53 - 0:26:12 -                EN_ES_SEMEVAL17        729            19       0.5767
INFO - 01/30/18 00:48:53 - 0:26:12 - ====================================================================
INFO - 01/30/18 00:48:53 - 0:26:12 - Cross-lingual word similarity score average: 0.57671
INFO - 01/30/18 00:48:53 - 0:26:12 - Found 2975 pairs of words in the dictionary (1500 unique). 0 other pairs contained at least one unknown word (0 in lang1, 0 in lang2)
INFO - 01/30/18 00:48:53 - 0:26:12 - 1500 source words - nn - Precision at k = 1: 20.666667
INFO - 01/30/18 00:48:53 - 0:26:12 - 1500 source words - nn - Precision at k = 5: 37.200000
INFO - 01/30/18 00:48:54 - 0:26:13 - 1500 source words - nn - Precision at k = 10: 43.866667
INFO - 01/30/18 00:48:54 - 0:26:13 - Found 2975 pairs of words in the dictionary (1500 unique). 0 other pairs contained at least one unknown word (0 in lang1, 0 in lang2)
INFO - 01/30/18 00:49:22 - 0:26:41 - 1500 source words - csls_knn_10 - Precision at k = 1: 31.800000
INFO - 01/30/18 00:49:22 - 0:26:41 - 1500 source words - csls_knn_10 - Precision at k = 5: 49.000000
INFO - 01/30/18 00:49:23 - 0:26:42 - 1500 source words - csls_knn_10 - Precision at k = 10: 56.466667
INFO - 01/30/18 00:49:43 - 0:27:02 - Building the train dictionary ...
INFO - 01/30/18 00:49:43 - 0:27:02 - New train dictionary of 2339 pairs.
INFO - 01/30/18 00:49:43 - 0:27:02 - Mean cosine (nn method, S2T build, 10000 max size): 0.56065
INFO - 01/30/18 00:51:09 - 0:28:28 - Building the train dictionary ...
INFO - 01/30/18 00:51:09 - 0:28:28 - New train dictionary of 3221 pairs.
INFO - 01/30/18 00:51:09 - 0:28:28 - Mean cosine (csls_knn_10 method, S2T build, 10000 max size): 0.53287
INFO - 01/30/18 00:51:13 - 0:28:32 - Discriminator source / target predictions: 0.89890 / 0.32330
INFO - 01/30/18 00:51:13 - 0:28:32 - Discriminator source / target / global accuracy: 0.99720 / 0.77079 / 0.88399
```

### Summary
Clearly, after 5 epoch for `k = 1, 5, 10` for the EN-PL task the model works terribly badly, whreas for the EN-ES task the model works as stated in the paper.
This significant difference might be caused by the fact that, compared to English or Spanish, there is an unnormous number of inflection rules in Polish.