This paper presents a method for producing a bilingual dictionary without any paralel data between the source and the target languages. 

We have run the proposed method on two pairs of embeddings (English, Spanish) and (English, Polish). We found that in case of EN-ES the method works reasonably well, as stated in the paper. On the other hand, for the PL-EN task the model works terribly badly. We suppose that this significant difference might be caused by the fact that, compared to English or Spanish, there is an enormous number of inflection rules in Polish. For example, the word 'mam' could be derived from any of these: 'mama' [the mother], 'mamić' [to beguile] or 'mieć' [to have]. We think that there may not be any good linear relationship between Polish and English embeddings.

We tried to improve the model performance on PL-EN translation task by changing Polish embeddings. First, using [Morfeusz](http://sgjp.pl/morfeusz/morfeusz.html.en) we mapped words form Polish Wikipedia to their uninflected forms. If there were multiple uninflected forms of a word then we chose the most frequent one (e.g., for 'mam' we chose 'mama'). Then, we used this corpora to train new Polish embeddings. They proved to train the model way better. 

TODO: dodać jakieś konkretne procenty

To conclude, the paper is nicely written, the subject is interesting, the method gives good performance but there is still room for improvement.  