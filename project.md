# Project: ICLR 2018 Reproducibility Challenge
### The final project for Neural Networks and Deep Learning 2017 class at the University of Wrocław
The aim of this report is to discuss results presented in ['Word translation without parallel data'](https://arxiv.org/pdf/1710.04087.pdf) by A. Conneau, G. Lample, M. Ranzato, L. Denoyer and H. Jegou. This notice is a part of a neural network class project. The report consists of two parts. First, we answer question raised by our teacher. Then, we give a summary of our efforts to reproduce results claimed by the authors.

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
It seems that there would be no benefit from enlarging the dictionary as it is already of not so high quality.


## Part 2: The reproduction of the results

We focused on word translation retrieval using CSLS KNN. To start with, we ran 5 epochs of the project (unsupervised version) for EN to ES & EN to PL embeddings mapping.

EN to PL word translation accuracy:
- @1 `0.0%`
- @5 `0.0%`
- @10 `0.0%`

where @k (like in the paper) means that we looked for correct translation among k nearest neighbours of source word multiplied by matrix `W`.

EN to ES word translation accuracy:
- @1 `31.8%`
- @5 `49.0%`
- @10 `56.5%`

Clearly, after 5 epochs for the EN-PL task the model works terribly badly, whereas for the EN-ES task the model works as stated in the paper.


This significant difference might be caused by the fact that, compared to English or Spanish, there is an enormous number of inflection rules in Polish. For example, the word 'mam' could be derived from any of these: 'mama' [the mother], 'mamić' [to beguile] or 'mieć' [to have]. Another example: the word 'dwa' [two] appears in Polish in a lot of forms: 'dwaj', 'dwie', 'dwóch', 'dwu', 'dwóm', 'dwom', 'dwoma', 'dwiema', 'dwójce', 'dwojgiem', 'dwojga', 'dwojgu', 'dwójka', etc. -- each means "two" but in different contexts. We suppose that there may not be any good linear relationship between Polish and English embeddings.

---

We tried to improve the model performance on EN-PL translation task by changing Polish embeddings. First, using [Morfeusz](http://sgjp.pl/morfeusz/morfeusz.html.en) we mapped words form Polish Wikipedia to their uninflected forms. If there were multiple uninflected forms of a word then we chose the most frequent one (e.g., for 'mam' we chose 'mieć'). Then, we used this corpora to train new Polish embeddings. They proved to train the model way better and we could gain acceptable results instead of just 0%.

Now we ran EN-ES and EN-PL tasks for huge amount of epochs but they automatically stoped near epoch 25, when the improvement was too small.

Results for EN-PL word translation accuracy:
- @1 `43.3%`
- @5 `60.7%`
- @10 `66.5%`

Results for EN-ES word translation accuracy:
- @1 `75.7%`
- @5 `87.8%`
- @10 `90.3%`

Results for EN-ES are comparable to those presented in the paper.

---

We also changed the source code a little in order to get those translations and not only percentage accuracy.

Here are some samples for EN-PL, you see 10 nearest neighbours starting with the most probable (the closest):
```
fountain: fontanna grota nimfeum wodotrysk kandelabr skwer kapliczny zdrojowy fontanka altana
vernon: michalin puchała augustynek tworek roszczynialski miegoń pietraszek kobiela poreda krzymiński
dust: szron pył pyłowy mgiełka dziura obłok drobinka kropelka brud okruch
moments: przebłysk przedsmak dwudziestominutowy emocjonować trzyminutowy dwuminutowy kilkuminutowy dziesięciominutowy pamiętny jednominutowy
discography: dyskografia bublé tangled foxy discografia koяn p-funk ub40 jarboe audioslave
barrier: bariera barierowy przegrodzić ochronny newralgiczny wodochronny odgrodzić zakotwić chronić igielny
bee: szerszeń chrabąszcz szarak świerszczak ćma skakun cykada szarańczak szpak trzmiel
gathering: zbierać urządzać zebranie gromadzić zebrać doroczny dorocznie odbywać dożynek obywać
researcher: naukowiec ekspertka badaczka amerykanista biolog badanie iranista biocybernetyka radiobiologia neurobiolog
lover: kochanek ukochać kochanka zakochać pokochać wejrzeć nieznajomy rozkochać pocałunek ukochany
rap: hip-hopowy trip-hop hip-hop hopowy rap gangsta hop doggystyle rapcore crunk
casting: montaż casting charakteryzacja ekranowy superprodukcja podkładać gorący castingowy obsada spellinga
morrison: roniewicz kwietniewski rynkowski solak promiński piętowski sipiński koralewski konwiński mazolewski
honored: uhonorować honorować wręczać nagrodzić zasłużyć honorowy patronować przyznawać cześć uczcić
chest: piersiowy dłoń klatka szyja głowa przedramię brzuch kark plecy pierś
strait: cieśnina półwysep morze zatoka płw kerczeński atlantyk bering kaletański mierzeja
indo: austronezyjski austroazjatycki indoirański indoeuropejski indoaryjski nieindoeuropejski atapaskański wschodnioazjatycki mon-khmer zachodniogermański
announcement: zapowiedzieć zapowiedź oficjalny poinformować zaanonsować ogłosić intencyjny anonsować sierpniowy oświadczyć
harmony: harmonia harmonijny dysharmonia harmonista harmonijnie harmonica instrumentalizacja współbrzmienie harmonies armonia
accuracy: niedokładność dokładność precyzja precyzyjność niedokładny korygować dokładny trafność poprawność porównywalność
```

And here are some sample translations for EN-ES:
```
thursday: sábado miércoles viernes martes jueves lunes mañana madrugada día víspera
operators: operadores operadoras operador demostradores prestadores contadores suministradores operados cooperadores operadas
chilean: chileno chileno» peruano chilena chilenos chilenas chile, arequipeño chilena» arequipeña
departed: arribando embarcaría partió partieron regresarían regresaba regresaron embarcaron arribó embarcando
dirty: sucia sucios sucias sucio limpia poquianchis trapos harapos maldita morocha
dimension: dimensión dimensionalidad dimensionado dimensionar dimensionales dimensional dimensionamiento tetradimensional unidimensional espaciotemporal
cleared: despejando despejar despeja despejaron despejó adelantadas quedando allanando aclaradas adelantando
simpson: simpson mcbain stewart matthews reubens partridge reid mccauley buckley willis
thriller: thriller suspense policiaca suspenso policíaca policíaco policiaco littín fotonovela ambientada
richardson: crowell willis richardson hollingsworth woodard stowell howells mcwilliams mcgillis milligan
grants: otorgaran becas concedan concedidos otorgara otorgamiento conceden otorgadas otorguen otorgados
tested: probando probados probadas probaron pruebas probado prueba probada probó probar
paying: pagar pagando pagada pago remuneración paga adeudaba retribución pagaran pagase
spy: espía espías espionaje contraespionaje secreta secreto desertor secretos encubierto exagente
silly: absurda tonta tontería estúpida ridícula ridículo descabellada absurdas chocante pataleta
novelists: novelistas ensayistas cuentistas escritoras escritores autobiógrafos literarios dramaturgos ilustradores poetas
warrior: guerrera espadachín guerreros samurái valeroso guerreras gladiador samurai paladín guerrero
cornell: cornell stanford tufts yale harvard northwestern depaul duke ucla wesleyan
probability: probabilidad probabilidades probabilístico probabilísticas probabilística probabilísticos probabilista estimador varianza estocástico
subway: metro monorraíl monorail –avenida premetro metrotrén square–calle metrotranvía shinkansen autobus
```

To conclude, the paper is nicely written, the subject is interesting, the method gives good performance but there is still room for improvement.

## Acknowledgments
The authors thank Google for GCE Credits awarded through Google Cloud Platform Education Grants to the Neural Networks and Deep Learning course and to this project.