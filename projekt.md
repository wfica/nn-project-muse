### 1. Czy kod robi to, co powinien? W szczególności, czy nie ma w nim haków nieopisanych w publikacji?

Kod wygląda jakby robił to co powinien.
- Nie znalazłem zaszytych stałych używanych do inicjalizacji generatora liczb losowych.
- Parametry modeli są inicjalizowane domyślną metodą z pytorcha.

Zastrzeżenia:
- W artykule piszą, że do przewidywań dyskryminatora stosują `smoothing coefficient s = 0.2`; natomiast w parematrach ```unsupervised.py``` wartość domyślna ```dis_smooth``` wynosi `0.1`.
- Przez to, że korzystają ze `smoothing coefficient`, mają inne funkcje straty, tj. wzory (3) i (4) z artykułu nie informują o tym, że te wartości sie skaluje.
- W paperze strata generatora uwzględnia dokładność dyskryminatora na zanurzeniach języka docelowego (tj. im gorsza dokładność dyskryminatora tym mniejsza wartość straty). Nie rozumiem dlaczego tak jest, bo ta dokładność nie zależy od parametrów generatora. Z drugiej strony to chyba niczego nie psuje, bo to nie zmienia gradientu.
- Oni cytują paper _Adversarial Training for Unsupervised Bilingual Lexicon Induction_, w którym wzór na stratę generatora nie uwzględnia dokładności dyskryminatora
na zanurzeniach języka docelowego.
- W kodzie strata generatora jest liczona tak jak w paperze (pomijając brak smoothing coefficient) - patrz `mapping_step()` w pliku `src/trainer.py`.

### 2. Czy Tabela 1 jest legitna (oni piszą, że Dinu et al [2015] wylicza dokładność dla 1500 testowych słów i robią tak samo, natomiast w pracy Dinu podaje szczegóły, że on wybiera ostrożnie te 1500 słów po 300 z binów różnej częstości)

Ich sposób walidacji jest następujący:
- ustalają k (równe 1, 5 lub 10),
- ustalają 1500 słów do przetłumaczenia,
- z docelowego słownika wybierają 200k najczęściej występujących słów,
- dla każdego słowa biora k najlepszych tłumaczeń zaproponowanych przez model, które należą w poprzednio wybranych 200k słów,
- liczą jaki procent tłumaczeń jest poprawnych.

### 3. Jak dokładnie działa kryterium z rozdziału 3.5?

Kryterium wyboru modelu przy adwersarzu jest liczone tak, że budujemy słownik używając zanurzeń 10000 (zahardkodowana stała wewnątrz funkcji `dist_mean_cosine` w pliku `src/evaluation/evaluator.py`) najczęstszych słów z obu języków i liczymy średnie podobieństwo tłumaczeń (cosine similarity).

### 4. Który etap uczenia korzysta z ilu tysięcy najczęstszych słów? Dyskryminator bierze 50k, Procrustes bierze ??? wypadających dobrze w CSLS. Ile słów jest w ostatecznym słowniku? Eksperymenty startują z 200k (rozdział 3.1)

- 200k najczęstszych słów jest wykorzystane do walidacji (opisanej wyżej w punkcie drugim).
- Dla trenowania dyskryminatora użyte jest 50k najczęstszych słów.
- Procrustes bierze tyle par ile uda się znaleźć, tzn. para (Wx, y) jest uwzględniona jako tłumaczenie jeśli y jest kNN Wx oraz Wx jest kNN y (używają k = 10, twierdzą, że próbowali dla 5, 10 i 50 i wyniki zawsze były porównywalne).
- Natomiast do walidacji bierzemy 10k najczęstszych słów, używamy CSLC aby znaleźć ich tłumacznie, a następnie liczymy średnią z odległości cosinusowej pomiędzy słowami a ich domniemanymi tłumaczeniami.
- Ostatecznie mamy macierz W i jej używamy do tłumacznia.
### 5. Czy high-quality dictionary użyty do zrobienia Tabelki 1 nie jest zbiasowany? Dlaczego on ma tylko 100k? Jak to się ma do tych wybieranych 1500?

Przeanalizowaliśmy słownik angielsko-niemiecki, nazywany ground-truth.

Z 1000 najpopularniejszych słów w języku angielskim, słownik posiada 951 (95.1%). Zabrakło:
```
a ago American among another as at be by Congress Democrat do establish go he herself himself I if in into it me Mr Mrs my no n't of oh ok on onto or PM relate Republican so than themselves those throughout to toward TV up upon us we
```

Z 3000 najpopularniejszych słów w języku angielskim, słownik posiada 2857 (95.2%). Zabrakło:
```
a ad African African-American ago ah AIDS AM American among another anymore appreciate Arab as Asian at be Bible British by Canadian Catholic CEO Chinese Christian Christmas conclude Congress congressional consist Democrat depending differently distinct DNA do elect e-mail emerge English entirely essentially establish European facility French furthermore German go God he heavily herself hi himself I ie if in incorporate Indian Internet into Iraqi Irish Islamic Israeli it Italian Japanese Jew Jewish Latin long-term manner me Mexican mm-hmm moreover Mr Mrs Ms Muslim my newly no nod notion n't obtain of oh ok Olympic on onto or ought ourselves Palestinian pant PC PM portion pursue rapidly regard regarding relate rely Republican Russian Senate shortly shrug so so-called Soviet Spanish suppose Supreme tablespoon tale than themselves those throughout to toward towards TV undergo United unless unlike up upon us vs we whenever whereas
```

Słownik zawiera 101931 par tłumaczeń, z czego niektóre angielskie słowa powtarzają się - unikalnych słów jest 74655.

### Raport z odpalenia EN-PL na 5 epok
```
Monolingual source word similarity score average: 0.63786
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

### Raport z odpalenia EN-ES na 5 epok
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

### 50 najpopularniejszych angielskich i polskich słów według wiki.en.vec i wiki.pl.vec
```
$ awk '{ print $1 }' data/wiki.en.vec | head -n 50
,
.
the
</s>
of
-
in
and
'
)
(
to
a
is
was
on
s
for
as
by
that
it
with
from
at
he
this
be
i
an
utc
his
not
–
are
or
talk
which
also
has
were
but
have
#
one
rd
new
first
page
```

```
$ awk '{ print $1 }' data/wiki.pl.vec | head -n 50
</s>
.
,
w
-
)
(
'
i
–
na
z
do
się
align
roku
a
jest
rd
nie
od
o
to
#
przez
left
po
score
jako
oraz
"
że
został
był
m
ur
za
r
jego
dla
center
tym
pod
cest
latach
dyskusja
są
jak
the
```

### BONUS: jakość tłumaczenia FB

**Yes/Nie**

![BONUS](fb-dict-quality-bonus.jpg "BONUS")