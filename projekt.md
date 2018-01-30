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

Z 1000 najpopularniejszych słów w języku angielskim, ground-truth English-German dictionary posiada 951 (95.1%). Zabrakło:
```
a ago American among another as at be by Congress Democrat do establish go he herself himself I if in into it me Mr Mrs my no n't of oh ok on onto or PM relate Republican so than themselves those throughout to toward TV up upon us we
```

Z 3000 najpopularniejszych słów w języku angielskim, ground-truth English-German dictionary posiada 2857 (95.2%). Zabrakło:
```
a ad African African-American ago ah AIDS AM American among another anymore appreciate Arab as Asian at be Bible British by Canadian Catholic CEO Chinese Christian Christmas conclude Congress congressional consist Democrat depending differently distinct DNA do elect e-mail emerge English entirely essentially establish European facility French furthermore German go God he heavily herself hi himself I ie if in incorporate Indian Internet into Iraqi Irish Islamic Israeli it Italian Japanese Jew Jewish Latin long-term manner me Mexican mm-hmm moreover Mr Mrs Ms Muslim my newly no nod notion n't obtain of oh ok Olympic on onto or ought ourselves Palestinian pant PC PM portion pursue rapidly regard regarding relate rely Republican Russian Senate shortly shrug so so-called Soviet Spanish suppose Supreme tablespoon tale than themselves those throughout to toward towards TV undergo United unless unlike up upon us vs we whenever whereas
```

