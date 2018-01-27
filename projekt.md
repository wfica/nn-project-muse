### 1. czy kod robi to, co powinien (w szczególności, czy nie ma w nim haków nieopisanych w publikacji),

Kod wygląda jakby robił to co powinien.
- Nie znalazłem zaszytych stałych używanych do inicjalizacji generatora liczb losowych
- Parametry modeli są inicjalizowane domyślną metodą z pytorcha

Zastrzeżenia:
- w artykule piszą, że do przewidywań dyskryminatora stosują 'smoothing coefficient s = 0.2'; natomiast w paramatrach ```unsupervised.py``` wartość domyślna ```dis_smooth``` wynosi 0.1
- przez to, że korzystają z 'smoothing coefficient', to mają inne funkcje straty, tj. wzory (3) i (4) z artykułu nie informują o tym, że te wartości sie skaluje.
- W paperze strata generatora uwzględnia dokładność dyskryminatora na zanurzeniach języka docelowego (tj. im gorsza dokładność dyskryminatora tym mniejsza wartość
straty). Nie rozumiem dlaczego tak jest, bo ta dokładność nie zależy od parametrów generatora. Z drugiej strony to chyba niczego nie psuje, bo to nie zmienia gradientu.
- Oni cytują paper _Adversarial Training for Unsupervised Bilingual Lexicon Induction_, w którym wzór na stratę generatora nie uwzględnia dokładności dyskryminatora
na zanurzeniach języka docelowego.
- W kodzie strata generatora jest liczona tak jak w paperze (pomijając brak smoothing coefficient) -- patrz `src/trainer.py` -> `mapping_step`.

### 2. czy Tabela 1 jest legitna (oni piszą, że Dinu et al [2015] wylicza accuracy dla 1500 testowych słów i robią tak samo, natomiast w pracy Dinu podaje szczegóły, że on wybiera ostrożnie te 1500 słów po 300 z binów różnej częstości),

Ich sposób walidacji jest następujący:
- ustalają k ( równe 1, 5 lub 10),
- ustalają 1500 słów do przetłumaczenia,
- z docelowego słownika wybierają 200k najczęściej występujących słów,
- dla każdego słowa biora k najlepszych tłumaczeń zaproponowanych przez model, które należą w poprzednio wybranych 200k słów
- liczą jaki procent tłumaczeń jest poprawnych,

### 3. jak dokładnie działa kryterium z rozdziału 3.5 (nie zrozumiałem jednoznacznie z tekstu),

Kryterium wyboru modelu przy adwersarzu jest liczone tak, że budujemy
słownik używając zanurzeń 10000 (zahardkodowana stała: `src/evaluation/evaluator.py` -> `dist_mean_cosine`)
najczęstszych słów z obu języków i liczymy średnie podobieństwo tłumaczeń (cosine similarity)

### 4. który etap uczenia korzysta z ilu tyś. najczęstszych słów? Dyskryminator bierze 50k, Procrustes bierze ??? wypadających dobrze w CSLS. Ile słów jest w ostatecznym słowniku? Eksperymenty startują z 200k (rozdział 3.1),

- 200k najczęstszych słów jest wykorzystane do walidacji (opisanej w 2.).
- Dla trenowania dyskryminatora użyte jest 50k najczęstszych słów.
- Procrustes bierze tyle par ile uda się znaleźć, tzn para (Wx, y) jest uwzględniona jako tłumaczenie jeśli y jest kNN Wx oraz Wx jest kNN y. (używają k = 10, twierdzą, że próbowali dla 5, 10 i 50 i wyniki zawsze były porównywalne)
- Natomiast do walidacji bierzemy 10k najczęstszych słów, używamy CSLC aby znaleźć ich tłumacznie, a następnie liczymy średnią z odległości cosinusowej pomiędzy słowami a ich domniemanymi tłumaczeniami.
- Ostatecznie mamy macierz W i jej używamy do tłumacznia.
### 5. czy high-quality dictionary użyty do zrobienia Tabelki 1 nie jest zbiasowany? Dlaczego on ma tylko 100k? Jak to się ma do tych wybieranych 1500?
