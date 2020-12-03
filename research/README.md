# Dokumentacja modułu visualisation:

1. Moduł służy wizualizacji poszczególnych segmentów dowolnego sygnału oraz udostępnia silnik filtrujący.

2. Moduł zawiera zmienne globalne `AFIB_THRESHOLD = 0.5`, która służy ustaleniu wartości progu AFIBU w obrębie segmentu oraz `WINDOW` określająca wielkośc okna segmentów w sekundach

3. Po utworzeniu obiektu klasy `my_visu = SegmentVisualization('datasets/mitbih_afdb/04015') ` otrzymujemy następujące parametry:

* .signal_path - ścieżka wczytanego sygnału
* .ann - aobiekt wfdb anotacji
* .ann_samples - numery próbek początków segmentów danego rytmu
* .fs - częstotliwośc próbkowania
* .record - obiekt klasy wfdb record
* .dataframe - dataframe zawierający kolumny:
    *index(numer próbki)*, *ECG1, ECG2 (dwa odprowadzenia - wartości surowego sygnału)*,
     *rhythm - zgodnie z referencją physionetu: (N, (AFIB, (brak rytmu oznaczone NOISE),
       *segment - numer segmentu zgodnie z .ann_samples**
    
*.keys = ['index', 'ratio', 'rhythm', 'chunk'] - klucze pozwalające na sortowanie obiektu klasy SegmentVisualization
* .window_size = 12 * self.fs - wielkość okna podana numerem próbek zawierających się w czasie = `WINDOW`
* .list_of_segments - **najważniejsza struktura** będąca listą słowników, gdzie każdy słownik to struktura z kluczem (.keys)

-> 'index' - numer `WINDOW`-sekundowego segmentu,

-> 'ratio' - stosunek ilości próbek z rytmem AFIB do pozostałcyh dla segmentu

-> 'rhythm' - AFIB lub NORMAL(dla pozostałych)

-> 'chunk' - dataframe o strukturze takiej jak .dataframe tylko że dla danego segmentu np. 12 sekundowego


## Zestaw narzędzi dla segmentu
### gdzie segment to chunk otrzymany z .list_of_df lub segment wyszukany z użyciem silnika search():
EXAMPLE:

`my_visu = SegmentVisualization('datasets/mitbih_afdb/04015')`

`segment = my_visu.list_of_segments[0]['chunk']['ECG1']` 

lub

`all_segments = my_visu.search(key='rhythm', value='AFIB')`

`segment = all_segments[0]['chunk']['ECG1']`


* `.show_segment(segment)` - pozwala na wyrysowanie surowego segmentu

* `.show_spectogram(segment)` - pozwala na wyrysowanie spektogramu segmentu

* `.show_scalogram(segment)` - pozwala na wyrysowanie skalogramu segmentu

* `.show_attractor(segment)` - pozwala na wyrysowanie rekonstrukcji attraktora zgodnie z opóźnieniami Tekkensa (1/3 długości QRS), rzutu na płaszczyznę prostopadłą do wektora (1,1,1) oraz gęstość dla rzutu

## Silnik wyszukiwania segmentów

`my_visu = SegmentVisualization('datasets/mitbih_afdb/04015')`

`.search()` pozwala wyszukać dowolny segment z określonymi parametrami:

przykładowo:

`my_visu.search(value='AFIB')` wyświetli indeksy segmentów o rytmie AFIB.

`my_visu.search(key='rhythm', value='AFIB', output='ratio')` wyświetli wartości ratio dla tych segmentów, dla których rytm=AFIB