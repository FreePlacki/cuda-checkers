#show link: underline

= CUDA Warcaby

== Zasady gry

Przyjęte zasady to #link("https://en.wikipedia.org/wiki/Checkers")[warcaby amerykańskie].

- Plansza 8x8.
- Pionki ruszają się o jeden tylko do przodu.
- Króle ruszają się o jeden do przodu i do tyłu.
- Przymus bicia -- można wybrać dowolne.
- Remis po 80 ruchach bez bicia.

== Struktury danych

=== Plansza
Plansza reprezentowana jest na 12 bajtach:

```c
typedef struct {
    uint32_t white;
    uint32_t black;
    uint32_t kings;
} Board;
```

Plansza warcabów ma wymiary 8x8 ale tylko połowa z tych pól jest grywalna zatem
do zareprezentowania pozycji pionka potrzeba nam 32 bitów.

Przykładowo aby sprawdzić czy na pozycji o indeksie `x` znajduje się biały król:

`(1 << x) & board.white & board.king != 0`.

Na pierwszy rzut oka intuicyjnym indeksowaniem wydawało by się coś w stylu:
```
   28  29  30  31 
 24  25  26  27
   20  21  22  23
 16  17  18  19
   12  13  14  15
 08  09  10  11  
   04  05  06  07 
 00  01  02  03
```

Zauważmy, że każdy pionek może poruszyć się o $plus.minus 4$ (oprócz odpowiednio górnego i dolnego wiersza).
Ale dodatkowo te w parzystych wierszach (numerowane od 0 od dołu) mogą ruszać się $plus.minus 3$
a te w nieparzystych o $plus.minus 5$. Powoduje to znaczne komplikacje algorytmu generowania dozwolonych
ruchów (a co za tym idzie gorsza wydajność).

Okazuje się, że istnieje #link("https://3dkingdoms.com/checkers/bitboards.htm#A1")[lepsze indeksowanie]:

```
   11  05  31  25 
 10  04  30  24 
   03  29  23  17 
 02  28  22  16 
   27  21  15  09 
 26  20  14  08 
   19  13  07  01 
 18  12  06  00
```

Zauważmy, że tutaj każdy pionek (bez względu na parzystość wiersza) może ruszać się o
$plus.minus 1$, $plus.minus 7$

=== Ruch

Okazuje się, że ruch możemy reprezentować na 4 bajtach:

```c
typedef uint32_t Move;
```

Bit jest `1` wtw. dany indeks jest:
- pozycją początkową lub końcową poruszającego się pionka
- pozycją zbitego pionka przeciwnika

Aby wykonać ruch wystarczy zxorować planszę z ruchem.
Mamy problem gdy pozycja początkowa i końcowa są równe, ale możemy sprawdzić
czy ilość pionków koloru poruszającego się pozostała stała po ruchu.

== Algorytm Monte Carlo

Przyjmijmy bso. że gramy białymi a nasz przeciwnik czarnymi pionkami.
Jesteśmy w pewnym momencie rozgrywki i musimy wybrać jeden z $N$ dostępnych ruchów.

=== "Flat" Monte Carlo

Najprostszym pomysłem jest rozegrać dla każdego z możliwych posunięć $k$ losowych
partii a następnie wybrać ruch maksymalizujący ilość wygranych minus ilość przegranych:

$$$
"argmax"_(i = 1, ..., N) sum_(j = 1)^k s_j, & quad 
s_j = cases(
  +1 "jeśli" j"-ta gra zakończyła się wygraną białego",
  -1 "jeśli" j"-ta gra zakończyła się przegraną białego",
  " 0  jeśli" j"-ta gra zakończyła się remisem",
)
$$$

Takie podejście można w naturalny sposób napisać dla GPU -- każdy wątek gra do
końca na swojej planszy i raportuje wynik (wygrana/przegrana/remis).
Wywołujemy kernel dla każdego możliwego ruchu początkowego i po wywołaniu każdego
zliczamy jego wynik.

=== Monte Carlo Tree Search

Powyższa metoda ma znaczącą wadę -- spędza tyle samo czasu na każdym ruchu.
Po wykonaniu pewnej liczby symulacji często wiemy już które ruchy wydają się
być lepsze od innych i to na nich powinniśmy się skupiać (alokować większą ilość symulacji).

#link("https://en.wikipedia.org/wiki/Monte_Carlo_tree_search")[Monte Carlo Tree Search] można podzielić na 4 etapy:
1. _Selection_: zaczynamy z pozycji początkowej (korzenia) i wybieramy kolejne plansze (dzieci)
biorąc pod uwagę rezulataty gier w ich poddrzewach:

Najczęściej używana jest w selekcji wartość wyrażenia:
$$$
w_i / n_i + c sqrt(ln(N_i)/n_i)
$$$

gdzie $w_i/n_i$ - wygrane w stosunku do ilości rozgrywek w poddrzewie (faworyzuje lepsze ruchy),
$c sqrt(ln(N_i) / n_i)$ - faworyzuje mniej eksplorowane poddrzewa ($N_i$ to liczba symulacji w rodzicu).

2. _Expansion_: Jeśli wybrany wierzchołek nie jest zakończoną grą, robimy losowy ruch,
tworząc nowy wierzchołek.

3. _Simulation_: Dla nowego wierzchołka wykonujemy jedną symulację (do zakończenia gry).

4. _Backpropagation_: Znając rezultat symulacji, aktualizujemy dane w wierzchołkach --
liczba wygranych i liczba rozgrywek. W przypadku remisu zwiększamy liczbę wygranych o 0.5
(ewentualnie można przeskalować wszystkie wartości x$2$ aby nie używać `float`ów).

#image("MCTS-steps.svg")

Implementacja:
1. CPU wybiera ileś liści i _expanduje_ każdy wybrany
2. GPU dla każdego z wybranych liści robi ustaloną liczbę symulacji (np. 32)
3. CPU robi _backpropagation_
4. Powtarzamy aż skończy się czas na zrobienie ruchu

Aby CPU podczas jednej tury nie wybierał ciągle tego zamego liścia, możemy
dla każdego wierzchołka na ścieżce do liścia dodać pewną wartość (karę) do
ilości jego wizyt, aby miał mniejszą szansę na zostanie wybranym. Karę zdejmujemy
podczas kroku 3 (_backpropagation_).

Dodatkowo, okazuje się (przy sensownie wybranej liczbie liści które wybiera CPU), że
czas spędzony przez CPU jest porównywalny z czasem spędzonym przez GPU, zatem
możemy te części algorytmu wykonywać równolegle (pracując na dwóch bufforach,
które są ze sobą zamieniane).

Przy testach najlepszą liczbą symulacji na liść okazało się 32 i rówież przyjęcie
32 jako rozmiar bloków.

== Generowanie liczb losowych

Algorytm potrzebuje metody generowania losowej liczby do wybrania jednego z dostępnych ruchów.

Liczy się dla nas raczej szybkość niż "jakość" generowania. Dobrym wyborem wydaje się
#link("https://en.wikipedia.org/wiki/Xorshift")[xorshift32], który używa jedynie
trzech shiftów i trzech xorów:

```c
uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}
```

gdzie `state` może być zainicjalizowany w każdym wątku:
```c
int id = blockIdx.x * blockDim.x + threadIdx.x;
uint32_t state = 0x9E3779B9 ^ id;
```

Magiczna stała `0x9E3779B9` to $floor(2^32 slash phi)$, często używana jako "losowe" bity.
