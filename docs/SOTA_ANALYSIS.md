# Valutazione SOTA delle Spiegazioni con CCC-MSE

## 1. Obiettivo, Contesto e Gap

Questo progetto studia come valutare la qualita' delle spiegazioni prodotte da
tecniche di Explainable AI. La letteratura considera molte dimensioni, tra cui
**faithfulness**, **plausibility**, **robustness**, **stability**,
**compactness** e **usability**. Tuttavia, molte metriche valutano una sola
proprieta' alla volta, rendendo difficile capire se una spiegazione sia insieme
fedele al modello e leggibile per un utente.

Il focus di questo lavoro e' il rapporto tra **faithfulness** e
**compactness**:

> Una spiegazione rimane fedele al modello anche quando viene limitata a un
> numero ridotto di feature interpretabili da un essere umano?

Dalla revisione dei paper emerge un gap specifico: le metriche di fidelity
tradizionali non distinguono chiaramente tra una spiegazione fedele perche' usa
molte feature e una spiegazione fedele pur restando compatta.

Per affrontare questo gap e' stata proposta e implementata la
**Complexity-Calibrated Local Concordance**, misurata tramite `ccc_mse`. La
metrica valuta la ricostruzione locale del modello dopo aver mantenuto solo le
top `K` feature.

## 2. Metrica Proposta: CCC-MSE

La metrica implementata e' una versione MSE della **Complexity-Calibrated Local
Concordance**.

Per ogni istanza:

1. si genera una spiegazione locale;
2. si ordinano le feature per importanza assoluta;
3. si mantengono solo le top `K` feature;
4. si ricostruisce la predizione locale usando solo quelle feature;
5. si misura l'errore rispetto alla probabilita' prodotta dal modello black-box.

Formalmente:

```text
ccc_mse = mean((f(x) - g_K(x))^2)
```

dove:

- `f(x)` e' la probabilita' della classe positiva prodotta dal modello black-box;
- `g(x)` e' la ricostruzione locale additiva prodotta dall'explainer usando
  tutte le feature;
- `g_K(x)` e' la stessa ricostruzione locale additiva dopo aver mantenuto solo
  le top `K` feature;
- `K` e' il vincolo di complessita' cognitiva.

In altre parole, se l'explainer produce un base value `phi_0` e contributi
additivi `phi_i(x)`, allora:

```text
g(x) = phi_0 + sum_i phi_i(x)
g_K(x) = phi_0 + sum_{i in S_K(x)} phi_i(x)
```

con `S_K(x)` pari agli indici delle `K` feature con contributo assoluto piu'
alto.

Un valore piu' basso indica una spiegazione piu' fedele sotto vincolo di
compattezza.

## 3. Baseline Random-K

Per verificare che la selezione top-`K` non sia arbitraria, e' stata aggiunta una
baseline `random_k_mse`.

Questa baseline usa gli stessi pesi della spiegazione, ma invece di scegliere le
top `K` feature sceglie `K` feature casuali. In questo modo si controlla se le
feature selezionate dall'explainer sono davvero piu' informative di un sottoinsieme
casuale della stessa dimensione.

La baseline risponde a una domanda importante:

> Le top `K` feature scelte dall'explainer spiegano meglio del caso?

Nel run corrente, la risposta e' si': `ccc_mse` batte `random_k_mse` in tutte le
72 configurazioni valutate.

## 4. Framework Implementato

Il framework sperimentale valuta spiegazioni su dati tabulari binari.

| Componente | Scelta implementata |
| --- | --- |
| Dataset | `breast_cancer`, `adult` |
| Modelli black-box | `xgboost`, `neuralnetwork` |
| Explainer | `lime`, `shap`, `maple` |
| Metrica proposta | `ccc_mse` |
| Baseline | `random_k_mse` |
| Metriche di confronto | `full_mse`, `top_k_degradation_mse`, `compactness_ratio`, `sufficiency_mse`, `comprehensiveness_abs_drop` |
| Valori di K | `4, 5, 6, 7, 8, 9` |
| Istanze spiegate su Breast Cancer | 114 |
| Istanze spiegate su Adult | 9769 |

## 5. Risultati Principali

### 5.1 Migliori configurazioni

| Dataset | Miglior modello | Miglior explainer | K | Accuracy | `ccc_mse` | `random_k_mse` |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `adult` | XGBoost | SHAP | 9 | 0.872351 | 0.000034 | 0.084953 |
| `breast_cancer` | XGBoost | SHAP | 9 | 0.956140 | 0.000346 | 0.112468 |

La configurazione migliore e':

> XGBoost + SHAP con `K = 9`.

Questo vale per entrambi i dataset.

### 5.2 Confronto tra explainer

Media di `ccc_mse` su tutti i valori di `K` e su entrambi i modelli:

| Dataset | Explainer | Mean `ccc_mse` |
| --- | --- | ---: |
| `adult` | SHAP | 0.002658 |
| `adult` | MAPLE | 0.041034 |
| `adult` | LIME | 0.404233 |
| `breast_cancer` | SHAP | 0.015021 |
| `breast_cancer` | MAPLE | 0.057192 |
| `breast_cancer` | LIME | 0.078627 |

Il ranking e' chiaro:

```text
SHAP < MAPLE < LIME
```

dove valori piu' bassi sono migliori.

SHAP e' il miglior explainer in termini di fedelta' sotto vincolo di `K`. MAPLE
si comporta come una baseline intermedia utile: e' spesso molto migliore di LIME,
ma non raggiunge SHAP.

### 5.3 Confronto a K = 9

| Dataset | Model | Explainer | Accuracy | `ccc_mse` | `random_k_mse` |
| --- | --- | --- | ---: | ---: | ---: |
| `adult` | XGBoost | SHAP | 0.872351 | 0.000034 | 0.084953 |
| `adult` | NeuralNetwork | SHAP | 0.818098 | 0.000175 | 0.118583 |
| `adult` | XGBoost | MAPLE | 0.872351 | 0.017218 | 0.034210 |
| `adult` | NeuralNetwork | MAPLE | 0.818098 | 0.064829 | 0.095013 |
| `adult` | XGBoost | LIME | 0.872351 | 0.106733 | 0.543361 |
| `adult` | NeuralNetwork | LIME | 0.818098 | 0.653839 | 2.202006 |
| `breast_cancer` | XGBoost | SHAP | 0.956140 | 0.000346 | 0.112468 |
| `breast_cancer` | NeuralNetwork | SHAP | 0.964912 | 0.000851 | 0.115746 |
| `breast_cancer` | NeuralNetwork | MAPLE | 0.964912 | 0.030099 | 0.471046 |
| `breast_cancer` | XGBoost | MAPLE | 0.956140 | 0.033755 | 0.461849 |
| `breast_cancer` | XGBoost | LIME | 0.956140 | 0.049569 | 0.208233 |
| `breast_cancer` | NeuralNetwork | LIME | 0.964912 | 0.109622 | 0.230874 |

## 6. Interpretazione dei Risultati

### 6.1 SHAP rimane lo stato dell'arte nel benchmark

SHAP ottiene il miglior `ccc_mse` in entrambi i dataset e per entrambi i modelli.
Inoltre migliora fortemente all'aumentare di `K`, come ci si aspetta da un metodo
additivo: mantenendo piu' contributi importanti, la ricostruzione locale si
avvicina alla predizione black-box.

Da `K = 4` a `K = 9`:

| Dataset | Model | Explainer | K=4 `ccc_mse` | K=9 `ccc_mse` | Variazione |
| --- | --- | --- | ---: | ---: | ---: |
| `adult` | XGBoost | SHAP | 0.002755 | 0.000034 | -98.8% |
| `adult` | NeuralNetwork | SHAP | 0.012369 | 0.000175 | -98.6% |
| `breast_cancer` | XGBoost | SHAP | 0.031699 | 0.000346 | -98.9% |
| `breast_cancer` | NeuralNetwork | SHAP | 0.051071 | 0.000851 | -98.3% |

### 6.2 MAPLE aggiunge una baseline informativa

MAPLE non supera SHAP, ma e' importante per il progetto per due motivi:

- migliora sensibilmente rispetto a LIME sul dataset Adult;
- fornisce una baseline locale lineare piu' forte, rendendo il confronto meno
  dipendente da un semplice dualismo LIME vs SHAP.

Questo aiuta a rendere la valutazione piu' sistematica e comparabile.

### 6.3 Random-K valida la metrica

Il controllo random-K e' uno dei risultati piu' importanti:

| Check | Risultato |
| --- | ---: |
| Configurazioni in cui `ccc_mse < random_k_mse` | 72 / 72 |
| Configurazioni in cui random-K eguaglia o supera top-K | 0 / 72 |

Questo dimostra che le feature selezionate dagli explainer non sono equivalenti a
feature casuali. Le top `K` feature conservano piu' informazione sulla predizione
del modello rispetto a un sottoinsieme casuale della stessa dimensione.

## 7. Perche' la Metrica e' Utile

La metrica e' utile perche' fornisce informazioni che l'accuracy non mostra.

### Adult

| Model | Accuracy | Best explainer | Best `ccc_mse` |
| --- | ---: | --- | ---: |
| XGBoost | 0.872351 | SHAP | 0.000034 |
| NeuralNetwork | 0.818098 | SHAP | 0.000175 |

Su Adult, XGBoost e' sia piu' accurato sia piu' facilmente spiegabile sotto il
vincolo top-`K`.

### Breast Cancer

| Model | Accuracy | Best explainer | Best `ccc_mse` |
| --- | ---: | --- | ---: |
| XGBoost | 0.956140 | SHAP | 0.000346 |
| NeuralNetwork | 0.964912 | SHAP | 0.000851 |

Su Breast Cancer, invece, emerge un trade-off: la rete neurale e' piu' accurata,
ma XGBoost ha spiegazioni piu' fedeli sotto vincolo di complessita'.

Questo e' esattamente il tipo di informazione che la metrica vuole catturare. Una
valutazione basata solo su accuracy sceglierebbe la rete neurale; una valutazione
che considera anche la qualita' della spiegazione mostra che XGBoost puo' essere
preferibile se l'obiettivo include interpretabilita' e fedelta' locale.

In sintesi, `ccc_mse` e' utile perche':

- separa performance predittiva e qualita' della spiegazione;
- misura la fedelta' sotto un vincolo di compattezza;
- permette di confrontare explainer diversi con lo stesso budget `K`;
- mostra come cambia la fedelta' al variare di `K`;
- evidenzia trade-off tra accuratezza e spiegabilita';
- include un controllo random-K per verificare che la selezione top-`K` sia
  significativa.

## 8. Limiti e Prossimi Esperimenti

Il run corrente e' una snapshot utile, ma non esaurisce la valutazione richiesta
da un framework completo. I limiti principali sono: un solo seed per dataset,
due soli dataset tabulari, assenza di una misura human-grounded diretta e
assenza di un'analisi di stabilita' delle top feature.

I prossimi esperimenti piu' rilevanti sono:

1. Aggiungere piu' seed.

   Ripetere ogni configurazione su piu' seed e riportare media, deviazione
   standard e intervalli di confidenza.

2. Estendere il range di `K`.

   Aggiungere `K = 1, 2, 3` per valutare vincoli cognitivi piu' severi.

3. Misurare la stabilita' delle top feature.

   Valutare quanto l'insieme delle top `K` feature cambia tra seed, run o
   piccole perturbazioni dell'input.

4. Aggiungere L2X con subset fidelity.

   L2X produce probabilita' di selezione delle feature, non coefficienti locali
   additivi. Va quindi valutato selezionando le top `K` feature, mascherando le
   altre e confrontando la predizione del modello sull'input mascherato.

5. Integrare una dimensione human-grounded.

   Possibili estensioni includono confronto con annotazioni umane, task-based
   evaluation o misure di utilita' per utenti finali.

## 9. Sintesi Finale

Il progetto propone una strategia di valutazione per spiegazioni locali che
combina fedelta' e compattezza. La metrica `ccc_mse` misura quanto una
spiegazione riesce a ricostruire la predizione di un modello black-box usando
solo le top `K` feature. Nel benchmark corrente, SHAP e' il miglior explainer,
MAPLE rappresenta una baseline intermedia utile e LIME e' il metodo piu' debole.
Il confronto con `random_k_mse` mostra che le top `K` feature selezionate dagli
explainer sono sempre piu' informative di feature casuali. Inoltre, la metrica
evidenzia trade-off che l'accuracy da sola non mostra, come nel caso Breast
Cancer, dove la rete neurale e' piu' accurata ma XGBoost produce spiegazioni piu'
fedeli sotto vincolo cognitivo. Questi risultati supportano `ccc_mse` come una
metrica utile per valutare spiegazioni in modo quantitativo, comparabile e piu'
sensibile alla complessita' richiesta all'utente.
