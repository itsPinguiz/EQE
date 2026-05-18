# Valutazione SOTA delle Spiegazioni con CCC-MSE

## 1. Obiettivo del Progetto

Questo progetto studia come valutare la qualita' delle spiegazioni prodotte da
tecniche di Explainable AI. L'obiettivo non e' soltanto confrontare metodi di
spiegazione esistenti, ma anche proporre una strategia di valutazione capace di
misurare aspetti spesso trattati separatamente nella letteratura.

In particolare, il progetto si concentra su una domanda specifica:

> Una spiegazione rimane fedele al modello anche quando viene limitata a un
> numero ridotto di feature interpretabili da un essere umano?

Per rispondere, e' stata proposta e implementata una metrica chiamata
**Complexity-Calibrated Local Concordance**, misurata tramite `ccc_mse`.

La metrica combina due dimensioni:

- **faithfulness**, cioe' quanto la spiegazione ricostruisce il comportamento
  locale del modello black-box;
- **compactness**, cioe' quanto la spiegazione resta fedele usando solo le top
  `K` feature.

Questa impostazione risponde direttamente alla specifica del progetto: analizzare
metodi esistenti, identificare limiti negli approcci di valutazione e proporre
una metrica che catturi un aspetto della qualita' delle spiegazioni ancora poco
esplorato in modo sistematico.

## 2. Contesto e Motivazione

I metodi di spiegazione, come LIME, SHAP e MAPLE, aiutano a interpretare le
decisioni di modelli complessi. Tuttavia, valutare la qualita' delle spiegazioni
rimane un problema aperto.

La letteratura propone molte famiglie di metriche:

- **faithfulness / fidelity**: misura se la spiegazione riflette davvero il
  comportamento del modello;
- **plausibility**: misura se la spiegazione e' coerente con il ragionamento
  umano o con annotazioni umane;
- **robustness / stability**: misura se la spiegazione resta stabile rispetto a
  piccole perturbazioni dell'input o del modello;
- **compactness / conciseness**: misura se la spiegazione e' sufficientemente
  sintetica da essere leggibile;
- **usability**: misura se la spiegazione aiuta effettivamente un utente in un
  compito reale.

Il problema e' che molte valutazioni si concentrano su una sola dimensione. Una
spiegazione puo' essere fedele ma troppo lunga, oppure compatta ma poco fedele.
Allo stesso modo, una spiegazione puo' sembrare plausibile a un essere umano ma
non rappresentare davvero il comportamento del modello.

Il contributo di questo progetto si colloca in questo spazio: valutare la
fedelta' locale tenendo conto di un vincolo di complessita' cognitiva.

## 3. Gap di Ricerca Identificato

Dalla revisione in `docs/PAPERS_REVIEW.md` emergono alcuni limiti ricorrenti:

- molte metriche valutano una sola proprieta' della spiegazione;
- i benchmark non sempre permettono un confronto uniforme tra explainer diversi;
- la fedelta' viene spesso misurata senza imporre un limite alla dimensione della
  spiegazione;
- il legame tra metriche quantitative e interpretabilita' umana rimane debole;
- i metodi post-hoc vengono spesso confrontati senza un controllo casuale, quindi
  non e' sempre chiaro se le feature selezionate siano davvero informative.

Il gap affrontato qui e' quindi:

> Le metriche di fidelity tradizionali non distinguono chiaramente tra una
> spiegazione fedele perche' usa molte feature e una spiegazione fedele pur
> restando compatta.

La metrica `ccc_mse` cerca di rendere misurabile questo trade-off.

## 4. Metrica Proposta: CCC-MSE

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
- `g_K(x)` e' la ricostruzione della spiegazione limitata alle top `K` feature;
- `K` e' il vincolo di complessita' cognitiva.

Un valore piu' basso indica una spiegazione piu' fedele sotto vincolo di
compattezza.

## 5. Baseline Random-K

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

## 6. Framework Implementato

Il framework sperimentale valuta spiegazioni su dati tabulari binari.

| Componente | Scelta implementata |
| --- | --- |
| Dataset | `breast_cancer`, `adult` |
| Modelli black-box | `xgboost`, `neuralnetwork` |
| Explainer | `lime`, `shap`, `maple` |
| Metrica proposta | `ccc_mse` |
| Baseline | `random_k_mse` |
| Valori di K | `4, 5, 6, 7, 8, 9` |
| Istanze spiegate su Breast Cancer | 114 |
| Istanze spiegate su Adult | 9769 |

Il run di riferimento e':

```text
results/SOTA/results_20260518_141724.md
```

Le configurazioni sono controllate da `config.yml`, in modo da avere un unico
punto per modificare dataset, modelli, explainer, metriche, valori di `K`,
parallelizzazione e parametri specifici degli explainer.

## 7. Risultati Principali

### 7.1 Migliori configurazioni

| Dataset | Miglior modello | Miglior explainer | K | Accuracy | `ccc_mse` | `random_k_mse` |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `adult` | XGBoost | SHAP | 9 | 0.872351 | 0.000034 | 0.084953 |
| `breast_cancer` | XGBoost | SHAP | 9 | 0.956140 | 0.000346 | 0.112468 |

La configurazione migliore e':

> XGBoost + SHAP con `K = 9`.

Questo vale per entrambi i dataset.

### 7.2 Confronto tra explainer

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

### 7.3 Confronto a K = 9

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

## 8. Interpretazione dei Risultati

### 8.1 SHAP rimane lo stato dell'arte nel benchmark

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

### 8.2 MAPLE aggiunge una baseline informativa

MAPLE non supera SHAP, ma e' importante per il progetto per due motivi:

- migliora sensibilmente rispetto a LIME sul dataset Adult;
- fornisce una baseline locale lineare piu' forte, rendendo il confronto meno
  dipendente da un semplice dualismo LIME vs SHAP.

Questo aiuta a rendere la valutazione piu' sistematica e comparabile.

### 8.3 Random-K valida la metrica

Il controllo random-K e' uno dei risultati piu' importanti:

| Check | Risultato |
| --- | ---: |
| Configurazioni in cui `ccc_mse < random_k_mse` | 72 / 72 |
| Configurazioni in cui random-K eguaglia o supera top-K | 0 / 72 |

Questo dimostra che le feature selezionate dagli explainer non sono equivalenti a
feature casuali. Le top `K` feature conservano piu' informazione sulla predizione
del modello rispetto a un sottoinsieme casuale della stessa dimensione.

## 9. Perche' la Metrica e' Utile

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

## 10. Limiti

### Un solo run per dataset

Il run corrente e' sufficiente per una snapshot SOTA, ma non per affermazioni
statistiche forti. Servono piu' seed e deviazioni standard.

### Dataset ancora limitati

Adult e Breast Cancer sono dataset tabulari utili, ma una valutazione piu'
completa dovrebbe includere altri domini, ad esempio credito, diabete o rischio
cardiaco.

### La metrica non misura direttamente la plausibilita'

`ccc_mse` misura una proprieta' functionally-grounded: la fedelta' della
spiegazione al modello. Non misura se la spiegazione e' percepita come plausibile
da esseri umani.

### La metrica non sostituisce uno user study

Il vincolo `K` e' motivato da considerazioni cognitive, ma la metrica non prova
che gli utenti comprendano meglio le spiegazioni. Per questo servirebbe uno
studio human-grounded.

### L2X richiede un percorso separato

L2X non produce coefficienti locali additivi come SHAP, LIME o MAPLE. Produce
probabilita' di selezione delle feature. Andrebbe quindi valutato con una
variante di subset fidelity, non forzato dentro la stessa interfaccia additiva.

## 11. Prossimi Esperimenti

1. Aggiungere piu' seed.

   Ripetere ogni configurazione su piu' seed e riportare media, deviazione
   standard e intervalli di confidenza.

2. Estendere il range di `K`.

   Aggiungere `K = 1, 2, 3` per valutare vincoli cognitivi piu' severi.

3. Aggiungere stabilita' delle top feature.

   Misurare quanto l'insieme delle top `K` feature cambia tra run, perturbazioni
   o seed diversi.

4. Aggiungere L2X con subset fidelity.

   Valutare L2X selezionando le top `K` feature per probabilita' di selezione,
   mascherando le altre, e confrontando la predizione del modello sull'input
   mascherato con quella originale.

5. Integrare una dimensione human-grounded.

   Possibili estensioni includono plausibilita' rispetto ad annotazioni umane,
   task-based evaluation o misure di utilita' per utenti finali.

## 12. Sintesi Finale

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
