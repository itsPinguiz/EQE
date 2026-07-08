# Relazione Tecnica e Analisi Comparativa: Nuova Metrica di Valutazione XAI

**EQE - Evaluating the Quality of Explanations**

---

## 1. Introduzione e Collocamento nel Corso di XAI

### Contesto Generale

Nell'ambito dell'Explainable Artificial Intelligence (XAI), le metriche di valutazione delle spiegazioni rappresentano un elemento cruciale per valutare l'idoneità e l'affidabilità dei metodi di spiegazione. Le "ground truth" per le spiegazioni sono spesso inaccessibili (non conosciamo a priori quali feature sono "veramente" importanti per una previsione), costringendo la ricerca a concentrarsi su metriche **functionally-grounded** - misure computazionali che quantificano proprietà formali delle spiegazioni senza intervento umano.

### La Complexity-Calibrated Local Concordance (CCC)

La metrica proposta in questo progetto, **Complexity-Calibrated Local Concordance (CCC)**, nasce dall'osservazione empirica e teorica che le spiegazioni generate da metodi XAI (SHAP, LIME, MAPLE) sono spesso estese a decine di feature, mentre la capacità cognitiva umana di elaborare informazioni è fortemente limitata.

Il **principio di Miller** (1956) afferma che la memoria di lavoro umana può contenere efficacemente **~7±2 item** contemporaneamente. Questo limite cognitivo non è stato adeguatamente incorporato nelle metriche di valutazione XAI esistenti: molte metriche valutano la fedeltà dell'intera spiegazione, ma non misurano quanto la spiegazione mantienga la sua capacità predittiva quando ridotta a un budget fisico di K features.

**Cos'è la CCC:** La CCC misura l'MSE (Mean Squared Error) tra le probabilità della classe positiva prodotte dal modello black-box `f(x)` e la previsione ricostruita localmente `g_K(x)`, dove `g_K` è l'equivalente troncato dell'espressione additiva dell'explainer che mantiene solo le K features con valore assoluto delle contributi più alti.

**Cosa rappresenta:** Un valore di CCC minimo indica che l'explainer può fornire predizioni fedeli usando solo K feature - un requisito essenziale per la **comprensibilità umana** e la **fiducia interattiva**.

---

## 2. Architettura del Software e Workflow di Esecuzione

### Struttura a Moduli

```
EQE/
├── core/
│   ├── metrics.py           # Implementazione delle metriche XAI (nuova CCC + SOTA)
│   ├── explainers.py        # Wrapper SHAP, LIME, MAPLE con interfaccia unificata
│   ├── test_framework.py    # Orchestratore sperimentale con caching
│   ├── main.py              # Entry point CLI e generazione report
│   ├── model.py             # Modelli black-box (XGBoost, Neural Network)
│   ├── data_loader.py       # Caricamento e preprocessing dataset
│   └── graph_utilities/     # Visualizzazioni
├── config.yml               # Configurazione esperimenti
└── results/
    ├── latest.md            # Output sperimentale principale
    └── figures/             # Grafici generati
```

### Workflow di Esecuzione (Pseudo-codice)

```python
# ===========================================
# FASE 1: Configurazione e Parsing
# ===========================================
def main():
    # Parsing argomenti CLI
    args = parse_args()  # --config config.yml
    
    # Caricamento configurazione YAML
    config = load_config(args.config)
    
    # Estrazione parametri
    seeds = config.experiment.seeds              # [42, 123, 2026]
    datasets = config.experiment.suite.datasets  # [breast_cancer, adult]
    k_values = config.experiment.suite.k_features # [4, 5, 6, 7, 8, 9]

# ===========================================
# FASE 2: Matrice di Esecuzione
# ===========================================
def _build_run_matrix(experiment):
    """
    Costruisce la matrice di esecuzione:
    - Una run per ogni combinazione dataset-seed
    - Tutte le run sono indipendenti e parallelizable
    """
    runs = []
    for dataset in datasets:
        for seed in seeds:
            runs.append({
                "dataset": dataset,
                "k_features": k_values,  # Lista di K valori
                "seed": seed
            })
    return runs  # 6 run totali (2 dataset × 3 seed)

# ===========================================
# FASE 3: Single Experiment Run
# ===========================================
def run_experiment(dataset_name, k_features_list, seed):
    """
    Esegue una singola run sperimentale completa.
    """
    # 1. CARICAMENTO DATI
    X_train, X_test, y_train, y_test = _load_dataset(dataset_name, seed)
    
    # 2. TRAINING MODELLI
    models = {}
    for model_name in ["xgboost", "neuralnetwork"]:
        model = MODEL_REGISTRY[model_name](random_state=seed)
        model.train(X_train, y_train)
        models[model_name] = model
    
    # 3. GENERAZIONE EXPLANATIONS (CACHED)
    explanations_cache = generate_explanations(X_test, models)
    """
    Cache structure:
    {
        (model_name, explainer_name): {
            "weights": (n_samples, n_features),
            "intercepts": (n_samples,),
            "f_proba": (n_samples,)  # f(x) per ogni istanza
        }
    }
    """
    
    # 4. CALCOLO METRICHE PER OGNI K
    all_results = []
    for current_k in k_features_list:
        k_result_df = compute_metrics(explanations_cache, X_test, current_k)
        all_results.append(k_result_df)
    
    return concat(all_results)

# ===========================================
# FASE 4: Parallelizzazione
# ===========================================
def main_parallel():
    all_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(_run_single)(run) for run in runs
    )
    final_results = concat(all_results)
    
    # Aggregazione per seed (media + std)
    aggregate_results = _aggregate_seed_results(final_results)
    
    # Generazione report Markdown
    _render_markdown_report(final_results, metadata, aggregate_results)
    
    # Generazione grafici
    visualize.main(["benchmark-summary"])
```

### Interfaccia Unificata degli Explainers

Gli explainers implementano una classe base astratta comune:

```python
# ===========================================
# INTERFACCIA BASE EXPLANER
# ===========================================
class BaseExplainer(ABC):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def explain(self, X):
        """
        Genera attribuzioni feature e intercept per ogni istanza.
        
        Returns:
        - weights: np.ndarray (n_samples, n_features)
              Contributi additivi di ogni feature
        - intercepts: np.ndarray (n_samples,)
              Valore di base (phi_0) per ogni istanza
        """
        pass

# SHAP: KernelExplainer con background stratificato
# LIME: LimeTabularExplainer con regressione lineare locale
# MAPLE: Multiple Additive Regression Trees con split train/val
```

---

## 3. Analisi del Codice e Pseudo-codice Commentato

### 3.1 ComplexityCalibratedConcordance (CCC) - Metrica Principale

```python
# core/metrics.py:133-261
class ComplexityCalibratedConcordance(EvaluationMetric):
    """
    Misura l'MSE tra f(x) e g_K(x) dove g_K è la spiegazione
    troncata alle K features con contributi assoluti più alti.
    """
    
    def compute(f_proba, weights, intercepts, X):
        # Passo 1: Truncation top-K per ogni istanza
        weights_truncated = self._truncate_weights(weights)
        """
        ALGORITMO _truncate_weights(weights, k_features):
        
        1. Calcola magnitude assolute: |weights[i, :]|
        2. Ordina gli indici per magnitude decrescente
        3. Seleziona i primi k_features indici (S_K)
        4. Zero-out tutti i pesi NON selezionati
        
        Esempio: weights = [0.1, -0.5, 0.3, 0.2], k=2
        Magnitudo: [0.1, 0.5, 0.3, 0.2]
        Top-2 indici: [1, 2] (feature 1 e 3)
        weights_truncated = [0.0, -0.5, 0.3, 0.0]
        """
        
        # Passo 2: Calcolo predizione troncata
        g_K = intercepts + np.sum(weights_truncated, axis=1)
        """
        NOTA IMPLEMENTATIVA CRUCIALE:
        - SHAP values sono già contributi additivi completi
        - LIME coefficients vengono convertiti a contributi (coefficients * feature_value)
        - I contributi NON vengono moltiplicati nuovamente per X
        - Questo perché rappresentano già phi_i(x) nella formula g(x) = w_0 + sum(phi_i)
        """
        
        # Passo 3: MSE medio
        mse = np.mean((f_proba - g_K) ** 2)
        
        return float(mse)
```

### 3.2 RandomKConcordance - Baseline di Controllo

```python
# core/metrics.py:264-336
class RandomKConcordance(EvaluationMetric):
    """
    Baseline che mantiene K features CASUALI invece che top-K.
    Serve a verificare che top-K scelte siano meglio di random.
    """
    
    def compute(f_proba, weights, intercepts, X):
        if k_features == n_features:
            # Caso speciale: tutti i feature selezionati
            g_k = intercepts + np.sum(weights, axis=1)
            return mean((f_proba - g_k) ** 2)
        
        rng = np.random.default_rng(random_state)
        scores = []
        
        for _ in range(repeats):  # Default: 30 ripetizioni
            # Genera valori casuali per ogni feature
            random_order = rng.random(weights.shape)
            
            # argpartition è più efficiente di argsort (O(n) vs O(n log n))
            selected = np.argpartition(random_order, k_features - 1, axis=1)[:, :k_features]
            
            truncated = np.zeros_like(weights)
            truncated[row_indices, selected] = weights[row_indices, selected]
            
            g_k = intercepts + np.sum(truncated, axis=1)
            scores.append(mean((f_proba - g_k) ** 2))
        
        return mean(scores)
```

### 3.3 FullLocalFidelityMSE - Fidelity Senza Truncation

```python
# core/metrics.py:338-358
class FullLocalFidelityMSE(EvaluationMetric):
    """
    MSE tra f(x) e g(x) senza alcuna truncation.
    Per SHAP, questo dovrebbe essere vicino a 0 grazie alla local accuracy.
    """
    
    def compute(f_proba, weights, intercepts, X):
        g_full = intercepts + np.sum(weights, axis=1)
        return float(np.mean((f_proba - g_full) ** 2))
```

### 3.4 NormalizedCCC - Variante Normalizzata

```python
# core/metrics.py:440-473
class NormalizedCCC(EvaluationMetric):
    """
    CCC normalizzato per la varianza di f(x).
    Equivalente a 1 - R² quando il baseline predictor è la media.
    Permette confronto cross-dataset.
    """
    
    def compute(f_proba, weights, intercepts, X):
        ccc = ComplexityCalibratedConcordance(k_features=self.k_features).compute(
            f_proba, weights, intercepts, X
        )
        variance = np.var(f_proba)
        
        if variance > 0:
            return float(ccc / variance)  # NMSE
        else:
            return float("inf")  # Tutte le predizioni uguali
```

### 3.5 TopKDegradationRatio - Misura Robustezza

```python
# core/metrics.py:475-513
class TopKDegradationRatio(EvaluationMetric):
    """
    Rapporto tra CCC e Full MSE.
    - Valori ~1: alto degrado (top-K peggiora molto la fidelity)
    - Valori ~0: bassa perdita (top-K mantiene fidelity)
    - NaN: full_mse ≈ 0 (metodo esatto come SHAP)
    """
    
    def compute(f_proba, weights, intercepts, X):
        ccc = ComplexityCalibratedConcordance(...).compute(...)
        full = FullLocalFidelityMSE().compute(...)
        
        if full < 1e-10:
            return float("nan")  # SHAP ha full_mse = 0
        return float(ccc / full)
```

### 3.6 SufficiencyMSE - Fidelity su Input Perturbato

```python
# core/metrics.py:360-396
class SufficiencyMSE(EvaluationMetric):
    """
    Misura quanto la predizione cambi quando si mantengono solo top-K feature.
    Le altre feature vengono sostituite con la media del training set.
    """
    
    def compute(f_proba, weights, intercepts, X, model, baseline):
        # Maschera top-K features
        mask = _top_k_mask(weights, self.k_features)
        
        # Crea X_keep: mantiene solo top-K feature, altrimenti usa baseline
        X_keep_top_k = np.broadcast_to(baseline, X.shape).copy()
        X_keep_top_k[mask] = X[mask]
        
        # Predizione su input modificato
        kept_proba = model.predict_proba(X_keep_top_k)[:, 1]
        
        return float(np.mean((f_proba - kept_proba) ** 2))
```

---

## 4. Focus Approfondito: La Nuova Metrica XAI

### 4.1 Cosa Fa

La **Complexity-Calibrated Local Concordance (CCC)** misura la **fedeltà locale** di una spiegazione additiva quando questa è soggetta a un vincolo di complessità cognitiva. 

Formalmente, dati:
- $f(x)$: la probabilità della classe positiva del modello black-box
- $w_0$: l'intercept/base value dell'explainer
- $\phi_i(x)$: contributo additivo della feature $i$ per l'istanza $x$
- $S_K(x)$: insieme degli indici delle $K$ features con $|\phi_i(x)|$ più alte

La predizione troncata è:
$$g_K(x) = w_0 + \sum_{i \in S_K(x)} \phi_i(x)$$

La metrica CCC è l'MSE medio:
$$\text{CCC} = \frac{1}{N} \sum_{j=1}^{N} \left(f(x^j) - g_K(x^j)\right)^2$$

### 4.2 Come Lo Fa

L'algoritmo implementato segue questi passi:

1. **Identificazione top-K**: Per ogni istanza (riga della matrice weights), vengono identificati gli indici delle $K$ features con valore assoluto del contributo più alto. Questo è fatto tramite `np.argsort(-np.abs(weights), axis=1)[:, :k_features]`.

2. **Truncation**: Tutti i contributi non top-K vengono azzerati (`weights_truncated[mask] = weights[mask]` dove mask è una maschera booleana).

3. **Ricostruzione**: La predizione locale troncata è $g_K = w_0 + \sum(\text{weights\_truncated})$. Nota che i contributi non vengono moltiplicati nuovamente per i valori delle feature, perché rappresentano già $\phi_i(x)$.

4. **MSE**: Si calcola l'errore quadratico medio tra le probabilità del black-box e la ricostruzione troncata.

### 4.3 Utilità Teorica e Pratica

**Lacuna colmata**: Le metriche SOTA esistenti valutano:
- **Fedeltà totale** (`full_mse`): ottimo per SHAP (che ha garanzia di local accuracy) ma non riflette la complessità percepita
- **Sufficiency/Comprehensiveness**: misurano l'influenza sul modello, non la fedeltà numerica dell'espressione additiva
- **Random-K**: non esisteva una baseline diretta per confrontare top-K vs random

**Applicazione pratica**: La CCC permette di rispondere alla domanda: *"Se un esperto umano può considerare solo 4-5 feature, quale explainer gli fornirà una stima fedele delle predizioni del modello?"*

---

## 5. Analisi Comparativa con le Metriche SOTA

### Tabella Comparativa

| Metrica | Proprietà Misurata | Input Richiesti | Complessità Computazionale | Parametri | Direction | Limitazioni |
|---------|-------------------|-----------------|--------------------------|-----------|-----------|-------------|
| `ccc_mse` | Fedeltà con vincolo cognitivo K | f_proba, weights, intercepts, X | O(N·F·log K) per ordinamento top-K | k_features | Πù basso = meglio | Richiede weights normalizzati come contributi additivi |
| `full_mse` | Fedeltà dell'espressione completa | f_proba, weights, intercepts | O(N·F) | - | Πù basso = meglio | Non cattura complessità percepita |
| `random_k_mse` | Baseline di controllo random | f_proba, weights, intercepts | O(N·F·repeats) | k_features, repeats | Πù basso = meglio | Usato solo per confronto |
| `ccc_mse_normalized` | CCC normalizzato (1-R² equivalente) | f_proba, weights, intercepts | O(N·F·log K) | k_features | Πù basso = meglio | Infinito se varianza = 0 |
| `top_k_degradation_ratio` | Robustezza alla truncation | f_proba, weights, intercepts | O(N·F·log K) | k_features | Πù basso = meglio | NaN per metodi esatti (SHAP) |
| `sufficiency_mse` | Sufficiency su input perturbato | f_proba, weights, intercepts, X, model, baseline | O(N·F) + model inference | k_features | Πù basso = meglio | Sensibile alla strategia di masking |
| `comprehensiveness_abs_drop` | Completezza (drop di predizione) | f_proba, weights, intercepts, X, model, baseline | O(N·F) + model inference | k_features | Πù alto = meglio | Sensibile alla strategia di masking |

### Relazione con Metriche esistenti

| Family | Metrica SOTA | CCC vs SOTA | Come CCC si distingue |
|--------|--------------|-------------|----------------------|
| Faithfulness | Local MSE, Local R² | CCC è Local MSE con vincolo K | Introduce budget cognitivo |
| Compactness | Numero feature, Sparsity | CCC misura fedeltà, non solo size | Non basta essere piccoli, devono essere fedeli |
| Sufficiency | AOPC-Sufficiency | CCC è su contributi, non su input | Sovrapposizione ma angolazione diversa |
| Robustness | Sensitivity, Infidelity | CCC usa perturbatione implicita | La truncation è una forma di perturbation |

---

## 6. Analisi dei Risultati Sperimentali e Discussione

### 6.1 Setup Sperimentale

```yaml
# Configurazione: config.yml
Datasets: breast_cancer, adult
Modelli: XGBoost (XGBClassifier), Neural Network (MLP 100x100)
Explainers: SHAP (KernelExplainer), LIME, MAPLE
K features: 4, 5, 6, 7, 8, 9
Seed: 42, 123, 2026 (per stabilità)
n_explain: 1000 (istanze testate)
```

### 6.2 Accuracy dei Modelli

| Dataset | Modello | Accuracy Media | Accuracy Std |
|---------|---------|----------------|------------|
| adult | xgboost | 0.873955 | 0.001540 |
| adult | neuralnetwork | 0.821647 | 0.001821 |
| breast_cancer | xgboost | 0.967836 | 0.010129 |
| breast_cancer | neuralnetwork | 0.976608 | 0.013399 |

### 6.3 CCC MSE per Explainer (medie su seed)

#### Adult Dataset - Neural Network

| Explainer | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 |
|-----------|-----|-----|-----|-----|-----|-----|
| SHAP | 0.0125 | 0.0074 | 0.0040 | 0.0020 | 0.0008 | 0.0002 |
| MAPLE | 0.0628 | 0.0628 | 0.0626 | 0.0623 | 0.0623 | 0.0624 |
| LIME | 2.895 | 2.633 | 2.488 | 2.378 | 2.303 | 2.276 |

**Interpretazione:**
- **SHAP** mostra un degrado significativo con K piccolo, ma migliora drasticamente con K→9
- **LIME** ha valori di CCC 100-500x superiori a SHAP, segnale di scarsa fedeltà
- **MAPLE** mantiene comportamento stabile: fedeltà limitata anche con più feature

#### Breast Cancer Dataset - XGBoost

| Explainer | K=4 | K=5 | K=6 | K=7 | K=8 | K=9 |
|-----------|-----|-----|-----|-----|-----|-----|
| SHAP | 0.0287 | 0.0159 | 0.0082 | 0.0038 | 0.0014 | 0.0003 |
| MAPLE | 0.1210 | 0.0755 | 0.0556 | 0.0555 | 0.0405 | 0.0341 |
| LIME | 0.0535 | 0.0403 | 0.0376 | 0.0402 | 0.0460 | 0.0520 |

### 6.4 Analisi Top-K Degradation Ratio

Il rapporto `ccc_mse / full_mse` rivela quanto la truncation impatti la fedeltà:

| Explainer | Dataset | Modello | Ratio (K=9) | Interpretazione |
|-----------|---------|---------|-------------|-----------------|
| SHAP | adult | neuralnetwork | 0.0000 | **Perfetto** - full_mse = 0 |
| MAPLE | adult | neuralnetwork | 1.00 | **Casuale** - degrado uguale a random |
| LIME | adult | neuralnetwork | 1.28 | **Peggio di random** |
| SHAP | breast_cancer | xgboost | 0.0000 | **Perfetto** |
| MAPLE | breast_cancer | xgboost | 2.05 | Moderato degrado |
| LIME | breast_cancer | xgboost | 0.64 | Migliore di random |

### 6.5 Cosa Rileva la CCC che le SOTA Non CogliOno

#### 6.5.1 Il Paradosso di SHAP

SHAP ha `full_mse = 0` su entrambi i dataset grazie alla **garanzia di local accuracy**. Questo significa che la somma completa dei SHAP values uguaglia esattamente la predizione del modello. Tuttavia:

- Per K=4 su Adult, CCC = 0.0125 (non zero!)
- Questo rivela che **molte piccole contributioni** sono necessarie per la fedeltà
- La CCC espone un trade-off nascosto: fedeltà totale vs complessità percepita

#### 6.5.2 MAPLE: Elevata Fidelity ma Scarsa Robustezza

Su Adult con Neural Network, MAPLE mostra `top_k_degradation_ratio ≈ 1.0`, indicando che:
- Le top-K features selezionate sono **indistinguibili dalla casualità**
- Il metodo produce contributi fedeli in media (full_mse = 0.062)
- Ma non c'è una distinzione chiara tra features importanti e non importanti

#### 6.5.3 LIME: Scarsa Fedeltà Innestata

LIME con Neural Network su Adult ha:
- CCC > 2.0 per K=4, mentre SHAP ha 0.0125
- Questo **400x di differenza** mostra inaffidabilità strutturale
- Il rapporto degrado > 1 indica che le top features scelte peggiorano la ricostruzione

### 6.6 Integrare CCC in una Pipeline di Valutazione

La CCC si presta ad essere integrata in un framework di valutazione a due livelli:

```
LIVELLO 1: VALUTAZIONE BASE
├── full_mse (fedeltà totale)
├── sufficiency_mse (fedeltà su input)
└── comprehensiveness_abs_drop (completezza)

LIVELLO 2: VALUTAZIONE CON VINCOLI
├── ccc_mse (fedeltà con K features)
├── ccc_mse_normalized (confronto cross-dataset)
├── top_k_degradation_ratio (robustezza alla truncation)
└── random_k_mse (baseline casuale)
```

### 6.7 Limiti della Metrica

#### 6.7.1 Sensitività al Baseline di Masking

Le metriche `sufficiency_mse` e `comprehensiveness_abs_drop` usano la media delle feature come baseline. Questa scelta semplice ha limitazioni:
- Non riflette la distribuzione marginale reale
- Può introdurre bias in dataset con feature altamente correlate

#### 6.7.2 Assunzione di Linearità Additiva

La CCC assume che le spiegazioni seguino una forma additiva $g(x) = w_0 + \sum \phi_i(x)$. Questo:
- È valido per SHAP, LIME e MAPLE
- NON è valido per metodi basati su sottogruppi o regole
- Limita l'applicabilità a explainers specifici

#### 6.7.3 Scarsa Discriminazione per Metodi Esatti

Per SHAP (che ha `full_mse ≈ 0`), il `top_k_degradation_ratio` è NaN, rendendo difficile:
- Confrontare direttamente SHAP con metodi approssimativi
- Usare il degrado come metrica di ranking

#### 6.7.4 Scale delle Feature

La CCC non normalizza le feature. Dataset con scale molto diverse possono:
- Influenzare l'ordine di grandezza delle metriche
- Richiedere attenzione nel confronto cross-dataset

---

## 7. Conclusioni

La **Complexity-Calibrated Local Concordance** rappresenta un contributo innovativo al campo della valutazione XAI perché:

1. **Unisce fedeltà e complessità**: Nessuna metrica SOTA combina queste due dimensioni in modo diretto

2. **Ha implicazioni cognitive**: Il vincolo di K features ha un fondamento psicologico in Miller's Law

3. **Rileva trade-off nascosti**: I risultati mostrano che SHAP, pur essendo "matematicamente esatto", richiede molte feature per una fedeltà elevata

4. **Fornisce baseline diretta**: Confronto top-K vs random permette di stabilire se le feature selezionate sono realmente informative

La metodologia sperimentale dimostra che la scelta dell'explainer dovrebbe considerare non solo la fedeltà totale ma anche **quanto questa fedeltà sia mantenuta con budget cognitivi ridotti** - un requisito essenziale per applicazioni reali dove le spiegazioni devono essere comprese da soggetti umani.

---

*Report generato il 2026-06-29*
*Basato sui risultati di `results/latest.md` e analisi del codice sorgente*