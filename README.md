# portfolio-optim — Allocation de portefeuille sous incertitude

> **Résultat visé (une phrase).** Une allocation construite *en tenant compte de
> l'incertitude des prévisions de rendement* obtient, sur un backtest 2007→aujourd'hui,
> un meilleur **Sharpe ratio** et un **max drawdown** plus faible qu'une allocation
> naïve (1/N et Markowitz max-Sharpe sur estimateurs d'échantillon).

Ce projet **n'est pas un benchmark de méthodes**. C'est une **décision**
d'allocation et la mesure de son **impact**. Les comparaisons d'architectures
(LSTM vs ARIMA, bootstrap vs MC Dropout vs VAE) sont des *détails d'implémentation*
documentés en annexe — une seule combinaison gagnante est mise en avant dans le
pipeline principal.

---

## Le pipeline en 4 couches

```
            ┌─────────────────────────────────────────────────────────────┐
 Couche 0   │  DONNÉES  (src/etl, src/features)                            │
            │  Univers d'ETFs → panel de prix/rendements propre (parquet)  │
            └─────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────────────────────────────────────────┐
 Couche 1   │  FORECASTING  (src/models)                                   │
            │  Prévision des rendements attendus par actif                 │
            │  → gagnant mis en avant ; baselines (ARIMA, MA) en annexe    │
            └─────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────────────────────────────────────────┐
 Couche 2   │  INCERTITUDE  (src/models)                                   │
            │  Quantification de l'incertitude de chaque prévision         │
            │  → distribution / intervalles qui alimentent l'optimiseur    │
            └─────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────────────────────────────────────────┐
 Couche 3   │  OPTIMISATION  (src/optimization, cvxpy)                     │
            │  Allocation sous contraintes qui PÉNALISE l'incertitude      │
            │  (≈ robust / Bayesian mean-variance) → poids du portefeuille │
            └─────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────────────────────────────────────────┐
 Couche 4   │  BACKTEST  (src/evaluation)                                  │
            │  Walk-forward vs 1/N et Markowitz naïf                        │
            │  → Sharpe, max drawdown, turnover, courbes d'équité          │
            └─────────────────────────────────────────────────────────────┘
```

La couche 2 est le cœur de la thèse : **l'incertitude n'est pas un diagnostic,
c'est un intrant de la décision.** Une prévision peu fiable doit se traduire par
une allocation plus prudente sur cet actif.

---

## Décisions de cadrage (figées)

| Sujet | Choix | Raison |
|---|---|---|
| **Univers** | ~17 ETFs hybrides par classe d'actifs (actions US/Europe/EM, oblig. govt/corp/IG/HY/TIPS, immobilier, or, matières premières, cash) | Diversification + historique long |
| **Période** | 2007 → aujourd'hui | Inclut la crise 2008 — meilleur stress-test du drawdown |
| **Devise (données)** | USD | CHF = reporting downstream (voir TODO) |
| **Benchmarks à battre** | **1/N** (plancher dur, DeMiguel 2009) **+ Markowitz max-Sharpe naïf** (repère d'instabilité) | Battre les deux = l'argument le plus fort |

Univers détaillé : voir [`src/config.py`](src/config.py).

---

## Structure

```
portfolio-optim/
├── data/
│   ├── raw/          # prix bruts téléchargés (gitignored)
│   └── processed/    # panel propre : prices / returns / coverage (parquet)
├── src/
│   ├── config.py     # source unique : chemins, dates, univers
│   ├── etl/          # ✅ couche 0 — téléchargement & nettoyage
│   ├── features/     # ✅ couche 0 — construction de features
│   ├── models/       # ✅ couche 1 forecasting ; ⬜ couche 2 incertitude
│   ├── optimization/ # ⬜ couche 3 — allocation (cvxpy)
│   └── evaluation/   # ⬜ couche 4 — backtest & métriques
├── notebooks/        # exploration uniquement
├── results/          # figures/ & metrics/ (gitignored)
├── tests/
├── Makefile
└── requirements.txt
```

---

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
make setup      # dépendances épinglées
make data       # télécharge & nettoie l'univers -> data/processed/
make test       # tests ETL (sans réseau)
make help       # liste toutes les targets
```

Targets du pipeline : `data` → `features` → `forecast` → `uncertainty`
→ `optimize` → `backtest` (`make all`).

---

## État & TODO

- [x] **Couche 0 — Données** : ETL yfinance, nettoyage (jours fériés / NaN),
      sortie parquet + rapport de couverture, tests.
- [x] **Couche 0 — Features** : momentum (1/3/6/12m, 12-1), vol & downside-vol
      glissantes, écart MA200, drawdown 1 an ; cible = rendement fin-de-mois →
      fin-de-mois (non chevauchant), contrat anti-fuite testé. Horizon **mensuel**.
- [x] **Couche 1 — Forecasting** : modèle de tête = **ML poolé cross-sectionnel**
      (GBM), walk-forward à fenêtre expansive sans fuite, métriques orientées
      allocation (IC, IR, hit-rate, RMSE). Annexe = ridge / moving-average /
      historical-mean derrière la même interface. ARIMA & LSTM/Transformer : à
      brancher en annexe (interface prête).
- [ ] **Couche 2 — Incertitude** : MC Dropout (réutilise immo_antibes) vs
      bootstrap vs VAE ; **calibration** des intervalles à vérifier.
- [ ] **Couche 3 — Optimisation** : formulation cvxpy mean-variance robuste
      pénalisant l'incertitude ; contraintes (long-only, plafonds, budget).
- [ ] **Couche 4 — Backtest** : walk-forward, rebalancement périodique, coûts
      de transaction ; Sharpe / max DD / turnover vs les deux benchmarks.
- [ ] **Reporting CHF** : conversion USD→CHF des courbes d'équité finales.
- [ ] **Annexe** : tableau comparatif des méthodes (architectures, incertitude).

---

## Lignée

Extension de `immo_antibes` (benchmark LSTM/GRU/Transformer/MA + MC Dropout sur
séries immobilières). La leçon retenue : **ne pas s'arrêter au benchmark.** Ici,
le livrable est une allocation et son impact mesuré, pas un classement de modèles.
```
