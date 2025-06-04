# Large Language Models for Early Prediction of Antimicrobial Multidrug Resistance in Intensive Care Unit

This repository contains the code developed for my Bachelor Thesis, focused on the development and analysis of Large Language Models (LLMs) adapted for clinical Multivariate Time Series (MTS) to enable early prediction of Antimicrobial Multidrug Resistance (AMR) in ICU patients.

---


## Project Structure

- **Conventional DL Models**: GRU, LSTM, Transformer
- **LLM-Based Models**: Blocks-LLM, InstructTime-LLM, LLM-Few
- Models are trained using irregular MTS data with predictive windows of 4, 7, and 14 days.
- Three training strategies were explored: binary cross-entropy minimization, ROC-AUC maximization, and a custom loss function penalizing sensitivity-specificity imbalance.


- `src/`
  - `EDA/`
    - `EDA_final.ipynb`

  - `Experiment1/`
    - `benchmark/`
      - `GRU/`, `LSTM/`, `Transformer/`
        - `w4/`
        - `w7/`
        - `w14/`

    - `llms/`
      - `Blocks-LLM/`, `InstructTime-LLM/`, `LLMFew/`
        - `w4/`
        - `w7/`
        - `w14/`

    - `Results/`
      - `Performance_Results.ipynb`

  - `Experiment2/`
    - `benchmark/`
      - `GRU/`, `LSTM/`, `Transformer/`
        - `w4/`
        - `w7/`
        - `w14/`

    - `llms/`
      - `Blocks-LLM/`, `InstructTime-LLM/`, `LLMFew/`
        - `w4/`
        - `w7/`
        - `w14/`

    - `Results/`
      - `Performance_Results.ipynb`

  - `Experiment3/`
    - `benchmark/`
      - `GRU/`, `LSTM/`, `Transformer/`
        - `w4/`
        - `w7/`
        - `w14/`

    - `llms/`
      - `Blocks-LLM/`, `InstructTime-LLM/`, `LLMFew/`
        - `w4/`
        - `w7/`
        - `w14/`

    - `Results/`
      - `Performance_Results.ipynb`

---



### Evaluation

- **Performance Metrics**: ROC-AUC, Sensitivity, Specificity and F1 score
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Statistical Analysis**: Hypothesis testing on model comparisons

---

### Data Availability

The dataset consists of 3,502 anonymized EHRs from ICU patients at the University Hospital of Fuenlabrada (Madrid, Spain), collected between 2004 and 2020.  
**Due to privacy and ethical constraints, the dataset is not publicly available.**

---
