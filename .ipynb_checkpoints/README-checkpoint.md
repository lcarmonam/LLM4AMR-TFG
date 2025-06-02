# Interpretable Large Language Models for Early Prediction of Antimicrobial Multidrug Resistance 
*Lucía Carmona-Martos, Paula Martín-Palomeque, Óscar Escudero-Arnanz, Cristina Soguero-Ruiz*

Purpose: The growing burden of Antimicrobial Resistance (AMR) in Intensive
Care Units (ICUs) poses a significant threat to global health, increasing patient
mortality, morbidity, and healthcare costs. Early prediction of AMR is essen-
tial for timely intervention and effective treatment. This study proposes novel
Large Language Model (LLM)-based architectures for the classification of AMR
in ICU patients, using Electronic Health Records (EHRs) modeled as irregular
Multivariate Time Series (MTS).
Methods: We evaluated the proposed LLM-based models using a dataset of
3,502 anonymized EHRs from ICU patients at the University Hospital of Fuen-
labrada (Madrid, Spain), collected between 2004 and 2020. Their performance
was compared to that of conventional deep learning (DL) models, including
Gated Recurrent Units, Long Short-Term Memory networks, and Transformers.
All models were trained on irregular MTS data with predictive windows of 4, 7,
and 14 days. Interpretability was assessed using SHapley Additive exPlanations
(SHAP) values, and performance differences were analyzed through hypothesis
testing.
Results: The LLM-based models significantly outperformed baseline DL archi-
tectures, achieving a maximum ROC-AUC of 0.792 ± 0.009 in the 7-day window.
Blocks-LLM and InstructTime-LLM showed superior performance in handling
irregular data and capturing relevant clinical patterns. SHAP analysis revealed
 clinically consistent risk factors, including catheter use, antibiotic exposure, and
microbial cultures.

---


## Project Structure

- **Conventional DL Models**: GRU, LSTM, Transformer
- **LLM-Based Models**: Blocks-LLM, InstructTime-LLM, LLM-Few
- Models are trained using irregular MTS data with predictive windows of 4, 7, and 14 days.

- `src/`
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
  - `results/`
    - `Performance_results`
    - `Stat_Test`

---



### Evaluation

- **Performance Metrics**: ROC-AUC and others
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Statistical Analysis**: Hypothesis testing on model comparisons

---

### Data Availability

The dataset consists of 3,502 anonymized EHRs from ICU patients at the University Hospital of Fuenlabrada (Madrid, Spain), collected between 2004 and 2020.  
**Due to privacy and ethical constraints, the dataset is not publicly available.**

---
