# Interpretable Credit Risk Modeling (SHAP & LIME)

This repository contains a synthetic credit risk dataset and a runnable Python script that trains a model and (optionally) produces SHAP and LIME explanations.

Files included:
- `credit_risk_synthetic.csv` : synthetic anonymized dataset (5000 rows)
- `main.py` : runnable pipeline (train, evaluate, save model, generate explainability outputs)
- `report.md` : short report summarizing model, results and interpretation

How to run:
1. Clone the repository or download these files.
2. Create a virtual environment and install required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate     # linux / mac
   venv\Scripts\activate        # windows
   pip install -U scikit-learn pandas numpy shap lime joblib matplotlib
   ```
3. Run the script:
   ```bash
   python main.py
   ```

Notes:
- SHAP and LIME are optional in the sense that if they're not installed the script will not crash but will skip the respective analysis steps. We still provide the code that uses them so you can reproduce the interpretability analysis locally.
- The synthetic dataset has labeled `default` with a probabilistic generation that gives realistic signal for features like `int_rate`, `dti`, `delinq_2yrs`, etc.
