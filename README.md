# 🎓 Graduate Admission Predictor

ML web app that estimates a student's chance of graduate admission from GRE, TOEFL,
university rating, SOP, LOR, CGPA and research experience.

**Stack:** Python, scikit-learn, pandas, Streamlit

## Run locally
```bash
pip install -r requirements.txt
python train.py          # trains, compares 4 models, saves model.pkl + metrics.json
streamlit run app.py
```

## Modelling notes
- Dropped `Serial No.` (row ID, no predictive meaning)
- Stripped trailing spaces from column names
- 80/20 split; models compared with 5-fold CV (Linear, Ridge, Lasso, Random Forest)
- Best model refit on all data for deployment

## Deploy (free)
Push to GitHub → share.streamlit.io → New app → select `app.py`.
Commit `model.pkl`, `metrics.json` and `Admission_Predict.csv`.
