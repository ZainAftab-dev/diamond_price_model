# 💎 Diamond Price Predictor

**Live app: https://zainaftab-diamond-price.streamlit.app**

Web app that predicts the price of a diamond from its carat, cut, colour, clarity and size.

- **Model:** Extra Trees regressor (scikit-learn pipeline with imputation, scaling and one-hot encoding)
- **Selection:** best of 5 algorithms (Ridge Regression, Linear SVR, Neural Network, Extra Trees, AdaBoost) compared with 10-fold cross validation on 53,794 diamonds
- **Cross-validation RMSE:** $536 ± $24 (13.6% of the average price)

AIC354 Machine Learning Fundamentals, Assignment 3 · Zain Aftab (FA24-BSE-155)

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app |
| `diamond_price_model.joblib` | Trained model pipeline |
| `model_info.json` | Algorithm name, cross-validation RMSE, input ranges |
| `requirements.txt` | Exact library versions used to train the model |
| `notebook/` | Colab notebook (10-fold cross validation) and the dataset |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Dataset: `diamonds` from the ggplot2 R package, via https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv
