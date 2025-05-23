import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_excel(r'C:\Users\talantr\Downloads\heart.xlsx')

y = df['target']
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

print('ORDINARY LEAST SQUARES REGRESSION')
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()

print(ols_model.summary())

print('LOGIT MODEL')

logit_model = sm.Logit(y, X_ols).fit()
print(logit_model.summary())

print('CHECKING MULTICOLLINEARITY')

vif_data = pd.DataFrame()
vif_data["feature"] = X_ols.columns
vif_data["VIF"] = [variance_inflation_factor(
    X_ols.values, i) for i in range(X_ols.shape[1])]

print(vif_data)
