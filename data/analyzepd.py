import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\bryan\ap_research\data\results.csv")
game = "prisoner's_dilemma"
print(f"Total PD rows: {len(df[df['game_type'] == game])}")
pd_df = df[df["game_type"] == "prisoner's_dilemma"].copy()
pd_df = pd_df[pd_df["decision"].isin(["A", "B"])].copy()


print(f"Rows with valid decisions: {len(pd_df)}")
print()

pd_df["cooperate"] = (pd_df["decision"] == "A").astype(int)

print("Decision distribution:")
print(pd_df["decision"].value_counts())
print(f"Cooperation rate: {pd_df['cooperate'].mean():.3f}")
print()

print("Cooperation rate by framing condition:")
print(pd_df.groupby("framing_condition")["cooperate"].agg(["mean", "count"]).round(3))
print()

print("Cooperation rate by stake level:")
print(pd_df.groupby("stake_level")["cooperate"].agg(["mean", "count"]).round(3))
print()

print("Cooperation rate by actor type:")
print(pd_df.groupby("actor_type")["cooperate"].agg(["mean", "count"]).round(3))
print()

print("Cooperation rate by model:")
print(pd_df.groupby("model_name")["cooperate"].agg(["mean", "count"]).round(3))
print()

formula = "cooperate ~ C(framing_condition) + C(stake_level) + C(actor_type) + C(model_name)"

model = logit(formula, data=pd_df).fit(maxiter=200, disp=False)

print("=" * 70)
print("BINARY LOGISTIC REGRESSION — PD COOPERATION (A=1, B=0)")
print("=" * 70)
print(model.summary2())
print()

print("=" * 70)
print("ODDS RATIOS WITH 95% CONFIDENCE INTERVALS")
print("=" * 70)
odds = pd.DataFrame({
    "OR":       np.exp(model.params),
    "CI_lower": np.exp(model.conf_int()[0]),
    "CI_upper": np.exp(model.conf_int()[1]),
    "p_value":  model.pvalues,
}).round(4)
odds["significant"] = odds["p_value"].apply(lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")))
print(odds.to_string())
print()

print("=" * 70)
print("MODEL FIT")
print("=" * 70)
print(f"Log-likelihood:       {model.llf:.4f}")
print(f"Null log-likelihood:  {model.llnull:.4f}")
print(f"Chi-squared:          {model.llr:.4f}  (df={model.df_model:.0f})")
print(f"p-value (LR test):    {model.llr_pvalue:.6f}")
print(f"Nagelkerke R²:        {model.prsquared:.4f}")
print(f"AIC:                  {model.aic:.4f}")
print(f"BIC:                  {model.bic:.4f}")
print(f"N:                    {int(model.nobs)}")