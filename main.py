import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t, norm
from statsmodels.graphics.tsaplots import plot_acf



# Step 1. Import data
df = pd.read_csv("PAYEMS.csv")
df.columns = ["Date", "Employment"]
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")
# Sample: January 1974 - December 2025
df = df.loc["1974-01-01":"2025-12-01"].copy()

# Step 2. Calculate growth rate and plot
df["Growth_Rate"] = df["Employment"].pct_change(periods=12) * 100

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Employment level
axes[0].plot(df.index, df["Employment"], color="black", linewidth=1.0)
axes[0].set_title("US employment")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("")

# Year-on-year employment growth rate
axes[1].plot(df.index, df["Growth_Rate"], color="black", linewidth=1.0)
axes[1].axhline(y=0, color="red", linewidth=1)
axes[1].set_title("Year-on-year employment growth rate")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("")

# Grid
for ax in axes:
    ax.grid(True, color="#eeeeee", linewidth=0.8)

plt.tight_layout()
plt.savefig("employment_growth_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# Step 3. OLS estimation of AR(3)
Y = df["Growth_Rate"].dropna().to_numpy()
p = 3
ytemp = Y[p:]
X = np.column_stack([
    np.ones(len(ytemp)),
    Y[p-1:-1],
    Y[p-2:-2],
    Y[p-3:-3]])

BETA = np.linalg.solve(X.T @ X, X.T @ ytemp)
u = ytemp - X @ BETA
SSE = np.sum(u**2)
T = X.shape[0]      # effective regression sample
K = X.shape[1]      # number of estimated coefficients
degrees_of_freedom = T - K

print("\nOLS sample information:")
print("T =", T)
print("K =", K)
print("T-K =", degrees_of_freedom)

SIGMA = SSE / (T - K)       # Standard OLS residual variance estimator
varbeta = SIGMA * np.linalg.inv(X.T @ X)
stdbeta = np.sqrt(np.diag(varbeta))
tvals = BETA / stdbeta
pvals = 2 * (1 - t.cdf(np.abs(tvals),df=degrees_of_freedom))    # p-values

OLS_OUTPUT = pd.DataFrame({
    "coefficient": [
        "Intercept",
        "LAG 1",
        "LAG 2",
        "LAG 3"],
    "estimate": np.round(BETA, 2),
    "std_error": np.round(
        stdbeta,
        2
    ),
    "tstat": np.round(
        tvals,
        2
    ),
    "p_value": np.round(
        pvals,
        7
    )
})

print("\nOLS results:")
print(OLS_OUTPUT)
print("\nResidual variance:")
print(round(SIGMA, 4))


# Step 4. Forecast
CONST = np.array([[BETA[0]],[0],[0]])
PHI = np.array([
    [BETA[1], BETA[2], BETA[3]],
    [1,        0,       0],
    [0,        1,       0]
])
YTEMP = np.array([
    [ytemp[-1]],
    [ytemp[-2]],
    [ytemp[-3]]
])
horz = 24

FORECAST_MATRIX = np.zeros((3, horz))
# One-step-ahead forecast
FORECAST_MATRIX[:, 0:1] = (CONST + PHI @ YTEMP)
# Forecast horizons 2 to 24
for i in range(1, horz):
    FORECAST_MATRIX[:, i:i+1] = (
        CONST
        + PHI @ FORECAST_MATRIX[:, i-1:i]
    )
FORECAST = FORECAST_MATRIX[0, :]
MSFE = np.full((3, horz),np.nan)
RMSFE_MATRIX = np.full((3, horz),np.nan)
MSFE[:, 0] = np.array([SIGMA,0,0])
RMSFE_MATRIX[:, 0] = np.sqrt(MSFE[:, 0])
for i in range(1, horz):
    PHI_PWR = np.linalg.matrix_power(PHI,i)
    MSFE[:, i] = ((PHI_PWR[:, 0] ** 2) * SIGMA+ MSFE[:, i-1])
    RMSFE_MATRIX[:, i] = np.sqrt(MSFE[:, i])

RMSFE = RMSFE_MATRIX[0, :]

TEMP = df.index[-1]
FORECAST_PERIOD = pd.date_range(
    start=TEMP,
    periods=horz + 1,
    freq="MS")[1:]

# 68% forecast interval
LB_16 = (FORECAST- norm.ppf(0.84) * RMSFE)
UB_84 = (FORECAST+ norm.ppf(0.84) * RMSFE)
# 90% forecast interval
LB_10 = (FORECAST- norm.ppf(0.95) * RMSFE)
UB_90 = (FORECAST+ norm.ppf(0.95) * RMSFE)
# 95% forecast interval
LB_5 = (FORECAST- norm.ppf(0.975) * RMSFE)
UB_95 = (FORECAST+ norm.ppf(0.975) * RMSFE)

OUTPUT_FORECAST = pd.DataFrame({
    "Date": FORECAST_PERIOD,
    "FORECAST": FORECAST,
    "LB_16": LB_16,
    "UB_84": UB_84,
    "LB_10": LB_10,
    "UB_90": UB_90,
    "LB_5": LB_5,
    "UB_95": UB_95
})

print("\nForecast results:")
print(OUTPUT_FORECAST)

fig, ax = plt.subplots(figsize=(12, 7))
# 95% forecast interval
ax.fill_between(
    FORECAST_PERIOD,
    LB_5,
    UB_95,
    color="#fc8c8c",
    alpha=0.3,
    label="95% Forecast Interval"
)
# 90% forecast interval
ax.fill_between(
    FORECAST_PERIOD,
    LB_10,
    UB_90,
    color="#fb4646",
    alpha=0.3,
    label="90% Forecast Interval")

# 68% forecast interval
ax.fill_between(
    FORECAST_PERIOD,
    LB_16,
    UB_84,
    color="#d10404",
    alpha=0.3,
    label="68% Forecast Interval")

# Actual data
ax.plot(
    df.index,
    df["Growth_Rate"],
    color="black",
    linewidth=1.2,
    label="Actual")

# Forecast
ax.plot(
    FORECAST_PERIOD,
    FORECAST,
    color="#d10404",
    linewidth=1.2,
    label="Forecast")

# Zero line
ax.axhline(
    y=0,
    color="black",
    linewidth=0.8)

# Plot formatting
ax.set_title(
    "Forecast of US employment growth rate",
    fontsize=14)

ax.set_xlabel("Years")
ax.set_ylabel("Percent")

ax.grid(
    True,
    color="#eeeeee",
    linewidth=0.8
)

ax.legend(
    loc="upper left"
)

plt.tight_layout()
plt.savefig("forecast.png", dpi=300, bbox_inches="tight")
plt.show()