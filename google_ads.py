import pandas as pd
from lightgbm import plot_importance
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold, cross_val_score,GridSearchCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor,plot_importance
import shap
import lightgbm as lgb
from lightgbm import LGBMRegressor



df = pd.read_csv("google_ads_full_30days_demo.csv")
df = df.drop_duplicates()

print(df.head())

row_data = df.loc[0:50,"roas"]
print(row_data)

print("BEFORE NULL VALUES :",df.isnull().sum())
print(df.isnull().sum())
print("\nTotal null before:", df.isnull().sum().sum())

# ============================================
# 2️⃣ SAVE BUSINESS IDENTIFIERS (🔥 CRITICAL)
# ============================================
id_df = df[["campaign", "ad_spend", "revenue"]].copy()

# ============================================
# 3️⃣ CLEAN TEXT COLUMNS
# ============================================
text_cols = df.select_dtypes(include="object").columns
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# ============================================
# 4️⃣ HANDLE NULLS (FILLNA)
# ============================================
categorical_cols = [
    "date","type","campaign","ad_group","keyword",
    "device","product_id","title","category","region"
]
df[categorical_cols] = df[categorical_cols].fillna("missing")

numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
imputer = KNNImputer(n_neighbors=3)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print("\nNULL COUNTS AFTER:\n", df.isnull().sum())

# ============================================
# 5️⃣ ONE-HOT ENCODING
# ============================================
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_array = ohe.fit_transform(df[categorical_cols])

ohe_df = pd.DataFrame(
    ohe_array,
    columns=ohe.get_feature_names_out(categorical_cols),
    index=df.index
)

df = pd.concat([df, ohe_df], axis=1)
df = df.drop(columns=categorical_cols)

# ============================================
# 6️⃣ FEATURES & TARGET
# ============================================
X = df.drop(columns=["roas"])
y = df["roas"]

# ============================================
# 7️⃣ TRAIN / TEST SPLIT
# ============================================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 8️⃣ DECISION TREE (BASELINE)
# ============================================
dt = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=3,
    min_samples_split=10,
    criterion="friedman_mse",
    random_state=42
)
dt.fit(x_train, y_train)
print("\nDecision Tree R2:", r2_score(y_test, dt.predict(x_test)))

# ============================================
# 9️⃣ REGULARIZED XGBOOST (MAIN MODEL)
# ============================================
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=20,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=1.0,
    reg_alpha=1.0,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)

print("XGBoost R2:", r2_score(y_test, xgb_pred))

# ---- Cross-Validation (OVERFITTING CHECK)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring="r2")

print("XGB CV Scores:", cv_scores)
print("XGB CV Mean:", cv_scores.mean())
print("XGB CV Std :", cv_scores.std())

# ============================================
# LIGHTGBM (LOG ROAS – COMPARISON)
# ============================================
y_train_log = np.log1p(y_train)

lgb_model = LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_samples=50,
    random_state=42
)

lgb_model.fit(x_train, y_train_log)
lgb_pred = np.expm1(lgb_model.predict(x_test))

print("LightGBM R2:", r2_score(y_test, lgb_pred))


#FINAL MODEL SELECTION
# ===========================================
best_model = "XGBoost"
final_pred = xgb_pred
print("\nBest Model Selected:", best_model)


#  BUSINESS DECISION LOGIC
# ============================================
def ad_decision(roas):
    if roas < 1:
        return "STOP"
    elif roas < 3:
        return "HOLD"
    else:
        return "SCALE"

final_df = x_test.copy()
final_df["Actual_ROAS"] = y_test.values
final_df["Predicted_ROAS"] = final_pred
final_df["Decision"] = final_df["Predicted_ROAS"].apply(ad_decision)

final_df["campaign"] = id_df.loc[final_df.index, "campaign"]
final_df["ad_spend"] = id_df.loc[final_df.index, "ad_spend"]
final_df["revenue"] = id_df.loc[final_df.index, "revenue"]

print("\nROW LEVEL SAMPLE:")
print(final_df[["campaign","Actual_ROAS","Predicted_ROAS","Decision"]].head())

# ============================================
# 1 CAMPAIGN LEVEL AGGREGATION
# ============================================
campaign_level = (
    final_df
    .groupby("campaign")
    .agg(
        total_spend=("ad_spend","sum"),
        total_revenue=("revenue","sum"),
        avg_pred_roas=("Predicted_ROAS","mean")
    )
    .reset_index()
)

campaign_level["campaign_roas"] = (
    campaign_level["total_revenue"] / campaign_level["total_spend"]
)
campaign_level["Decision"] = campaign_level["campaign_roas"].apply(ad_decision)

print("\nCAMPAIGN LEVEL DECISIONS:")
print(campaign_level)

# ============================================
# 1️⃣4️⃣ SHAP EXPLAINABILITY
# ============================================
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test)

shap_df = pd.DataFrame(
    shap_values,
    columns=x_test.columns,
    index=x_test.index
)

# 🔥 IMPORTANT: keep campaign separate
shap_df["campaign"] = final_df["campaign"]

# ---- GLOBAL SHAP
shap.summary_plot(shap_values, x_test)
shap.summary_plot(shap_values, x_test, plot_type="bar")

# ============================================
# 1️⃣5️⃣ CAMPAIGN-LEVEL SHAP (FIXED)
# ============================================
shap_features_only = shap_df.drop(columns=["campaign"])

campaign_shap = (
    shap_features_only
    .join(final_df["campaign"])
    .groupby("campaign")
    .mean()
)

campaign_name = campaign_shap.index[0]

top_positive = campaign_shap.loc[campaign_name].sort_values(ascending=False).head(10)
top_negative = campaign_shap.loc[campaign_name].sort_values().head(10)

print(f"\nSHAP – ROAS BOOSTERS for {campaign_name}")
print(top_positive)

print(f"\nSHAP – ROAS KILLERS for {campaign_name}")
print(top_negative)

# ============================================
# 1️⃣6️⃣ SHAP-BASED FEATURE REDUCTION (NO ERROR)
# ============================================
important_features = (
    shap_features_only
    .abs()
    .mean()
    .sort_values(ascending=False)
    .head(50)
    .index
)

X_reduced = X[important_features]

print("\nTop 50 Features Selected Using SHAP")
print(important_features)

# ============================================
# 🟢 IMAGE 1 – FIVERR HOOK VISUAL
# STOP / HOLD / SCALE DECISION PIE
# ============================================

# Count decisions
decision_counts = final_df["Decision"].value_counts()

# Business colors (psychology-based)
decision_colors = {
    "STOP": "#e74c3c",   # Red → Stop wasting money
    "HOLD": "#f1c40f",   # Yellow → Watch carefully
    "SCALE": "#2ecc71"   # Green → Profitable growth
}

colors = [decision_colors[d] for d in decision_counts.index]

plt.figure(figsize=(7, 7))

plt.pie(
    decision_counts,
    labels=decision_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)

plt.title(
    "AI Budget Decision System\nSTOP • HOLD • SCALE",
    fontsize=16,
    fontweight="bold"
)

plt.text(
    0, -1.25,
    " tell you where to STOP wasting money\nand where to SCALE profitably",
    ha="center",
        fontsize=11
)

plt.tight_layout()
plt.show()

# ============================================
#  CAMPAIGN LEVEL ACTION BOARD (CLIENT FAVORITE)
# ============================================

# Sort campaigns by ROAS
campaign_vis = campaign_level.sort_values("campaign_roas")

# Color mapping for decisions
decision_color_map = {
    "STOP": "#e74c3c",   # Red
    "HOLD": "#f1c40f",   # Yellow
    "SCALE": "#2ecc71"   # Green
}

colors = campaign_vis["Decision"].map(decision_color_map)

plt.figure(figsize=(10, 6))

plt.barh(
    campaign_vis["campaign"],
    campaign_vis["campaign_roas"],
    color=colors
)

plt.axvline(1, color="black", linestyle="--", linewidth=1)
plt.axvline(3, color="black", linestyle="--", linewidth=1)

plt.xlabel("ROAS")
plt.title("Campaign Action Board: STOP | HOLD | SCALE", fontsize=14, fontweight="bold")

# Add labels on bars
for index, row in campaign_vis.iterrows():
    plt.text(
        row["campaign_roas"] + 0.05,
        campaign_vis.index.get_loc(index),
        row["Decision"],
        va="center",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("fiverr_campaign_action_board.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# 3D Campaign Clustering for Technical Clients
# STOP | HOLD | SCALE
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
df = pd.read_csv("google_ads_full_30days_demo.csv")
df = df.drop_duplicates()

# ----------------------------------------------------------
# 2. CAMPAIGN LEVEL AGGREGATION
# ----------------------------------------------------------
campaign_level = (
    df.groupby("campaign")
      .agg(
          total_spend=("ad_spend", "sum"),
          total_revenue=("revenue", "sum"),
          avg_roas=("roas", "mean")
      )
      .reset_index()
)

campaign_level["campaign_roas"] = (
    campaign_level["total_revenue"] / campaign_level["total_spend"]
)

# ----------------------------------------------------------
# 3. FEATURES FOR CLUSTERING
# (More technical = more dimensions)
# ----------------------------------------------------------
X = campaign_level[
    ["total_spend", "total_revenue", "campaign_roas", "avg_roas"]
]

# ----------------------------------------------------------
# 4. FEATURE SCALING (CRITICAL)
# ----------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------
# 5. KMEANS CLUSTERING
# ----------------------------------------------------------
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=30,
    random_state=42
)

campaign_level["cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------
# 6. CLUSTER VALIDATION
# ----------------------------------------------------------
sil_score = silhouette_score(X_scaled, campaign_level["cluster"])
print(f"Silhouette Score: {sil_score:.3f}")

# ----------------------------------------------------------
# 7. MAP CLUSTERS → BUSINESS DECISION
# ----------------------------------------------------------
cluster_order = (
    campaign_level
    .groupby("cluster")["campaign_roas"]
    .mean()
    .sort_values()
    .index
)

cluster_map = {
    cluster_order[0]: "STOP",
    cluster_order[1]: "HOLD",
    cluster_order[2]: "SCALE"
}

campaign_level["Decision"] = campaign_level["cluster"].map(cluster_map)

print("\nCampaign Decisions:")
print(campaign_level[["campaign", "campaign_roas", "Decision"]])

# ----------------------------------------------------------
# 8. PCA → 3D PROJECTION
# ----------------------------------------------------------
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

campaign_level["PC1"] = X_pca[:, 0]
campaign_level["PC2"] = X_pca[:, 1]
campaign_level["PC3"] = X_pca[:, 2]

print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# ----------------------------------------------------------
# 9. 3D CLUSTER VISUALIZATION
# ----------------------------------------------------------
color_map = {
    "STOP": "#e74c3c",   # Red
    "HOLD": "#f1c40f",   # Yellow
    "SCALE": "#2ecc71"   # Green
}

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

for decision in ["STOP", "HOLD", "SCALE"]:
    subset = campaign_level[campaign_level["Decision"] == decision]
    ax.scatter(
        subset["PC1"],
        subset["PC2"],
        subset["PC3"],
        s=150,
        c=color_map[decision],
        label=decision,
        edgecolors="black",
        depthshade=True
    )

# Campaign name labels
for _, row in campaign_level.iterrows():
    ax.text(
        row["PC1"],
        row["PC2"],
        row["PC3"],
        row["campaign"],
        fontsize=8
    )

ax.set_title(
    "3D Campaign Clustering (PCA Space)\nSTOP | HOLD | SCALE",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

ax.legend()
plt.tight_layout()

plt.savefig(
    "campaign_3d_cluster_stop_hold_scale.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ----------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------
