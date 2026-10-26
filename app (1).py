"""
Student Service Request Mining - Streamlit app
Converted from SectionB_Student_Service_Request_Mining.ipynb

Run with:  streamlit run app.py
"""
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.patches import Ellipse, Rectangle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score, classification_report,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Student Service Request Mining", layout="wide")

EXPECTED = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score",
]
COLUMN_ALIASES = {c.lower(): c for c in EXPECTED}

CATEGORY_MAPS = {
    "Status": {
        "complete": "Completed", "completed": "Completed", "done": "Completed",
        "pending": "Pending", "open": "Pending",
        "cancelled": "Cancelled", "canceled": "Cancelled",
    },
    "Priority": {"low": "Low", "normal": "Normal", "medium": "Normal", "high": "High"},
    "Required_Documents_Complete": {
        "yes": "Yes", "y": "Yes", "true": "Yes", "complete": "Yes",
        "no": "No", "n": "No", "false": "No", "incomplete": "No",
    },
    "Reopened": {
        "yes": "Yes", "y": "Yes", "true": "Yes",
        "no": "No", "n": "No", "false": "No",
    },
}

CANDIDATE_FEATURES = [
    "Service_Type", "Requester_Type", "Submission_Weekday",
    "Requested_Channel", "Priority", "Assigned_Office",
    "Queue_Length", "Queue_Category", "Required_Documents_Complete", "SLA_Days",
]


# ----------------------------------------------------------------------------
# Data processing (cached)
# ----------------------------------------------------------------------------
def standardize_column_name(name):
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip())
    return name.strip("_")


@st.cache_data(show_spinner=False)
def load_and_standardize(file_bytes):
    import io
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df = df_raw.copy()
    df.columns = [standardize_column_name(c) for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES[c.lower()] for c in df.columns if c.lower() in COLUMN_ALIASES})
    return df_raw, df


@st.cache_data(show_spinner=False)
def clean_data(df):
    log = []

    def log_issue(issue, count, treatment):
        log.append({"Issue": issue, "Rows_or_cases_affected": int(count), "Treatment": treatment})

    clean = df.copy()

    for col in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[col]) and not pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].astype(object).map(
                lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x
            )

    if "Submission_Date" in clean.columns:
        clean["Submission_Date"] = pd.to_datetime(clean["Submission_Date"], errors="coerce")
        log_issue("Missing or invalid Submission_Date", clean["Submission_Date"].isna().sum(),
                  "Converted to datetime; invalid/missing dates remain NaT and are excluded from weekday feature.")

    for col in ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]:
        if col in clean.columns:
            before = clean[col].notna().sum()
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
            newly_invalid = int(before - clean[col].notna().sum())
            if newly_invalid:
                log_issue(f"Non-numeric values in {col}", newly_invalid,
                          "Coerced invalid numeric strings to missing values.")

    if "Request_ID" in clean.columns:
        dup = clean["Request_ID"].notna() & clean["Request_ID"].duplicated(keep="first")
        log_issue("Duplicate Request_ID values", dup.sum(),
                  "Kept the first record for each non-missing Request_ID; removed later duplicates.")
        clean = clean.loc[~dup].copy()
    else:
        log_issue("Request_ID column unavailable", 0,
                  "Identifier duplicate check could not be performed; verify the source schema.")

    for col, mapping in CATEGORY_MAPS.items():
        if col in clean.columns:
            before = clean[col].copy()
            clean[col] = clean[col].map(
                lambda x: mapping.get(str(x).strip().lower(), x) if pd.notna(x) else x
            )
            changed = (before.astype(str) != clean[col].astype(str)).sum()
            log_issue(f"Inconsistent category labels in {col}", changed,
                      "Mapped common spelling/case variants to standard labels; unrecognized labels retained.")

    for col in ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]:
        if col in clean.columns:
            neg = clean[col] < 0
            log_issue(f"Negative values in {col}", neg.sum(),
                      "Set negative values to missing because they are invalid for this field.")
            clean.loc[neg, col] = np.nan

    return clean, pd.DataFrame(log)


@st.cache_data(show_spinner=False)
def engineer_features(clean):
    clean = clean.copy()
    if "Submission_Date" in clean.columns:
        clean["Submission_Weekday"] = clean["Submission_Date"].dt.day_name()
    else:
        clean["Submission_Weekday"] = np.nan

    if "Queue_Length" in clean.columns:
        clean["Queue_Category"] = pd.cut(
            clean["Queue_Length"], bins=[-np.inf, 5, 15, np.inf],
            labels=["Low", "Medium", "High"], right=True,
        )
    else:
        clean["Queue_Category"] = np.nan

    clean["SLA_Breached"] = np.nan
    if {"Status", "Processing_Days", "SLA_Days"}.issubset(clean.columns):
        mask = (
            clean["Status"].astype(str).str.lower().eq("completed")
            & clean["Processing_Days"].notna()
            & clean["SLA_Days"].notna()
        )
        clean.loc[mask, "SLA_Breached"] = (
            clean.loc[mask, "Processing_Days"] > clean.loc[mask, "SLA_Days"]
        ).astype(int)

    labeled = clean.loc[clean["SLA_Breached"].notna()].copy()
    if not labeled.empty:
        labeled["SLA_Breached"] = labeled["SLA_Breached"].astype(int)
    return clean, labeled


def score_binary(y_true, pred):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision_Breach_Class_1": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "Recall_Breach_Class_1": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "F1_Breach_Class_1": f1_score(y_true, pred, pos_label=1, zero_division=0),
    }


def train_model(labeled, max_depth, min_leaf, test_size, seed):
    features = [c for c in CANDIDATE_FEATURES if c in labeled.columns]
    target = "SLA_Breached"

    if len(labeled) < 10:
        return {"error": "Not enough labeled rows for a reliable evaluation (at least 10 required)."}
    if labeled[target].nunique() < 2:
        return {"error": "The target contains only one class; a binary classifier cannot be evaluated."}
    if len(features) < 5:
        return {"error": "Fewer than five submission-time predictors are available. Check the CSV schema."}

    X = labeled[features].copy()
    y = labeled[target].astype(int)

    cat_features = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
    num_features = [c for c in features if c not in cat_features]
    for c in cat_features:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in num_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
    ])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    model = Pipeline([
        ("preprocessor", pre),
        ("classifier", DecisionTreeClassifier(
            random_state=seed, max_depth=max_depth, min_samples_leaf=min_leaf)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)

    tree_scores = score_binary(y_test, y_pred)
    base_scores = score_binary(y_test, base_pred)
    evaluation = pd.DataFrame([tree_scores, base_scores],
                              index=["Decision Tree", "Majority-Class Baseline"])

    importance = pd.DataFrame({
        "Encoded_Feature": model.named_steps["preprocessor"].get_feature_names_out(),
        "Importance": model.named_steps["classifier"].feature_importances_,
    }).sort_values("Importance", ascending=False)

    return {
        "features": features, "y_test": y_test, "y_pred": y_pred,
        "tree_scores": tree_scores, "baseline_scores": base_scores,
        "evaluation": evaluation, "importance": importance,
        "report": classification_report(
            y_test, y_pred, labels=[0, 1],
            target_names=["Not breached (0)", "Breached (1)"], zero_division=0),
    }


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def hbar(series, title, xlabel, ylabel, fmt="{:.0f}", figsize=(10, 5), offset=1):
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(kind="barh", ax=ax)
    for bar in ax.patches:
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                fmt.format(bar.get_width()), va="center", ha="left",
                fontsize=10, fontweight="bold")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def use_case_diagram():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10.4); ax.axis("off")
    ax.add_patch(Rectangle((2.4, 0.5), 9.2, 9.4, fill=False, lw=2, edgecolor="#333"))
    ax.text(7.0, 9.55, "Student Service Request Monitoring System",
            ha="center", fontsize=11, fontweight="bold")

    OLD, NEW = "#DCE8F7", "#D6F0DC"
    use_cases = {
        "[1] Submit Service Request": (4.6, 8.4, OLD),
        "[2] Track Request Status": (4.6, 6.8, OLD),
        "[3] Update Request Status": (4.6, 2.4, OLD),
        "[4] View SLA Performance": (9.0, 8.4, NEW),
        "[5] Generate Analytics Report": (9.0, 6.9, NEW),
        "[6] Review Delay Risk": (9.0, 5.3, NEW),
        "[7] Monitor Queue Status": (9.0, 3.4, NEW),
    }
    actors = {
        "(A1) Student / Requester": (0.9, 7.6),
        "(A2) Service Staff": (0.9, 3.4),
        "(A3) Administrator": (13.0, 6.8),
    }

    def draw_actor(x, y, label):
        ax.add_patch(plt.Circle((x, y + 0.75), 0.2, fill=False, lw=1.8))
        ax.plot([x, x], [y + 0.55, y - 0.1], "k", lw=1.8)
        ax.plot([x - 0.35, x + 0.35], [y + 0.35, y + 0.35], "k", lw=1.8)
        ax.plot([x, x - 0.3], [y - 0.1, y - 0.6], "k", lw=1.8)
        ax.plot([x, x + 0.3], [y - 0.1, y - 0.6], "k", lw=1.8)
        ax.text(x, y - 0.95, label, ha="center", fontsize=9.5, fontweight="bold")

    for name, (x, y) in actors.items():
        draw_actor(x, y, name)
    for name, (x, y, color) in use_cases.items():
        ax.add_patch(Ellipse((x, y), 3.2, 1.1, facecolor=color, edgecolor="#333", lw=1.4, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=8, zorder=4, fontweight="bold")

    def associate(actor, uc, dx):
        ax0, ay0 = actors[actor]
        ux, uy, _ = use_cases[uc]
        ax.plot([ax0 + dx, ux], [ay0 + 0.3, uy], color="#333", lw=1.2, zorder=1)

    for uc in ["[1] Submit Service Request", "[2] Track Request Status"]:
        associate("(A1) Student / Requester", uc, 0.35)
    for uc in ["[3] Update Request Status", "[7] Monitor Queue Status", "[6] Review Delay Risk"]:
        associate("(A2) Service Staff", uc, 0.35)
    for uc in ["[4] View SLA Performance", "[5] Generate Analytics Report", "[6] Review Delay Risk"]:
        associate("(A3) Administrator", uc, -0.35)

    ax.add_patch(Rectangle((2.7, 0.7), 0.35, 0.25, facecolor=OLD, edgecolor="#333"))
    ax.text(3.15, 0.74, "Existing Use Cases (1-3)", fontsize=8.5)
    ax.add_patch(Rectangle((5.5, 0.7), 0.35, 0.25, facecolor=NEW, edgecolor="#333"))
    ax.text(5.95, 0.74, "New Analytics Use Cases (4-7)", fontsize=8.5)
    ax.set_title("Figure 6: UML Use Case Diagram with Numbered Elements",
                 fontsize=12, fontweight="bold")
    return fig


def to_csv_bytes(df, index=False):
    return df.to_csv(index=index).encode("utf-8")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("Student Service Request Mining")
st.caption("SLA breach analysis and Decision Tree risk model for student service requests.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Upload CSV (e.g. student_service_requests.csv)", type=["csv"])

    st.header("Model settings")
    max_depth = st.slider("Max tree depth", 1, 15, 5)
    min_leaf = st.slider("Min samples per leaf", 1, 20, 3)
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, step=0.05)
    seed = st.number_input("Random seed", value=42, step=1)

if uploaded is None:
    st.info("Upload a CSV file in the sidebar to begin.")
    st.stop()

df_raw, df = load_and_standardize(uploaded.getvalue())
clean, quality_log_df = clean_data(df)
clean, labeled = engineer_features(clean)

tabs = st.tabs([
    "1. Data overview", "2. Dictionary & frequencies", "3. Cleaning log",
    "4. Features & target", "5. EDA", "6. Decision Tree",
    "7. Requirements", "8. Use case diagram", "9. Dashboard", "10. Reflection",
])

# ---------------- Tab 1 ----------------
with tabs[0]:
    st.subheader("Loaded file")
    st.write(f"**File:** {uploaded.name}  |  **Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head(10))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data types**")
        st.dataframe(df.dtypes.astype(str).rename("Data_Type").to_frame())
    with c2:
        st.markdown("**Missing values**")
        st.dataframe(df.isna().sum().rename("Missing_Count").to_frame())

    missing_expected = [c for c in EXPECTED if c not in df.columns]
    if missing_expected:
        st.warning(f"Expected columns not found: {missing_expected}. "
                   "If these are naming differences, rename them in the CSV.")
    else:
        st.success("All expected columns found.")

    st.markdown("#### Programmatic collection method")
    st.markdown(
        "The student-services portal could store each request in a relational database. "
        "An authorized analytics script could retrieve records through a secured API endpoint or a "
        "read-only database connection. The process should authenticate securely, select only required "
        "fields, validate dates and values, log extraction time, and avoid exposing unnecessary personal "
        "information. The CSV can then be exported for analysis. This app analyzes the supplied CSV; it "
        "does not connect to a live university portal."
    )

# ---------------- Tab 2 ----------------
with tabs[1]:
    st.subheader("Category frequencies (raw standardized data)")
    cat_cols = [c for c in ["Service_Type", "Requester_Type", "Requested_Channel", "Priority",
                            "Assigned_Office", "Required_Documents_Complete", "Status", "Reopened"]
                if c in df.columns]
    cols = st.columns(2)
    for i, col in enumerate(cat_cols):
        with cols[i % 2]:
            st.markdown(f"**{col}**")
            st.dataframe(df[col].fillna("[Missing]").astype(str).value_counts()
                         .rename("Count").to_frame())

    st.subheader("Data dictionary")
    dictionary = pd.DataFrame([
        ("Request_ID", "Unique transaction identifier", "Identifier"),
        ("Service_Type", "Type of service requested", "Categorical predictor"),
        ("Requester_Type", "Student, Faculty, or Staff", "Categorical predictor"),
        ("Submission_Date", "Date request was lodged", "Date / feature source"),
        ("Requested_Channel", "Portal, Email, Walk-in, etc.", "Categorical predictor"),
        ("Priority", "Request priority", "Categorical predictor"),
        ("Assigned_Office", "Office processing the request", "Categorical predictor"),
        ("Queue_Length", "Open requests at submission", "Numerical predictor"),
        ("Required_Documents_Complete", "Whether required documents are complete", "Categorical predictor"),
        ("Processing_Days", "Actual days to completion", "Outcome field; NOT a predictor"),
        ("SLA_Days", "Target days allowed for the request", "Numerical predictor if known at submission"),
        ("Status", "Completed, Pending, Cancelled", "Eligibility filter; NOT a predictor"),
        ("Reopened", "Whether the request was reopened", "Outcome field; NOT a predictor"),
        ("Satisfaction_Score", "Optional post-service rating", "Outcome field; NOT a predictor"),
    ], columns=["Field", "Meaning", "Role"])
    dictionary = dictionary[dictionary["Field"].isin(df.columns)]
    st.dataframe(dictionary, hide_index=True)
    st.download_button("Download data_dictionary.csv", to_csv_bytes(dictionary),
                       "data_dictionary.csv", "text/csv")

# ---------------- Tab 3 ----------------
with tabs[2]:
    st.subheader("Data-quality checks and cleaning log")
    st.dataframe(quality_log_df, hide_index=True)
    st.write(f"Original shape: **{df.shape}**  →  Cleaned shape: **{clean.shape}**")
    st.markdown("**Categories after cleaning**")
    for col in cat_cols:
        if col in clean.columns:
            st.write(f"`{col}`: {sorted(clean[col].dropna().astype(str).unique().tolist())}")
    st.download_button("Download data_quality_log.csv", to_csv_bytes(quality_log_df),
                       "data_quality_log.csv", "text/csv")

# ---------------- Tab 4 ----------------
with tabs[3]:
    st.subheader("Feature engineering and labeled modeling subset")
    st.markdown(
        "- `Submission_Weekday` derived from `Submission_Date`\n"
        "- `Queue_Category`: Low (≤5), Medium (6–15), High (>15)\n"
        "- `SLA_Breached` = 1 if `Processing_Days` > `SLA_Days`, labeled only for **Completed** requests "
        "with known processing and SLA days"
    )
    m1, m2 = st.columns(2)
    m1.metric("Rows in cleaned dataset", len(clean))
    m2.metric("Eligible completed labeled records", len(labeled))
    if labeled.empty:
        st.warning("No eligible completed records with known Processing_Days and SLA_Days.")
    else:
        counts = (labeled["SLA_Breached"].value_counts().sort_index()
                  .rename(index={0: "Not breached (0)", 1: "Breached (1)"}).to_frame("Count"))
        st.dataframe(counts)

# ---------------- Tab 5 ----------------
office_stats = pd.DataFrame()
queue_stats = pd.DataFrame()

with tabs[4]:
    st.subheader("Exploratory Data Analysis")

    # Q1
    if "Service_Type" in clean.columns and clean["Service_Type"].notna().any():
        svc = clean["Service_Type"].fillna("Unknown").value_counts()
        st.markdown(f"**Q1. Service type with most requests:** {svc.index[0]} ({int(svc.iloc[0])})")
        c1, c2 = st.columns([1, 2])
        c1.dataframe(svc.rename("Request_Count").to_frame())
        c2.pyplot(hbar(svc.sort_values(), "Figure 1: Number of Requests by Service Type",
                       "Number of Requests", "Service Type"))
        st.info(f"{svc.index[0]} has the highest observed volume ({int(svc.iloc[0])} requests). "
                "Administrators can review staffing and workload for this service before deciding "
                "whether capacity changes are needed.")
    else:
        st.warning("Service_Type column missing or empty.")

    # Q2
    if not labeled.empty and "Assigned_Office" in labeled.columns:
        office_stats = labeled.groupby("Assigned_Office", dropna=False)["SLA_Breached"].agg(
            Requests="count", Breaches="sum", Breach_Rate="mean"
        ).sort_values("Breach_Rate", ascending=False)
        office_stats["Breach_Rate_Percent"] = (office_stats["Breach_Rate"] * 100).round(2)
        st.markdown("**Q2. Office with highest SLA breach rate (completed requests)**")
        st.dataframe(office_stats)
        st.pyplot(hbar(office_stats["Breach_Rate"].sort_values() * 100,
                       "Figure 3: SLA Breach Rate by Assigned Office",
                       "SLA Breach Rate (%)", "Assigned Office", fmt="{:.1f}%",
                       figsize=(10, max(4, 0.4 * len(office_stats)))))
        st.info(f"{office_stats.index[0]} has the highest observed breach rate among the offices shown "
                f"({office_stats.iloc[0]['Breach_Rate_Percent']:.2f}%). Check its request count and case "
                "mix before concluding that office performance alone explains delays.")

    # Q3
    if not labeled.empty and "Queue_Category" in labeled.columns:
        queue_stats = labeled.groupby("Queue_Category", observed=False)["SLA_Breached"].agg(
            Requests="count", Breaches="sum", Breach_Rate="mean"
        )
        queue_stats["Breach_Rate_Percent"] = (queue_stats["Breach_Rate"] * 100).round(2)
        st.markdown("**Q3. Queue category vs. breach rate**")
        st.dataframe(queue_stats)

        fig, ax = plt.subplots(figsize=(7, 4))
        (queue_stats["Breach_Rate"] * 100).plot(kind="bar", ax=ax)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", va="bottom", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_title("Figure 2: SLA Breach Rate by Queue Category")
        ax.set_xlabel("Queue Category"); ax.set_ylabel("SLA Breach Rate (%)")
        plt.xticks(rotation=0); fig.tight_layout()
        st.pyplot(fig)

        nonempty = queue_stats[queue_stats["Requests"] > 0]
        if not nonempty.empty:
            top = nonempty["Breach_Rate"].idxmax()
            st.info(f"The {top} queue category has the highest observed breach rate "
                    f"({nonempty.loc[top, 'Breach_Rate_Percent']:.2f}%) among categories with labeled "
                    "records. This association may support queue monitoring, but does not by itself "
                    "prove causation.")

# ---------------- Tab 6 ----------------
model_results = {}
with tabs[5]:
    st.subheader("Decision Tree using submission-time predictors only")
    st.caption("Processing_Days, Status, Reopened, Satisfaction_Score, Request_ID and SLA_Breached "
               "are excluded as predictors to avoid data leakage.")

    if labeled.empty:
        st.warning("No labeled records available for modeling.")
    else:
        result = train_model(labeled, max_depth, min_leaf, test_size, int(seed))
        if "error" in result:
            st.warning(result["error"])
        else:
            model_results = result
            st.write("**Predictors used:**", result["features"])

            st.markdown("**Evaluation (held-out test set)**")
            st.dataframe(result["evaluation"].round(3))

            c1, c2 = st.columns(2)
            with c1:
                cm = confusion_matrix(result["y_test"], result["y_pred"], labels=[0, 1])
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm, display_labels=["Not breached (0)", "Breached (1)"]).plot(ax=ax)
                ax.set_title("Figure 4: Decision Tree Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)
            with c2:
                st.markdown("**Classification report**")
                st.code(result["report"])

            imp = result["importance"]
            st.markdown("**Top 10 feature importances**")
            top_imp = imp.head(10).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(top_imp["Encoded_Feature"], top_imp["Importance"])
            for bar in ax.patches:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.3f}", va="center", ha="left",
                        fontsize=10, fontweight="bold")
            ax.set_title("Figure 5: Top Decision Tree Feature Importances")
            ax.set_xlabel("Importance"); ax.set_ylabel("Encoded feature")
            fig.tight_layout()
            st.pyplot(fig)

            d1, d2 = st.columns(2)
            d1.download_button("Download model_evaluation.csv",
                               to_csv_bytes(result["evaluation"], index=True),
                               "model_evaluation.csv", "text/csv")
            d2.download_button("Download feature_importance.csv",
                               to_csv_bytes(imp), "feature_importance.csv", "text/csv")

    st.markdown("#### Model risks and suitability")
    st.markdown(
        "- **Data leakage risk:** Using `Processing_Days`, `Status`, `Reopened`, or `Satisfaction_Score` "
        "as predictors would reveal information unavailable at submission time. They are excluded.\n"
        "- **Class imbalance risk:** Accuracy can look high even when the model misses many breaches. "
        "Review breach-class precision, recall, F1, and the confusion matrix alongside the baseline.\n"
        "- **Advisory use:** Treat the model as a decision-support flag for staff review, not an "
        "automatic decision or denial.\n"
        "- **Interpretation:** Feature importance shows which encoded inputs the model used more; it "
        "does not prove cause and effect."
    )

# ---------------- Tab 7 ----------------
with tabs[6]:
    st.subheader("Requirements-to-data mapping")
    reqs = []
    if "Service_Type" in clean.columns and clean["Service_Type"].notna().any():
        svc = clean["Service_Type"].fillna("Unknown").value_counts()
        reqs.append({
            "Finding_or_Evidence": f"Highest service volume: {svc.index[0]} ({int(svc.iloc[0])} requests).",
            "Functional_Requirement": f"The system shall display request volume by service type and highlight {svc.index[0]} for workload monitoring.",
            "Evidence_Source": "Service_Type frequency table",
        })
    if not office_stats.empty:
        reqs.append({
            "Finding_or_Evidence": (f"Highest observed office breach rate: {office_stats.index[0]} "
                                    f"({office_stats.iloc[0]['Breach_Rate_Percent']:.2f}%, "
                                    f"n={int(office_stats.iloc[0]['Requests'])})."),
            "Functional_Requirement": "The system shall provide an office-level SLA performance report showing completed request counts and breach rates.",
            "Evidence_Source": "Office SLA breach-rate table",
        })
    if not queue_stats.empty:
        qs = queue_stats[queue_stats["Requests"] > 0]
        if not qs.empty:
            cat = qs["Breach_Rate"].idxmax()
            reqs.append({
                "Finding_or_Evidence": f"Highest observed queue-category breach rate: {cat} ({qs.loc[cat, 'Breach_Rate_Percent']:.2f}%).",
                "Functional_Requirement": "The system shall group open requests by queue category and let staff review queue levels alongside historical SLA breach rates.",
                "Evidence_Source": "Queue-category breach-rate table",
            })
    if model_results:
        reqs.append({
            "Finding_or_Evidence": f"Decision Tree breach-class recall on test data: {model_results['tree_scores']['Recall_Breach_Class_1']:.3f}.",
            "Functional_Requirement": "The system shall present any model-generated delay-risk flag as an advisory item for authorized staff to review, with model performance metrics visible to administrators.",
            "Evidence_Source": "Held-out model evaluation; validate before deployment",
        })
    requirements_df = pd.DataFrame(reqs)
    if requirements_df.empty:
        st.warning("Requirements not generated because required columns/eligible records were unavailable.")
    else:
        st.dataframe(requirements_df, hide_index=True)
        st.download_button("Download requirements_to_data_mapping.csv",
                           to_csv_bytes(requirements_df),
                           "requirements_to_data_mapping.csv", "text/csv")

# ---------------- Tab 8 ----------------
with tabs[7]:
    st.subheader("UML Use Case Diagram")
    fig = use_case_diagram()
    st.pyplot(fig)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    st.download_button("Download diagram (PNG)", buf.getvalue(),
                       "student_service_use_case_diagram.png", "image/png")

# ---------------- Tab 9 ----------------
with tabs[8]:
    st.subheader("Administrator dashboard")
    status_lower = clean["Status"].astype(str).str.lower() if "Status" in clean.columns else pd.Series(dtype=str)
    breach_rate = labeled["SLA_Breached"].mean() * 100 if not labeled.empty else np.nan

    k = st.columns(5)
    k[0].metric("Total Requests", len(clean))
    k[1].metric("Completed", int(status_lower.eq("completed").sum()))
    k[2].metric("Pending/Open", int(status_lower.isin(["pending", "open"]).sum()))
    k[3].metric("Cancelled", int(status_lower.isin(["cancelled", "canceled"]).sum()))
    k[4].metric("SLA Breach Rate", f"{breach_rate:.1f}%" if pd.notna(breach_rate) else "N/A")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Section 1: Service volume by type**")
        if "Service_Type" in clean.columns:
            st.bar_chart(clean["Service_Type"].value_counts())
    with r2:
        st.markdown("**Section 2: SLA performance by office**")
        if not office_stats.empty:
            st.bar_chart(office_stats["Breach_Rate_Percent"])
    r3, r4 = st.columns(2)
    with r3:
        st.markdown("**Section 3: Queue status**")
        if not queue_stats.empty:
            st.bar_chart(queue_stats["Breach_Rate_Percent"])
    with r4:
        st.markdown("**Section 4: Requests flagged for staff review**")
        st.caption("Advisory only; a human must verify before any action is taken.")
        if model_results:
            pipeline_features = model_results["features"]
            st.write("Model flags are generated from:", pipeline_features)

# ---------------- Tab 10 ----------------
with tabs[9]:
    st.subheader("Critical reflection")
    st.markdown(
        "**Why should a high-risk flag trigger human review instead of automatic denial?** "
        "A prediction is an estimate, not proof that a request will miss its target. Staff should verify "
        "the request details, context, and available support options before taking action."
    )
    st.markdown(
        "**How does missing data affect reliability?** Missing dates or numeric values can reduce the "
        "number of records available for analysis and modeling. Imputation can help a model run, but it "
        "does not restore information that was never recorded; missingness should be reported and monitored."
    )
    st.markdown(
        "**What should be shown to explain a prediction?** Show the risk flag, the relevant "
        "submission-time details, the model's performance metrics, and a clear notice that the result is "
        "advisory. Do not present feature importance as proof of causation."
    )
