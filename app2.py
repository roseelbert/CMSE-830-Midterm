# Repeated Measures Utilization in Horticulture - Streamlit prototype
# Run with:  streamlit run app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Repeated Measures in Horticulture", layout="wide")

# ---------- Title and purpose ----------
st.title("Repeated Measures Utilization in Horticulture")
st.write(
    "Plan a repeated-measures experiment before you run it. Choose your design and "
    "expected effect size, and see how many experimental units you need to reach "
    "80% power, then explore how real postharvest basil data look over time."
)

# ---------- Power formula (noncentral F, same as G*Power) ----------
def rm_power(test, f, N, k, m, rho, eps, alpha=0.05):
    if test == "between":
        lam = f**2 * N * m / (1 + (m - 1) * rho)
        df1, df2 = k - 1, N - k
    elif test == "within":
        lam = f**2 * N * m * eps / (1 - rho)
        df1, df2 = (m - 1) * eps, (N - k) * (m - 1) * eps
    else:
        lam = f**2 * N * m * eps / (1 - rho)
        df1, df2 = (k - 1) * (m - 1) * eps, (N - k) * (m - 1) * eps
    return stats.ncf.sf(stats.f.ppf(1 - alpha, df1, df2), df1, df2, lam)

TESTS = {"Treatment main effect": "between",
         "Time main effect": "within",
         "Treatment × time": "interaction"}
EFFECTS = {"Small (f = 0.10)": 0.10, "Medium (f = 0.25)": 0.25, "Large (f = 0.40)": 0.40}

tab1, tab2 = st.tabs(["Sample size planner", "Postharvest example"])

# ---------- Tab 1: power planner ----------
with tab1:
    with st.sidebar:
        st.header("Design settings")
        effect_label = st.selectbox("Expected effect size", list(EFFECTS), index=1)
        test_label = st.selectbox("Effect you want to detect", list(TESTS))
        k = st.slider("Number of treatments", 2, 6, 3)
        m = st.slider("Number of time points", 2, 8, 5)
        rho = st.slider("Correlation between time points (ρ)", 0.1, 0.9, 0.5, 0.1)
        s = st.selectbox("Plants measured per experimental unit", [1, 2, 4])
        icc = st.slider("ICC (how alike plants in one unit are)", 0.0, 0.9, 0.3, 0.1,
                        disabled=(s == 1))

    f = EFFECTS[effect_label]
    test = TESTS[test_label]
    f_eff = f / np.sqrt((icc if s > 1 else 0) + (1 - (icc if s > 1 else 0)) / s)
    ns = np.arange(2, 61)
    power = rm_power(test, f_eff, ns * k, k, m, rho, 1.0)
    reached = np.nonzero(power >= 0.80)[0]
    n_req = int(ns[reached[0]]) if reached.size else None

    c1, c2 = st.columns(2)
    c1.metric("Units per treatment for 80% power", n_req if n_req else "> 60")
    c2.metric("Total experimental units", n_req * k if n_req else "—")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ns, power, color="#2a78d6", lw=2.5)
    ax.axhline(0.8, color="#999999", ls=":", lw=1)
    if n_req:
        ax.axvline(n_req, color="#eb6834", ls="--", lw=1.5)
        ax.annotate(f"n = {n_req}", (n_req, 0.05), xytext=(5, 0),
                    textcoords="offset points", color="#eb6834")
    ax.set_xlabel("Experimental units per treatment (n)")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{test_label}: {effect_label.lower()}, {k} treatments, {m} time points",
                 loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.caption("Where the curve crosses the dotted line, the design reaches 80% power. "
               "Assumes sphericity holds and α = 0.05.")

# ---------- Tab 2: postharvest example ----------
with tab2:
    ph = pd.read_csv("ph_combined.txt", delimiter="\t")
    names = {"CF": "Chlorophyll fluorescence", "leaf_QI": "Leaf quality index",
             "apical_shoot_QI": "Apical shoot quality", "axil_shoot_QI": "Axillary shoot quality",
             "apical_stem_WL": "Apical stem weight", "axil_shoot_WL": "Axillary shoot weight"}
    var = st.selectbox("Measurement", list(names), format_func=names.get)
    d = ph[ph["variable"] == var]
    means = d.groupby(["TOH", "day"])["value"].agg(["mean", "sem"]).reset_index()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for toh, color in {3: "#2a78d6", 7: "#eb6834"}.items():
        g = means[means["TOH"] == toh]
        ax2.fill_between(g["day"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                         color=color, alpha=0.2, lw=0)
        ax2.plot(g["day"], g["mean"], color=color, lw=2.5, marker="o", label=f"TOH {toh}")
    ax2.set_xlabel("Days in storage at 5 °C")
    ax2.set_ylabel(names[var])
    ax2.set_xticks(sorted(d["day"].unique()))
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)
    st.caption("Mean ± standard error of all plants in each harvest group.")
