# Repeated Measures Utilization in Horticulture

A Streamlit app that enables horticultural researchers to complete their own power calculations 
and better plan repeated measures experiments to discern intended effects.

**Author:** Rose Elbert, Michigan State University · CMSE 830 Midterm, Fall 2026

**Question:** How do power and required sample size change with the number of
treatments, time points, correlation between measurements, subsampling, and *n*?

## Repository structure

```
CMSE-830-Midterm/
├── README.md
├── app2.py                    # Streamlit app
├── requirements.txt           # Python packages
├── ph_combined.txt            # postharvest basil data
├── rm_power_validation.csv    # analytic vs. simulated power
├── rm_required_n.csv          # n needed for 80% power
├── rm_power_grid.csv          # power over a grid of designs
└── notebooks/
    └── Elbert_Midterm.ipynb   # abstract, EDA, static figures
```

## How to run

```bash
pip install -r requirements.txt
streamlit run app2.py
```

## Data

- **Postharvest basil (`ph_combined.txt`):** Some of my M.S. data. 'Genovese' basil stored
  at 5 °C, measured on 7 days over 13 days; TOH (time of harvest) (0700 or 1500 hr) × si (0, 25, 50, 75), 2 reps.
  One row = one plant on one day (4,032 rows). Reps are the experimental units;
  plants are subsamples. No missing values or duplicates.
- **Power tables (`rm_*.csv`):** computed with the noncentral F distribution (as in
  G\*Power), and checked against simulated datasets for each of 144 designs.

## Limitations

- This app assumes balanced designs, normal residuals, and one correlation for all time points
- Effect size must be estimated in advance, as well as whether the researcher is seeking a treatment effect,
  time effect, or interaction effect.
- Example dataset is small (2 reps, one cultivar, one temperature)

## AI use

Claude (Anthropic) helped write the Python/Streamlit code. The project question, data,
analysis decisions, and interpretation are my own. Claude (Anthropic) helped write the Python/Streamlit 
code and with the formatting for this readme. I checked the power calculations against simulation and in R, and read the SciPy
and statsmodels documentation for unfamiliar functions. The project question, data,
analysis decisions, and interpretation are my own.
