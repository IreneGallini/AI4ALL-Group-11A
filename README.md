# AI4ALL-Group-11A
## Repository Workflow

### EDA Phase
For initial exploration, everyone should work in their own Jupyter notebook in the `notebooks/` folder. Notebooks are fast and convenient for EDA, visualization, and testing ideas, but multiple people editing the same `.ipynb` file often leads to difficult Git merge conflicts. Keeping separate notebooks allows everyone to work independently without interfering with each other's progress.

### Project Development
As we move into data processing and modeling, reusable code should be moved into Python scripts in the `src/` folder. We will use a shared environment defined by `requirements.txt` or `environment.yml` so everyone has the same package versions and dependencies installed.

### Why?
- Easier collaboration and version control
- Fewer merge conflicts
- Consistent environments across team members
- More reproducible results
- Cleaner and more maintainable codebase 

```
AI4ALL-Group-11A/
│
├── data/                      --> mostly gitignored, regenerate locally (see below)
│   ├── raw/                   --> gitignored
│   └── processed/             --> gitignored, except eia_with_features.csv (committed as a
│                                   static baseline snapshot so deployment doesn't need to
│                                   rerun the pipeline)
│
├── notebooks/                 --> for individual work
│   ├── diego_eda.ipynb
│   ├── irene_eda.ipynb
│   ├── leul_eda.ipynb
│   └── sujjal_eda.ipynb
│
├── src/
│   ├── eia_api_script.py      --> pulls EIA demand/solar/wind data
│   ├── weather_merge.py       --> pulls weather data, merges with EIA data
│   ├── demand_lag_add.py      --> adds lagged-demand features
│   ├── features.py            --> shared feature engineering, used by both models below (and app.py)
│   ├── linear_regression_model.py
│   ├── xgboost_model.py       --> multi-horizon XGBoost (1 day / 1 week / 1 month)
│   └── train_demand_forecast.py  --> multi-horizon Random Forest (1 day / 1 week / 1 month)
│
├── models/                    --> trained models (tracked in git)
├── reports/                   --> plots (tracked in git)
│
├── app.py                     --> Streamlit dashboard (deployed via Streamlit Community Cloud)
├── requirements.txt
├── environment.yml
└── README.md
```

## Rebuilding the data and models

Run in order from the repo root:

```bash
python src/eia_api_script.py       # -> data/raw/eia_energy_data.csv
python src/weather_merge.py        # -> data/raw/weather_data.csv, data/processed/eia_with_weather.csv
python src/demand_lag_add.py       # -> data/processed/eia_with_features.csv
python src/linear_regression_model.py   # -> models/linear_regression_model.pkl
python src/xgboost_model.py             # -> models/xgb_model_{1_day,1_week,1_month}.json, models/xgb_demand_metrics.json
python src/train_demand_forecast.py     # -> models/rf_model_{1_day,1_week,1_month}.joblib, models/rf_demand_metrics.json
```

See `CLAUDE.md` for details on each step's inputs/outputs.

---

## Setting up

### Create the conda environment

This installs Python and all the packages the project needs.

```bash
conda env create -f environment.yml
```

Then activate it (you'll need to do this every time you open a new terminal):

```bash
conda activate ai4all-11a
```


### Get a free EIA API key

1. Go to https://www.eia.gov/opendata/ and click **Register**
2. Fill out the form — it's free and instant
3. Your API key will be emailed to you


### Add your API key

1. In the project folder, copy the example file

2. Open `.env` and replace `your_key_here` with your actual key:
`.env` is gitignored — it will never be committed, so your key stays private.


### Set up your Jupyter kernel

This makes Jupyter use the project environment instead of your system Python:

```bash
python -m ipykernel install --user --name ai4all-11a --display-name "Python (ai4all-11a)"
```

Then launch JupyterLab:

```bash
jupyter lab
```

Open your notebook in `notebooks/`, click the kernel name in the top-right corner, and select **Python (ai4all-11a)**.

## Running the app locally

```bash
streamlit run app.py
```

`app.py` is currently a minimal stub — see the TODO docstring at the top of that file for what's
still left to build.

## Deployment

Target: **Streamlit Community Cloud**, deploying straight from this GitHub repo. It installs
dependencies from `requirements.txt` (not `environment.yml`), so keep that file's pins in sync
with whatever the committed models were trained under.

For the static-baseline dashboard (current scope — no live data refresh yet), no secrets are
needed: `models/` and `data/processed/eia_with_features.csv` are both committed, so a fresh
clone has everything required to serve predictions without rerunning the pipeline.
`EIA_API_KEY` only becomes relevant once a live-refresh feature is built.

## Bias and Responsible AI Considerations

Our model has several potential sources of bias:

- **Regional reliability bias:** Model performance varies across grid regions. Some regions have less reliable predictions than others, meaning a single overall performance metric may not represent every region equally.

- **Weather proxy bias:** We use one representative city to represent each grid region. This may not capture local weather differences, especially in larger regions that cover multiple states.

- **Training data bias:** The model was trained on approximately one year of data, which may not include enough examples of rare events such as extreme weather conditions or unusual demand patterns.

These limitations should be considered when interpreting model predictions. The model is designed for regional electricity demand forecasting and may not perform equally across all locations or conditions.

## References

- U.S. Energy Information Administration. (2024). *EIA Grid Monitor*.  
  https://www.eia.gov/electricity/gridmonitor/

- Open-Meteo. (2024). *Historical Weather API*.  
  https://open-meteo.com/en/docs/historical-weather-api

- Bourdeau, M., et al. (2019). *Modeling and forecasting building energy consumption: A review of data-driven techniques*. Sustainable Cities and Society, 48, 101533.  
  https://doi.org/10.1016/j.scs.2019.101533

- Salman, S. (2025). *Seasonal forecasting of the hourly electricity demand applying machine and deep learning algorithms*. Scientific Reports.  
  https://www.nature.com/articles/s41598-025-91878-0

- DNV. (2024). *Transforming grid operations with accurate short-term energy predictions*.  
  https://www.dnv.com/article/transforming-grid-operations-with-accurate-short-term-energy-predictions/

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.

- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.

- Streamlit Inc. *Streamlit*.  
  https://streamlit.io/