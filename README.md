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
├── data/                      --> gitignored, regenerate locally (see below)
│   ├── raw/
│   └── processed/
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
│   ├── linear_regression_model.py
│   └── xgboost_model.py
│
├── models/                    --> trained models (tracked in git)
├── reports/                   --> plots (tracked in git)
│
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
python src/xgboost_model.py             # -> models/xgb_demand_model.json
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

## Issues
-  1,280 missing weather values in the merged file — likely from the edge of the date range — so you may want to handle those before training.
