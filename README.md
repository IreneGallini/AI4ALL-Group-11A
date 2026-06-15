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
energy-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/ --> for individual work
│   ├── name1_eda.ipynb
│   ├── name1_eda.ipynb etc
│
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── reports/
│
├── requirements.txt
├── environment.yml
└── README.md
```

For the API key, I'd recommend each person create a .env file (already gitignored by default) with:
EIA_API_KEY=your_key_here