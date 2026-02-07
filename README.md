# Population Pyramid for Japan

This project simulates and visualizes Japan's population pyramid using official e-Stat CSVs for population, fertility, and mortality. The output is an interactive HTML chart.

## Contents

- `population_pyramid_japan.py` — main script for loading data, projecting population, and generating the visualization.
- `index.html` — exported interactive Plotly output.
- `data/` — source CSV files (population, fertility, mortality).

## Data Source

CSV files in `data/` are from e-Stat (Japanese government statistics), including:
- Population estimates by age (5-year groups), sex, and month
- Fertility rates by mother's age (5-year groups)
- Mortality rates by age (5-year groups), sex, and year

## Prerequisites

- Python 3.10+
- `pip`

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If needed, install dependencies manually:

```bash
pip install pandas plotly
```

## Usage

Run:

```bash
python population_pyramid_japan.py
```

The script will:

1. Auto-detect input CSV files in `data/` by table title keywords.
2. Convert 5-year grouped population counts to single-year ages using a smooth, continuous distribution that preserves each bracket total exactly.
3. Use age-specific fertility and mortality rates for cohort-component projection.
4. Generate a slider-based interactive pyramid (`index.html`) with:
   - Total population by year
   - Working-age to non-working ratio, where working age is 22-65 and non-working is 0-21 plus 66+

## Output

Open `index.html` in a browser.

## Configuration

Key constants are in `population_pyramid_japan.py`:

- `MAX_AGE = 110`
- `DISPLAY_START_YEAR = 2026`
- `END_YEAR = 2070`
- `FERTILITY_RATE_PER = 1000`
- `MORTALITY_RATE_PER = 100000`
