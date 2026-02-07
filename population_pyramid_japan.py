import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Input files (auto-detect by title if None)
POPULATION_FILE = None
FERTILITY_FILE = None
MORTALITY_FILE = None

# If None, latest time in each file is used
BASE_TIME_LABEL = None  # e.g., "2026年1月"
FERTILITY_TIME_LABEL = None
MORTALITY_TIME_LABEL = None

DISPLAY_START_YEAR = 2026
END_YEAR = 2070
MAX_AGE = 110

# Units (set to match your downloaded files)
FERTILITY_RATE_PER = 1000    # ASFR is usually per 1,000 women
MORTALITY_RATE_PER = 100000  # Mortality rate usually per 100,000 population

# Model knobs
FERTILITY_MULTIPLIER = 1.0
MORTALITY_MULTIPLIER = 1.0
SEX_RATIO_AT_BIRTH = 1.05  # male births per female birth
NET_MIGRATION_TOTAL = 0    # per year, set non-zero to include migration

OUTPUT_HTML = Path("index.html")


def find_file_by_title_contains(required_all):
    if isinstance(required_all, str):
        required_all = [required_all]
    for path in DATA_DIR.glob("*.csv"):
        try:
            head = path.read_bytes()[:12000].decode("cp932", errors="replace")
        except Exception:
            continue

        # Prefer matching the title line, but fall back to the whole header text
        title_line = None
        for line in head.splitlines():
            if line.startswith('"表題："'):
                title_line = line
                break

        if title_line and all(k in title_line for k in required_all):
            return path

        if all(k in head for k in required_all):
            return path

    return None


def select_input_file(explicit_path, required_title_keywords, kind_label):
    if explicit_path is not None:
        return Path(explicit_path)
    path = find_file_by_title_contains(required_title_keywords)
    if not path:
        raise FileNotFoundError(
            f"Could not find {kind_label} file in {DATA_DIR}. "
            f"Download the official CSV and place it in data/, or set the file path explicitly."
        )
    return path


def read_estat_csv(path: Path) -> pd.DataFrame:
    # Read e-Stat style CSV (Shift-JIS/CP932, metadata rows before header)
    with open(path, "rb") as f:
        head = f.read(12000).decode("cp932", errors="replace")
    head_lines = head.splitlines()

    header_row = None
    for i, line in enumerate(head_lines):
        if line.startswith('"表章項目 コード"'):
            header_row = i
            break
    if header_row is None:
        for i, line in enumerate(head_lines):
            if (
                ("母の年齢(5歳階級) コード" in line or "母の年齢（５歳階級） コード" in line)
                and ("時間軸" in line)
            ):
                header_row = i
                break
    if header_row is None:
        for i, line in enumerate(head_lines):
            if (
                ("年齢(5歳階級) コード" in line or "年齢5歳階級 コード" in line or "年齢（５歳階級） コード" in line)
                and ("時間軸" in line)
            ):
                header_row = i
                break
    if header_row is None:
        raise ValueError(f"Could not find header row in {path}")

    return pd.read_csv(path, encoding="cp932", skiprows=header_row, header=0, engine="python")


def parse_time_label(label: str):
    # Examples: "2026年1月", "2023年"
    s = str(label)
    m = re.search(r"(\d{4})年(\d{1,2})月", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d{4})年", s)
    if m:
        return int(m.group(1)), 0
    return None


def latest_time_label(series: pd.Series) -> str:
    labels = series.dropna().unique().tolist()
    best = None
    for label in labels:
        ym = parse_time_label(label)
        if not ym:
            continue
        if best is None or ym > best[0]:
            best = (ym, label)
    if not best:
        raise ValueError("Could not determine latest time label")
    return best[1]


def parse_age_group(label: str, max_age: int):
    s = str(label)
    if "総数" in s or "再掲" in s:
        return None
    if "以上" in s:
        digits = re.findall(r"\d+", s)
        start = int(digits[0]) if digits else max_age
        return start, max_age
    digits = re.findall(r"\d+", s)
    if not digits:
        return None
    if len(digits) == 1:
        return int(digits[0]), int(digits[0])
    return int(digits[0]), int(digits[1])


def to_number(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").strip()
    if s == "" or s == "-":
        return np.nan
    return float(s)


def expand_5yr_to_single_counts(age_group_values, max_age=100):
    arr = np.zeros(max_age + 1, dtype=float)
    bins = []
    for (start, end), value in age_group_values.items():
        if start is None:
            continue
        start = max(0, min(start, max_age))
        end = max(0, min(end, max_age))
        if end < start:
            continue
        bins.append((start, end, float(value)))

    if not bins:
        return arr

    bins.sort(key=lambda x: x[0])
    contiguous = all(bins[i][0] == bins[i - 1][1] + 1 for i in range(1, len(bins)))
    if not contiguous:
        # Fallback to uniform split when groups are not contiguous.
        for start, end, value in bins:
            width = end - start + 1
            arr[start:end + 1] += value / width
        return arr

    widths = np.array([end - start + 1 for start, end, _ in bins], dtype=float)
    means = np.array([value / width for (_, _, value), width in zip(bins, widths)], dtype=float)
    n = len(bins)

    # Build all edge values as a function of one free parameter x0:
    # x_{i+1} = 2*m_i - x_i  ->  x_i = a_i*x0 + b_i
    a = np.zeros(n + 1, dtype=float)
    b = np.zeros(n + 1, dtype=float)
    a[0] = 1.0
    for i in range(n):
        a[i + 1] = -a[i]
        b[i + 1] = 2.0 * means[i] - b[i]

    # Feasible interval for non-negative edges.
    lo = -np.inf
    hi = np.inf
    feasible = True
    for ai, bi in zip(a, b):
        if ai > 0:
            lo = max(lo, -bi)
        elif ai < 0:
            hi = min(hi, bi)
        elif bi < 0:
            feasible = False

    # Choose x0 that minimizes curvature within the feasible interval.
    if n > 1:
        alphas = np.zeros(n - 1, dtype=float)
        betas = np.zeros(n - 1, dtype=float)
        for i in range(n - 1):
            alphas[i] = a[i] - 2.0 * a[i + 1] + a[i + 2]
            betas[i] = b[i] - 2.0 * b[i + 1] + b[i + 2]
        denom = np.sum(alphas * alphas)
        if denom > 0:
            x0 = -np.sum(alphas * betas) / denom
        else:
            x0 = means[0]
    else:
        x0 = means[0]

    if feasible and lo <= hi:
        x0 = min(max(x0, lo), hi)

    edges = a * x0 + b

    for i, (start, end, _value) in enumerate(bins):
        width = end - start + 1
        left = edges[i]
        right = edges[i + 1]
        for k, age in enumerate(range(start, end + 1)):
            t = (k + 0.5) / width
            arr[age] = (1.0 - t) * left + t * right

    return arr


# For rates (per person per year), apply the same rate to each age in the group.
def expand_5yr_to_single_rates(age_group_rates, max_age=100):
    arr = np.zeros(max_age + 1, dtype=float)
    for (start, end), rate in age_group_rates.items():
        if start is None:
            continue
        start = max(0, min(start, max_age))
        end = max(0, min(end, max_age))
        arr[start:end + 1] = rate
    return arr


# --- Column helpers ---

def pick_label_column(columns, keyword):
    # Prefer label columns that include keyword but not "コード"/"補助コード"
    candidates = [c for c in columns if keyword in str(c)]
    for c in candidates:
        s = str(c)
        if ("コード" not in s) and ("補助コード" not in s):
            return c
    return candidates[0] if candidates else None


# --- Population (5-year groups -> single-year) ---

def load_population_by_age(path: Path, time_label: str | None, max_age: int):
    df = read_estat_csv(path)

    # Filter to national totals and total population
    if "人口" in df.columns:
        df = df[df["人口"].astype(str).str.contains("総人口")]
    if "全国" in df.columns:
        df = df[df["全国"].astype(str).str.contains("全国")]
    if "概算値" in df.columns:
        df = df[df["概算値"].astype(str).str.contains("概算値")]

    time_col = "時間軸（年月日現在）"
    if time_label is None:
        time_label = latest_time_label(df[time_col])
    df = df[df[time_col] == time_label]

    age_col = "年齢5歳階級"

    male = {}
    female = {}

    for _, row in df.iterrows():
        rng = parse_age_group(row[age_col], max_age)
        if not rng:
            continue
        m = to_number(row.get("男"))
        f = to_number(row.get("女"))
        if math.isnan(m) or math.isnan(f):
            continue
        # population is in 万人 in this dataset -> convert to persons
        m *= 10000
        f *= 10000
        male[rng] = m
        female[rng] = f

    male_single = expand_5yr_to_single_counts(male, max_age=max_age)
    female_single = expand_5yr_to_single_counts(female, max_age=max_age)

    year, _ = parse_time_label(time_label)
    return year, time_label, male_single, female_single


# --- Fertility rates (ASFR) ---

def load_asfr(path: Path, time_label: str | None, max_age: int):
    df = read_estat_csv(path)

    # Wide format with years as columns (e.g., 2023年, 2022年...)
    year_cols = [c for c in df.columns if re.match(r"\d{4}年", str(c))]
    if year_cols:
        if time_label is None:
            time_label = sorted(year_cols)[-1]

        label_col = pick_label_column(df.columns, "母の年齢")
        if label_col is None:
            label_col = df.columns[0]

        labels = df[label_col].astype(str)
        rows = df[labels.str.contains("出生率_") & labels.str.contains("女性人口")]

        values = {}
        for _, row in rows.iterrows():
            label = str(row[label_col])
            rng = parse_age_group(label, max_age)
            if not rng:
                continue
            rate = to_number(row.get(time_label))
            if math.isnan(rate):
                continue
            rate = rate / FERTILITY_RATE_PER * FERTILITY_MULTIPLIER
            values[rng] = rate

        asfr = expand_5yr_to_single_rates(values, max_age=max_age)
        return time_label, asfr

    raise ValueError("Unsupported fertility file format (expected wide format with year columns)")


# --- Mortality rates (by sex) ---

def load_mortality(path: Path, time_label: str | None, max_age: int):
    df = read_estat_csv(path)

    # Wide format with years as columns
    year_cols = [c for c in df.columns if re.match(r"\d{4}年", str(c))]
    if year_cols:
        if time_label is None:
            time_label = sorted(year_cols)[-1]

        # Keep death rate rows only
        if "表章項目" in df.columns:
            df = df[df["表章項目"].astype(str).str.contains("死亡率")]

        # Keep total cause if present
        for cause_key in ["死因年次推移分類", "死因"]:
            cause_col = pick_label_column(df.columns, cause_key)
            if cause_col:
                df = df[df[cause_col].astype(str).str.contains("総数")]
                break

        sex_col = pick_label_column(df.columns, "性別")

        age_col = pick_label_column(df.columns, "年齢")

        if sex_col is None or age_col is None:
            raise ValueError("Could not find sex or age column in mortality file")

        male_vals = {}
        female_vals = {}

        for _, row in df.iterrows():
            rng = parse_age_group(row[age_col], max_age)
            if not rng:
                continue
            rate = to_number(row.get(time_label))
            if math.isnan(rate):
                continue
            rate = rate / MORTALITY_RATE_PER * MORTALITY_MULTIPLIER

            sex = str(row.get(sex_col, ""))
            if "男" in sex:
                male_vals[rng] = rate
            elif "女" in sex:
                female_vals[rng] = rate

        rate_male = expand_5yr_to_single_rates(male_vals, max_age=max_age)
        rate_female = expand_5yr_to_single_rates(female_vals, max_age=max_age)
        return time_label, rate_male, rate_female

    raise ValueError("Unsupported mortality file format (expected wide format with year columns)")


# --- Optional migration profile ---

def default_migration_profile(max_age, peak=27, spread=10):
    ages = np.arange(max_age + 1)
    profile = np.exp(-0.5 * ((ages - peak) / spread) ** 2)
    profile /= profile.sum()
    return profile


# --- Cohort-component simulation ---

def simulate_population(
    pop_male,
    pop_female,
    asfr,
    surv_male,
    surv_female,
    start_year,
    end_year,
    srb=1.05,
    net_migration_male=None,
    net_migration_female=None,
):
    years = list(range(start_year, end_year + 1))
    results = {}
    male = pop_male.copy()
    female = pop_female.copy()

    for year in years:
        results[year] = (male.copy(), female.copy())

        # Births (female age-specific fertility rates)
        births = np.sum(female * asfr)
        male_births = births * (srb / (1 + srb))
        female_births = births * (1 / (1 + srb))

        # Age the population
        new_male = np.zeros_like(male)
        new_female = np.zeros_like(female)

        new_male[1:] = male[:-1] * surv_male[:-1]
        new_female[1:] = female[:-1] * surv_female[:-1]

        # Keep the open-ended age group
        new_male[-1] += male[-1] * surv_male[-1]
        new_female[-1] += female[-1] * surv_female[-1]

        # Add births to age 0 (apply survival during first year)
        new_male[0] = male_births * surv_male[0]
        new_female[0] = female_births * surv_female[0]

        # Migration
        if net_migration_male is not None:
            new_male += net_migration_male
        if net_migration_female is not None:
            new_female += net_migration_female

        male, female = new_male, new_female

    return results


def build_plotly_pyramid(results):
    import plotly.graph_objects as go

    ages = np.arange(MAX_AGE + 1)
    years = list(results.keys())
    plot_years = [y for y in years if y >= DISPLAY_START_YEAR]

    def working_ratio(male, female, work_start=22, work_end=65):
        total = male + female
        working = total[work_start : work_end + 1].sum()
        non_working = total[:work_start].sum() + total[work_end + 1 :].sum()
        if non_working == 0:
            return float("inf")
        return working / non_working

    max_pop = 0
    for year in plot_years:
        male, female = results[year]
        max_pop = max(max_pop, male.max(), female.max())

    max_pop = math.ceil(max_pop / 10000) * 10000

    start_year = plot_years[0]
    start_male, start_female = results[start_year]
    start_ratio = working_ratio(start_male, start_female)

    fig = go.Figure(
        data=[
            go.Bar(x=-start_male, y=ages, orientation="h", name="Male", marker_color="#4C78A8"),
            go.Bar(x=start_female, y=ages, orientation="h", name="Female", marker_color="#F58518"),
        ],
        layout=go.Layout(
            title=f"Japan Population Pyramid ({start_year})",
            barmode="overlay",
            bargap=0.1,
            xaxis=dict(title="Population", range=[-max_pop, max_pop], tickformat=",") ,
            yaxis=dict(title="Age", range=[0, MAX_AGE]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=(
                    f"Total population: {int(start_male.sum()+start_female.sum()):,} | "
                    f"Working (22-65) : Non-working = {start_ratio:.2f}"
                ),
            )
        ],
    ),
)

    frames = []
    for year in plot_years:
        male, female = results[year]
        total = int(male.sum() + female.sum())
        ratio = working_ratio(male, female)
        frames.append(
            go.Frame(
                data=[
                    go.Bar(x=-male, y=ages, orientation="h", marker_color="#4C78A8"),
                    go.Bar(x=female, y=ages, orientation="h", marker_color="#F58518"),
                ],
                name=str(year),
                layout=go.Layout(
                    title=f"Japan Population Pyramid ({year})",
                    annotations=[
                        dict(
                            x=0.5,
                            y=1.08,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            text=(
                                f"Total population: {total:,} | "
                                f"Working (22-65) : Non-working = {ratio:.2f}"
                            ),
                        )
                    ],
                ),
            )
        )

    fig.frames = frames

    fig.update_layout(
        margin=dict(t=160, r=40, b=190, l=40),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                direction="left",
                x=0.5,
                y=-0.30,
                xanchor="center",
                yanchor="top",
                pad={"r": 8, "t": 0},
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.08,
                y=-0.16,
                len=0.84,
                pad={"t": 18, "b": 0},
                steps=[
                    dict(method="animate", args=[[str(y)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], label=str(y))
                    for y in plot_years
                ],
            )
        ],
    )

    return fig


def main():
    pop_path = select_input_file(POPULATION_FILE, ["人口推計", "年齢"], "population")
    fert_path = select_input_file(FERTILITY_FILE, ["母の年齢", "出生率"], "fertility (ASFR)")
    mort_path = select_input_file(MORTALITY_FILE, ["死亡率", "年齢"], "mortality")

    base_year, base_time, base_male, base_female = load_population_by_age(
        pop_path, BASE_TIME_LABEL, MAX_AGE
    )
    print("Base time:", base_time)
    print("Base year:", base_year)
    print("Total population:", int(base_male.sum() + base_female.sum()))

    fert_time, asfr = load_asfr(fert_path, FERTILITY_TIME_LABEL, MAX_AGE)
    print("Fertility time:", fert_time)

    mort_time, rate_male, rate_female = load_mortality(mort_path, MORTALITY_TIME_LABEL, MAX_AGE)
    print("Mortality time:", mort_time)

    surv_male = np.exp(-rate_male)
    surv_female = np.exp(-rate_female)

    if NET_MIGRATION_TOTAL != 0:
        profile = default_migration_profile(MAX_AGE)
        male_share = SEX_RATIO_AT_BIRTH / (1 + SEX_RATIO_AT_BIRTH)
        female_share = 1 / (1 + SEX_RATIO_AT_BIRTH)
        mig_male = profile * (NET_MIGRATION_TOTAL * male_share)
        mig_female = profile * (NET_MIGRATION_TOTAL * female_share)
    else:
        mig_male = None
        mig_female = None

    results = simulate_population(
        base_male,
        base_female,
        asfr,
        surv_male,
        surv_female,
        base_year,
        END_YEAR,
        srb=SEX_RATIO_AT_BIRTH,
        net_migration_male=mig_male,
        net_migration_female=mig_female,
    )

    try:
        fig = build_plotly_pyramid(results)
        fig.write_html(OUTPUT_HTML)
        print(f"Saved visualization to {OUTPUT_HTML}")
    except Exception as e:
        print("Plotly visualization skipped:", e)
        print("Install plotly to enable HTML output: pip install plotly")


if __name__ == "__main__":
    main()
