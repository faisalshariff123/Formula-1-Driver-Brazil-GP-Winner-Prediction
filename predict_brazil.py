import os
import time
import warnings
from collections import defaultdict
from dotenv import load_dotenv
import requests
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

import fastf1
from fastf1.core import DataNotLoadedError
from fastf1.events import get_event_schedule

try:
    fastf1.Cache.enable_cache('cache')
except Exception:
    pass


def sec(td):
    # convert pandass Timedelta or python timedelta to seconds
    if pd.isna(td):
        return np.nan
    try:
        return td.total_seconds()
    except Exception:
        return float(td)


def load_session_safe(year, event_name):
    try:
        s = fastf1.get_session(year, event_name, 'R')
        s.load()
        return s
    except DataNotLoadedError:
        print(f"Skipping {event_name}: data not available")
        return None
    except Exception as e:
        print(f"Error loading {event_name}: {e}")
        return None


def build_cumulative_features(year):
    schedule = get_event_schedule(year)
    if 'StartDate' in schedule.columns:
        schedule = schedule.sort_values('StartDate')

    races = list(schedule['EventName'])

    race_results = {}
    race_laps = {}

    for r in races:
        s = load_session_safe(year, r)
        if s is None:
            continue
        try:
            res = s.results
            race_results[r] = res
        except Exception:
            race_results[r] = None
        try:
            laps = s.laps
            race_laps[r] = laps
        except Exception:
            race_laps[r] = None

    # build training rows w cumulative stats up to each race
    driver_cum = defaultdict(lambda: {
        'wins': 0,
        'podiums': 0,
        'races': 0,
        'sum_pos': 0.0,
        'sum_fastest_lap': 0.0,
        'sum_lap_time': 0.0,
        'lap_count': 0,
        'lap_time_sq_sum': 0.0,
    })

    X_rows = []
    y = []
    race_order = [r for r in races if r in race_results]

    def _get_driver_from_row(row):
        for col in ('Driver', 'Abbreviation', 'Abbrev', 'DriverCode', 'Code'):
            if col in row.index:
                return row[col]
        for col in row.index:
            val = row[col]
            if isinstance(val, str) and len(val) <= 4 and val.isalpha():
                return val
        return None


    def _get_position_from_row(row):
        for col in ('Position', 'Pos', 'position'):
            if col in row.index:
                return row[col]
        return None


    for idx, r in enumerate(race_order):
        res = race_results.get(r)
        if res is None or res.empty:
            laps = race_laps.get(r)
            if laps is not None and not laps.empty:
                for drv, g in laps.groupby('Driver'):
                    times = g['LapTime'].dropna().map(sec)
                    driver_cum[drv]['sum_lap_time'] += times.sum()
                    driver_cum[drv]['lap_count'] += times.count()
                    driver_cum[drv]['lap_time_sq_sum'] += (times**2).sum()
            continue

        laps = race_laps.get(r)
        for _, row in res.iterrows():
            drv = _get_driver_from_row(row)
            if drv is None:
                continue
            cum = driver_cum[drv]
            if cum['races'] == 0:
                continue

            avg_pos = cum['sum_pos'] / cum['races'] if cum['races'] else np.nan
            avg_fastest = (cum['sum_fastest_lap'] / cum['races']) if cum['races'] else np.nan
            avg_lap = (cum['sum_lap_time'] / cum['lap_count']) if cum['lap_count'] else np.nan
            lap_var = (cum['lap_time_sq_sum'] / cum['lap_count'] - (avg_lap ** 2)) if cum['lap_count'] else np.nan

            X_rows.append({
                'driver': drv,
                'wins': cum['wins'],
                'podiums': cum['podiums'],
                'races': cum['races'],
                'avg_pos': avg_pos,
                'avg_fastest_lap': avg_fastest,
                'avg_lap_time': avg_lap,
                'lap_var': lap_var,
            })
            pos = _get_position_from_row(row)
            y.append(1 if pos == 1 or str(pos) == '1' else 0)

        for _, row in res.iterrows():
            drv = _get_driver_from_row(row)
            if drv is None:
                continue
            pos = _get_position_from_row(row)
            driver_cum[drv]['races'] += 1
            if pd.notna(pos):
                driver_cum[drv]['sum_pos'] += float(pos)
                if int(pos) == 1:
                    driver_cum[drv]['wins'] += 1
                if int(pos) <= 3:
                    driver_cum[drv]['podiums'] += 1

        if laps is not None and not laps.empty:
            for drv, g in laps.groupby('Driver'):
                times = g['LapTime'].dropna().map(sec)
                if times.count() == 0:
                    continue
                # fastest lap in that race for driver
                fastest = times.min()
                driver_cum[drv]['sum_fastest_lap'] += fastest
                driver_cum[drv]['sum_lap_time'] += times.sum()
                driver_cum[drv]['lap_count'] += times.count()
                driver_cum[drv]['lap_time_sq_sum'] += (times**2).sum()

    X = pd.DataFrame(X_rows)
    y = np.array(y)
    return X, y, driver_cum, races, race_results, race_laps


def train_and_predict(X, y, driver_cum, races, race_results, race_laps, year):
    target_name = None
    for r in races:
        if 'São Paulo' in r or 'Brazil' in r or 'Sao Paulo' in r or 'São Paulo Grand Prix' in r:
            target_name = r
            break
    if target_name is None:
        target_name = races[-1]  

    participants = []
    if target_name in race_results and race_results[target_name] is not None:
        resdf = race_results[target_name]
        for col in ('Driver', 'Abbreviation', 'Abbrev', 'DriverCode', 'Code'):
            if col in resdf.columns:
                participants = list(resdf[col].unique())
                break
        else:
            for c in resdf.columns:
                if resdf[c].dtype == object:
                    participants = list(resdf[c].unique())
                    break
    else:
        participants = list(driver_cum.keys())

    # prepare driver-level test features using cumulative stats
    rows = []
    for drv in participants:
        cum = driver_cum.get(drv, None)
        if cum is None or cum['races'] == 0:
            rows.append({'driver': drv, 'wins': 0, 'podiums': 0, 'races': 0, 'avg_pos': np.nan, 'avg_fastest_lap': np.nan, 'avg_lap_time': np.nan, 'lap_var': np.nan})
            continue
        avg_pos = cum['sum_pos'] / cum['races'] if cum['races'] else np.nan
        avg_fastest = (cum['sum_fastest_lap'] / cum['races']) if cum['races'] else np.nan
        avg_lap = (cum['sum_lap_time'] / cum['lap_count']) if cum['lap_count'] else np.nan
        lap_var = (cum['lap_time_sq_sum'] / cum['lap_count'] - (avg_lap ** 2)) if cum['lap_count'] else np.nan
        rows.append({'driver': drv, 'wins': cum['wins'], 'podiums': cum['podiums'], 'races': cum['races'], 'avg_pos': avg_pos, 'avg_fastest_lap': avg_fastest, 'avg_lap_time': avg_lap, 'lap_var': lap_var})

    X_test = pd.DataFrame(rows)

    if X_test.empty:
        fallback = set()
        if 'driver' in X.columns:
            fallback.update(X['driver'].unique())
        fallback.update(driver_cum.keys())
        for laps in race_laps.values():
            if laps is not None and hasattr(laps, 'Driver'):
                try:
                    fallback.update(laps['Driver'].unique())
                except Exception:
                    pass
        participants = sorted(fallback)
        rows = []
        for drv in participants:
            cum = driver_cum.get(drv, None)
            if cum is None or cum['races'] == 0:
                rows.append({'driver': drv, 'wins': 0, 'podiums': 0, 'races': 0, 'avg_pos': np.nan, 'avg_fastest_lap': np.nan, 'avg_lap_time': np.nan, 'lap_var': np.nan})
            else:
                avg_pos = cum['sum_pos'] / cum['races'] if cum['races'] else np.nan
                avg_fastest = (cum['sum_fastest_lap'] / cum['races']) if cum['races'] else np.nan
                avg_lap = (cum['sum_lap_time'] / cum['lap_count']) if cum['lap_count'] else np.nan
                lap_var = (cum['lap_time_sq_sum'] / cum['lap_count'] - (avg_lap ** 2)) if cum['lap_count'] else np.nan
                rows.append({'driver': drv, 'wins': cum['wins'], 'podiums': cum['podiums'], 'races': cum['races'], 'avg_pos': avg_pos, 'avg_fastest_lap': avg_fastest, 'avg_lap_time': avg_lap, 'lap_var': lap_var})
        X_test = pd.DataFrame(rows)

    # train model (drop driver col)
    X_train = X.copy()
    if 'driver' in X_train.columns:
        X_train = X_train.drop(columns=['driver'])
    else:
        pass
    X_train = X_train.fillna(X_train.median())
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    if len(X_train) == 0 or len(np.unique(y)) < 2:
        print('Not enough training data to train classifier')
        return None
    clf.fit(X_train, y)

    # prepare test features, safely drop driver col if present
    if X_test.empty:
        print('No participants found for target race; aborting')
        return None
    X_test_proc = X_test.copy()
    if 'driver' in X_test_proc.columns:
        X_test_proc = X_test_proc.drop(columns=['driver'])
    X_test_proc = X_test_proc.fillna(X_train.median())
    probs = clf.predict_proba(X_test_proc)[:, 1]
    X_test['prob_win'] = probs

    X_test = X_test.sort_values('prob_win', ascending=False).reset_index(drop=True)

    # plot top 10
    topn = min(10, len(X_test))
    plt.figure(figsize=(10, 6))
    plt.barh(X_test['driver'][:topn][::-1], X_test['prob_win'][:topn][::-1])
    plt.xlabel('Predicted probability of winning')
    plt.title(f'Predicted win probabilities - {year} {target_name}')
    plt.tight_layout()
    out = 'brazil_prediction.png'
    plt.savefig(out)
    print(f'Saved probability chart to {out}')

    print('\nTop predictions:')
    print(X_test[['driver', 'prob_win']].head(5))

    return X_test


def get_iam_token(api_key):
    """Exchange IBM Cloud API key for IAM access token"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    
    response = requests.post(url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to get IAM token: {response.text}")
    
    return response.json()["access_token"]

def call_watson(iam_token, winner, second, third, features):
    import requests
    url = os.getenv('WATSON_URL') or "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {iam_token}'
    }
    prompt = (
        f"Predictive model picked {winner} as most likely winner, with {second} and {third} next.\n"
        "Explain concisely why the predicted winner would win, using the provided features (wins, podiums, avg_pos, avg_lap_time, lap_var). Then explain why the 2nd and 3rd would not win.\n"
        f"Features snapshot: {features}\n"
        "Answer in plain readable text with no warning or parantheses or terms and conditions and dont mention model id or name too"
    )
    body = {
        'messages': [{'role': 'user', 'content': prompt}],
        'project_id': os.getenv('WATSON_PROJECT_ID'),
        'model_id': os.getenv('WATSON_MODEL_ID', 'meta-llama/llama-3-3-70b-instruct'),
        'max_tokens': 500,
    }
    r = requests.post(url, headers=headers, json=body)
    if r.status_code != 200:
        print('Watson call failed:', r.status_code, r.text[:500])
        return None
    return r.json()


#######################
# Watson response helpers
#######################

def pretty_print_response(resp):
    """Print the full Watson response as pretty JSON."""
    try:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
    except Exception:
        print(resp)


def extract_assistant_text(resp):
    """Extract assistant message text from common Watson/Chat response shapes.

    Handles: resp['choices'][*]['message']['content'], resp['results'][*]['generated_text'],
    or top-level 'generated_text'. Falls back to stringifying resp.
    """
    if not resp:
        return ""

    # common chat response format with 'choices' list
    if isinstance(resp, dict) and 'choices' in resp and isinstance(resp['choices'], list):
        texts = []
        for choice in resp['choices']:
            # try nested message -> content
            msg = choice.get('message') or {}
            if isinstance(msg, dict):
                content = msg.get('content') or msg.get('text')
            else:
                content = choice.get('text')
            if content:
                texts.append(str(content).strip())
        if texts:
            return "\n\n".join(texts).strip()

    # results/generate style
    if isinstance(resp, dict) and 'results' in resp and isinstance(resp['results'], list):
        first = resp['results'][0]
        # try common fields
        for key in ('generated_text', 'text', 'content'):
            if key in first:
                return str(first[key]).strip()

    if isinstance(resp, dict) and 'generated_text' in resp:
        return str(resp['generated_text']).strip()

    # last resort
    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)


def assistant_text_to_markdown(title, assistant_text):
    """Return a small Markdown document for the assistant text."""
    if not assistant_text:
        return f"## {title}\n\n_No content returned by the assistant._\n"
    paragraphs = [p.strip() for p in assistant_text.split('\n\n') if p.strip()]
    md_lines = [f"## {title}", ""]
    for p in paragraphs:
        md_lines.append(p)
        md_lines.append("")
    return "\n".join(md_lines)


def parse_stats_from_text(text):
    """Heuristic parse of numeric stats from assistant text.

    Returns a dict with any found fields.
    """
    stats = {}
    if not text:
        return stats

    patterns = {
        'wins': r"(\d+)\s+wins?",
        'podiums': r"(\d+)\s+podiums?",
        'races': r"(\d+)\s+races?",
        'avg_pos': r"avg(?:erage)?\s+position(?:[:\s]*)?([0-9]+\.[0-9]+|[0-9]+)",
        'avg_lap_time': r"avg(?:erage)?\s+lap\s+time(?:[:\s]*)?([0-9]+\.[0-9]+|[0-9]+)",
        'lap_var': r"lap\s+var(?:iance)?(?:[:\s]*)?([0-9]+\.[0-9]+|[0-9]+)"
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            v = m.group(1)
            try:
                if '.' in v:
                    stats[key] = float(v)
                else:
                    stats[key] = int(v)
            except Exception:
                stats[key] = v

    return stats


def save_json(resp, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(resp, f, indent=2, ensure_ascii=False)
        print(f"Saved raw response to {path}")
    except Exception as e:
        print(f"Failed to save JSON to {path}: {e}")


def save_markdown(md_str, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md_str)
        print(f"Saved markdown to {path}")
    except Exception as e:
        print(f"Failed to save markdown to {path}: {e}")


def main():
    year = 2025
    print('Building cumulative features (this may take a while)...')
    X, y, driver_cum, races, race_results, race_laps = build_cumulative_features(year)
    print('Training and predicting...')
    res = train_and_predict(X, y, driver_cum, races, race_results, race_laps, year)

    if res is None:
        return

    # ask Mr. Watson for explanation
    if os.getenv('CALL_WATSON', '0') == '1':
        api_key = os.getenv('WATSON_API_KEY')
        if not api_key:
            print('No Watson API key found; set WATSON_API_KEY in .env file')
        else:
            try:
                iam_token = get_iam_token(api_key)
                top = res.head(3)
                features = top.to_dict(orient='records')
                winner = top.iloc[0]['driver']
                second = top.iloc[1]['driver'] if len(top) > 1 else ''
                third = top.iloc[2]['driver'] if len(top) > 2 else ''
                resp = call_watson(iam_token, winner, second, third, features)

                # Save raw response for auditing
                save_json(resp, 'watson_raw.json')

                # Extract assistant text and save a Markdown summary
                assistant_text = extract_assistant_text(resp)
                md = assistant_text_to_markdown('Watson Analysis', assistant_text)
                save_markdown(md, 'watson_analysis.md')

                # Try to extract numeric stats heuristically and save
                stats = parse_stats_from_text(assistant_text)
                if stats:
                    save_json(stats, 'watson_stats.json')

                # Print a concise view to console
                print('\nWatson Analysis (clean):')
                print(assistant_text)
                if stats:
                    print('\nExtracted stats:')
                    print(stats)

            except Exception as e:
                print(f'Error calling Watson: {e}')
                # attempt to save raw resp if available
                try:
                    save_json(resp, 'watson_raw.json')
                except Exception:
                    pass


if __name__ == '__main__':
    main()
