import yaml
import requests
import json
import time

# ============ CONFIGURATION ============
INPUT_YAML = "./data/papers_with_citations.yaml"
OUTPUT_JSON = "./data/citations_over_time.json"
PAPER_TITLE_KEY = "title"
API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SLEEP_BETWEEN_REQUESTS = 1.0
# ========================================

def get_citation_history(title):
    try:
        print(f"🔍 Searching for: {title}")

        params = {
            "query": title,
            "fields": "title,year,citationCount,citationStats",
            "limit": 1
        }
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(API_URL, params=params, headers=headers)

        if response.status_code != 200:
            print(f"⚠️ Failed to fetch Semantic Scholar data for {title}")
            return {}

        data = response.json()

        if not data.get("data"):
            print(f"⚠️ No results for {title}")
            return {}

        paper = data["data"][0]

        if "citationStats" not in paper or "citationsPerYear" not in paper["citationStats"]:
            print(f"⚠️ No citation per year stats for {title}")
            return {}

        history = {int(item["year"]): int(item["citationCount"]) for item in paper["citationStats"]["citationsPerYear"]}

        print(f"✅ Got data for {title}: {history}")
        return history

    except Exception as e:
        print(f"❌ Error processing {title}: {e}")
        return {}

# Step 1: Load YAML
with open(INPUT_YAML, "r", encoding="utf-8") as f:
    papers = yaml.safe_load(f)

titles = [paper[PAPER_TITLE_KEY] for paper in papers if PAPER_TITLE_KEY in paper]

# Step 2: Fetch citation histories
all_histories = []
for title in titles:
    history = get_citation_history(title)
    if history:
        all_histories.append(history)
    time.sleep(SLEEP_BETWEEN_REQUESTS)

# Step 3: Build the full year range
all_years = set()
for history in all_histories:
    all_years.update(history.keys())
all_years = sorted(all_years)

# Step 4: Sum citations per year
total_citations_per_year = {year: 0 for year in all_years}
for history in all_histories:
    for year, count in history.items():
        total_citations_per_year[year] += count

# Step 5: Build cumulative citations
cumulative_citations = {}
running_total = 0
for year in all_years:
    running_total += total_citations_per_year[year]
    cumulative_citations[year] = running_total

# Step 6: Save to JSON
output_data = {
    "years": list(cumulative_citations.keys()),
    "cumulative_citations": list(cumulative_citations.values())
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Saved cumulative citation data to {OUTPUT_JSON}")
