# from scholarly import scholarly
# import yaml

# def update_citations(papers):
#     updated = []
#     for paper in papers:
#         print(f"Searching: {paper['title']}")
#         search_query = scholarly.search_pubs(paper["title"])
#         try:
#             result = next(search_query)
#             paper["citations"] = result.get("num_citations", 0)
#         except StopIteration:
#             paper["citations"] = "Not found"
#         updated.append(paper)
#     return updated

# if __name__ == "__main__":
#     # 🔽 Load papers from file
#     with open("./data/papers.yaml", "r", encoding="utf-8") as f:
#         papers = yaml.safe_load(f)

#     # 🔄 Update citation info
#     updated_papers = update_citations(papers)

#     # 💾 Save to new file
#     with open("./data/papers_with_citations.yaml", "w", encoding="utf-8") as f:
#         yaml.dump(updated_papers, f, allow_unicode=True, sort_keys=False)

#     print("✅ Saved to papers_with_citations.yaml")

from scholarly import scholarly
import yaml
import requests
import time
from bs4 import BeautifulSoup

def fetch_gsc_spans(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/112.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    year_spans = soup.find_all('span', class_='gsc_oci_g_t')
    citation_spans = soup.find_all('span', class_='gsc_oci_g_al')
    
    return {int(year.text.strip()): int(citations.text.strip()) 
            for year, citations in zip(year_spans, citation_spans)}

def update_citations(papers):
    updated = []
    for paper in papers:
        time.sleep(1)
        print(f"Processing: {paper['title']}")
        try:
            if "google_scholar_link" in paper and paper["google_scholar_link"]:
                print(f"  ↳ Fetching from Google Scholar link")
                citations_per_year = fetch_gsc_spans(paper["google_scholar_link"])
                total_citations = sum(citations_per_year.values())
                paper["citations"] = total_citations
                paper["citations_per_year"] = citations_per_year
            else:
                print(f"  ↳ Searching by title with scholarly")
                search_query = scholarly.search_pubs(paper["title"])
                result = next(search_query)
                paper["citations"] = result.get("num_citations", 0)
                paper["citations_per_year"] = None
        except Exception as e:
            print(f"⚠️ Error processing {paper['title']}: {e}")
            paper["citations"] = "Not found"
            paper["citations_per_year"] = None
        updated.append(paper)
    return updated

if __name__ == "__main__":
    # 🔽 Load papers from file
    with open("./data/papers.yaml", "r", encoding="utf-8") as f:
        papers = yaml.safe_load(f)

    # 🔄 Update citation info
    updated_papers = update_citations(papers)

    # 💾 Save to new file
    with open("./data/papers_with_citations.yaml", "w", encoding="utf-8") as f:
        yaml.dump(updated_papers, f, allow_unicode=True, sort_keys=False)

    print("✅ Saved to papers_with_citations.yaml")


