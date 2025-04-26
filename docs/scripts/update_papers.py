from scholarly import scholarly
import yaml

def update_citations(papers):
    updated = []
    for paper in papers:
        print(f"Searching: {paper['title']}")
        search_query = scholarly.search_pubs(paper["title"])
        try:
            result = next(search_query)
            paper["citations"] = result.get("num_citations", 0)
        except StopIteration:
            paper["citations"] = "Not found"
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

