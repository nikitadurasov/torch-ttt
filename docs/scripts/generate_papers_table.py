import yaml

with open("./data/papers_with_citations.yaml", "r", encoding="utf-8") as f:
    papers = yaml.safe_load(f)

# Sort by citations descending by default
papers = sorted(papers, key=lambda p: p.get("citations", 0), reverse=True)

html = """
<style>
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    transition: background-color 0.3s, color 0.3s;
  }

  .controls {
    margin-bottom: 30px;
    text-align: center;
  }

  .sort-select {
    padding: 8px 12px;
    font-size: 14px;
    border: 1px solid #ccc;
    border-radius: 5px;
    background-color: #f0f0f0;
    color: #333;
    transition: background-color 0.3s, color 0.3s;
  }

  .papers-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
  }

  .paper-card {
    border-radius: 12px;
    padding: 20px;
    background-color: #ffffff;
    color: #000000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s, box-shadow 0.2s, background-color 0.3s;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .paper-card:hover {
    transform: translateY(-5px);
    background-color: #e0f7fa;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    cursor: pointer;
  }

  .paper-top {
    flex: 1;
  }

  .paper-title {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
  }

  .paper-authors {
    font-size: 14px;
    margin-bottom: 8px;
  }

  .paper-bottom {
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 70px;
  }

  .paper-venue, .paper-links, .paper-citations {
    font-size: 14px;
  }

  .paper-links {
    display: flex;
    gap: 10px;
  }

  .paper-links a {
    color: #4285f4;
  }
  
  .paper-links a:visited {
    color: #4285f4;
  }

  /* DARK MODE */
  @media (prefers-color-scheme: dark) {
    body {
      background-color: #0d0d0d;
      color: #e0e0e0;
    }

    .sort-select {
      background-color: #1e1e1e;
      color: #e0e0e0;
      border: 1px solid #444;
    }

    .paper-card {
      background-color: #1e1e1e;
      color: #e0e0e0;
      box-shadow: 0 2px 8px rgba(0,0,0,0.6);
    }

    .paper-card:hover {
      background-color: #263238;
      box-shadow: 0 4px 12px rgba(0,0,0,0.8);
      cursor: pointer;
    }

    .paper-citations {
      color: #66bb6a;
    }

    .paper-links a {
      color: #82b1ff;
      text-decoration: none;
    }
    .paper-links a:visited {
      color: #82b1ff;
      text-decoration: none;
    }
    .paper-links a:hover {
      color: #82b1ff;
      text-decoration: none;
    }
    .paper-links a:active {
      color: #82b1ff;
      text-decoration: none;
    }
    .paper-links a, .paper-links a * {
      cursor: auto;
    }
  }
</style>

<div class="controls" style="padding: 30px;">
  <label for="sortSelect">Sort by:</label>
  <select id="sortSelect" class="sort-select" onchange="sortPapers()">
    <option value="citations">Citations (high to low)</option>
    <option value="year">Year (newest first)</option>
    <option value="conference">Conference (A-Z)</option>
    <option value="title">Title (A-Z)</option>
  </select>
</div>

<div class="papers-container" id="papersContainer">
"""

for paper in papers:
    if paper.get("venue"):
        parts = paper["venue"].rsplit(" ", 1)
        conference = parts[0]
        year = parts[1] if len(parts) > 1 and parts[1].isdigit() else "N/A"
    else:
        conference, year = "N/A", "N/A"

    paper['conference'] = conference
    paper['year'] = year

    links = []
    if 'paper' in paper and paper['paper']:
        links.append(f'<a href="{paper["paper"]}" target="_blank">paper</a>')
    if 'code' in paper and paper['code']:
        links.append(f'<a href="{paper["code"]}" target="_blank">code</a>')
    links_html = " | ".join(links)

    html += f"""
    <div class="paper-card" 
         data-title="{paper['title']}" 
         data-authors="{paper['authors']}" 
         data-conference="{conference}" 
         data-year="{year}" 
         data-citations="{paper['citations']}">
      <div class="paper-top">
        <div class="paper-title">{paper['title']}</div>
        <div class="paper-authors">{paper['authors']}</div>
      </div>
      <div class="paper-bottom">
        <div class="paper-venue">{conference} ({year})</div>
        {'<div class="paper-links">' + links_html + '</div>' if links_html else ''}
        <div class="paper-citations">Citations: {paper['citations']}</div>
      </div>
    </div>
    """

html += """
</div>

<script>
function sortPapers() {
  const container = document.getElementById('papersContainer');
  const cards = Array.from(container.getElementsByClassName('paper-card'));
  const sortValue = document.getElementById('sortSelect').value;

  cards.sort((a, b) => {
    if (sortValue === 'citations') {
      return parseInt(b.dataset.citations) - parseInt(a.dataset.citations);
    } else if (sortValue === 'year') {
      return parseInt(b.dataset.year) - parseInt(a.dataset.year);
    } else if (sortValue === 'conference') {
      return a.dataset.conference.localeCompare(b.dataset.conference);
    } else if (sortValue === 'title') {
      return a.dataset.title.localeCompare(b.dataset.title);
    }
  });

  container.innerHTML = '';
  cards.forEach(card => container.appendChild(card));
}
</script>
"""

with open("./data/papers_table.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Fully aligned, polished adaptive papers page saved to ./data/papers_table.html")
