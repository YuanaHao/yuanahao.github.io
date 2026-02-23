from scholarly import scholarly
import jsonpickle
import json
from datetime import datetime
import os


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_total_citations(author):
    # `scholarly` responses are not always stable across versions/sections.
    # Ensure we always emit a total citation count for downstream consumers.
    if 'citedby' in author:
        return _to_int(author.get('citedby'))

    cites_per_year = author.get('cites_per_year')
    if isinstance(cites_per_year, dict):
        return sum(_to_int(v) for v in cites_per_year.values())

    publications = author.get('publications')
    if isinstance(publications, list):
        return sum(_to_int(p.get('num_citations')) for p in publications if isinstance(p, dict))
    if isinstance(publications, dict):
        return sum(_to_int(p.get('num_citations')) for p in publications.values() if isinstance(p, dict))

    return 0


author: dict = scholarly.search_author_id(os.environ['GOOGLE_SCHOLAR_ID'])
scholarly.fill(author, sections=['basics', 'indices', 'counts', 'publications'])
name = author['name']
author['updated'] = str(datetime.now())
author['publications'] = {v['author_pub_id']:v for v in author['publications']}
author['citedby'] = _resolve_total_citations(author)
print(json.dumps(author, indent=2))
os.makedirs('results', exist_ok=True)
with open(f'results/gs_data.json', 'w') as outfile:
    json.dump(author, outfile, ensure_ascii=False)

shieldio_data = {
  "schemaVersion": 1,
  "label": "citations",
  "message": f"{author['citedby']}",
}
with open(f'results/gs_data_shieldsio.json', 'w') as outfile:
    json.dump(shieldio_data, outfile, ensure_ascii=False)
