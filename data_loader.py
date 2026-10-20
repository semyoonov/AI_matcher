import re
import os
from pathlib import Path
from typing import List, Dict
from models import RawCandidate
from config import settings

def load_raw_candidates(file_path: Path = None) -> List[RawCandidate]:
    """
    Parses PostgreSQL INSERT statements from resumes_domains_rows.sql
    extracting candidate id and domain lists.
    """
    path = file_path or settings.DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mock data SQL file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to match: ('<id>', '<escaped_string>')
    pattern = re.compile(r"\('(\d+)',\s*'((?:''|[^'])*)'\)")
    candidates: List[RawCandidate] = []

    for m in pattern.finditer(content):
        cid = int(m.group(1))
        payload = m.group(2).replace("''", "'").strip()

        # Clean outer braces and quotes
        if payload.startswith('"'):
            payload = payload[1:]
        if payload.endswith('"'):
            payload = payload[:-1]
        if payload.startswith('{'):
            payload = payload[1:]
        if payload.endswith('}'):
            payload = payload[:-1]

        payload = payload.replace('\\"', '"')

        # Extract quoted strings: "item1","item2"
        items = re.findall(r'"([^"]+)"', payload)
        if not items:
            items = [x.strip() for x in payload.split(',') if x.strip()]

        # Clean unicode characters (non-breaking hyphen \u2011, non-breaking space \xa0)
        clean_items = []
        for it in items:
            cleaned = it.replace('\u2011', '-').replace('\xa0', ' ').strip()
            if cleaned:
                clean_items.append(cleaned)

        candidates.append(RawCandidate(id=cid, raw_domains=clean_items))

    # Sort by ID
    candidates.sort(key=lambda c: c.id)
    return candidates

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    cands = load_raw_candidates()
    print(f"Successfully loaded {len(cands)} candidates from SQL mock data.")
    print("Sample Candidate #79:", [c for c in cands if c.id == 79][0].raw_domains)
