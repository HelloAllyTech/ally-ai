import argparse
import json
import os
import sys
from pathlib import Path

import httpx

# Configuration
PROMPT_DIR = Path("app/prompts")

# Use ALLY_CORE__ENDPOINT and ALLY_CORE__API_KEY from .env
ALLY_BE_URL = os.getenv("ALLY_CORE__ENDPOINT")
SYNC_ENDPOINT = f"{ALLY_BE_URL.rstrip('/')}/api/v1/prompts/sync" if ALLY_BE_URL else None
API_TOKEN = os.getenv("ALLY_CORE__API_KEY")

PROMPT_CODE_PREFIX = "ally_ai_"


def scan_prompts():
    prompts = []
    print(f"Scanning {PROMPT_DIR} for prompts...")

    for txt_file in PROMPT_DIR.glob("**/*.txt"):
        # relative path from PROMPT_DIR
        rel_path = txt_file.relative_to(PROMPT_DIR)
        parts = rel_path.parts
        
        if len(parts) < 2:
            print(f"Skipping {txt_file}: Must be in a subdirectory (category/name.txt)")
            continue

        category = parts[0]
        name = Path(parts[-1]).stem
        
        # Combine subdirectories for the code if nested deeper than 1 level
        internal_path = "/".join(parts[:-1] + (name,))
        prompt_code = f"{PROMPT_CODE_PREFIX}{internal_path.replace('/', '_')}"
        content = txt_file.read_text(encoding="utf-8").strip()

        # Look for optional name/description from same dir's _meta/<stem>.meta.json
        meta_path = txt_file.parent / "_meta" / f"{txt_file.stem}.meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error reading meta for {txt_file}: {e}")

        prompts.append({
            "promptCode": prompt_code,
            "prompt": content,
            "name": meta.get("name", f"AI: {category.capitalize()} - {name.replace('_', ' ').capitalize()}"),
            "description": meta.get("description", f"Generated from ally-ai: {rel_path}"),
            "category": meta.get("category", category.upper()),
            "useDashboardOverride": meta.get("useDashboardOverride", False),
            "isDefault": meta.get("isDefault", True)
        })

    return prompts


def sync_prompts(prompts, dry_run=False):
    if not dry_run and (not SYNC_ENDPOINT or not API_TOKEN):
        print("Error: ALLY_CORE__ENDPOINT or ALLY_CORE__API_KEY not set in environment.")
        sys.exit(1)

    if dry_run:
        print("\n[DRY RUN] Would sync the following prompts:")
        for p in prompts:
            print(f" - {p['promptCode']} ({p['promptName']})")
        return

    print(f"\nSyncing {len(prompts)} prompts to {SYNC_ENDPOINT}...")
    
    payload = {"prompts": prompts}
    headers = {"x-api-key": API_TOKEN, "Content-Type": "application/json"}

    try:
        with httpx.Client() as client:
            response = client.post(SYNC_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            print(f"Successfully synced! {result.get('count', 0)} prompts processed.")
    except Exception as e:
        print(f"Error syncing prompts: {e}")
        if isinstance(e, httpx.HTTPStatusError):
            print(f"Response: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync ally-ai prompts to backend")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without syncing")
    args = parser.parse_args()

    prompts = scan_prompts()
    if not prompts:
        print("No prompts found to sync.")
    else:
        sync_prompts(prompts, dry_run=args.dry_run)
