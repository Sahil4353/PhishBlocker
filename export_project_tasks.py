# "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"
import os
import requests
import csv
import time

# --- CONFIGURATION ---
TOKEN = "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"
OWNER = "Sahil4353"
REPO = "PhishBlocker"
PROJECT_NUMBER = 1
OUTPUT_CSV = "project_items_all_repo.csv"

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}"
}

GRAPHQL_URL = "https://api.github.com/graphql"

QUERY = """
query($owner:String!, $repo:String!, $number:Int!, $after:String) {
  repository(owner:$owner, name:$repo) {
    projectV1(number:$number) {
      items(first:100, after:$after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          content {
            __typename
            ... on Issue {
              number
              title
              assignees(first:10) { nodes { login } }
            }
            ... on PullRequest {
              number
              title
              assignees(first:10) { nodes { login } }
            }
          }
          fieldValues(first:20) {
            nodes {
              name
              ... on ProjectV2ItemFieldUserValue {
                users { nodes { login } }
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
              }
              ... on ProjectV2ItemFieldDateValue {
                date
              }
              ... on ProjectV2ItemFieldNumberValue {
                number
              }
              ... on ProjectV2ItemFieldTextValue {
                text
              }
            }
          }
        }
      }
    }
  }
}
"""


def run_query(variables):
    resp = requests.post(GRAPHQL_URL, json={
                         "query": QUERY, "variables": variables}, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def fetch_all_items():
    items = []
    cursor = None
    while True:
        variables = {"owner": OWNER, "repo": REPO,
                     "number": PROJECT_NUMBER, "after": cursor}
        resp = run_query(variables)
        data = resp.get("data", {})
        repo_data = data.get("repository", {})
        proj = repo_data.get("projectV2")
        if not proj:
            print("Error: projectV2 not found. Response:", resp)
            break
        page = proj["items"]
        nodes = page["nodes"]
        items.extend(nodes)
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
        time.sleep(0.3)
    return items


def write_to_csv(items):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf‑8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "number", "title",
                        "assignees", "field_names", "field_values"])
        for it in items:
            content = it.get("content") or {}
            ctype = content.get("__typename", "")
            num = content.get("number", "")
            title = content.get("title", "")
            assignees = []
            if "assignees" in content and content["assignees"]:
                assignees = [n["login"] for n in content["assignees"]["nodes"]]
            field_names = []
            field_values = []
            for fv in it.get("fieldValues", {}).get("nodes", []):
                fname = fv.get("name")
                if not fname:
                    continue
                field_names.append(fname)
                # pick correct value
                if "users" in fv and fv["users"]:
                    vals = [u["login"] for u in fv["users"]["nodes"]]
                    field_values.append(",".join(vals))
                elif "name" in fv and not fv.get("users"):
                    field_values.append(fv["name"])
                elif "date" in fv:
                    field_values.append(fv["date"])
                elif "number" in fv:
                    field_values.append(str(fv["number"]))
                elif "text" in fv:
                    field_values.append(fv["text"])
                else:
                    field_values.append("")
            writer.writerow([
                it.get("id", ""),
                ctype,
                num,
                title,
                ";".join(assignees),
                ";".join(field_names),
                ";".join(field_values)
            ])
    print(f"Wrote {len(items)} items to {OUTPUT_CSV}")


if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: Set GITHUB_TOKEN environment variable.")
    else:
        print("Fetching all project items …")
        items = fetch_all_items()
        print(f"Total items fetched: {len(items)}")
        write_to_csv(items)
