import re


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
    assert r.headers.get("X-Request-ID")


def test_get_scan_redirects_to_home(client):
    r = client.get("/scan", allow_redirects=False)
    assert r.status_code in (303, 307)  # we use 303
    assert r.headers["location"].endswith("/")


def test_prg_post_scan_redirects_to_detail(client):
    form = {
        "raw": "urgent action required, click here to verify your account",
        "subject": "Test PRG",
        "sender": "test@example.com",
    }
    r = client.post("/scan", data=form, allow_redirects=False)
    assert r.status_code == 303
    loc = r.headers.get("location", "")
    assert "/scan/" in loc and "/view" in loc
    assert "ok=1" in loc
    assert r.headers.get("X-Request-ID")


def test_follow_redirect_and_detail_html(client):
    # Post and then follow to the detail view
    r = client.post(
        "/scan",
        data={
            "raw": "verify your password",
            "subject": "Subj",
            "sender": "s@example.com",
        },
        allow_redirects=False,
    )
    loc = r.headers["location"]
    r2 = client.get(loc)
    assert r2.status_code == 200
    # HTML contains our header text
    assert "Scan Detail" in r2.text


def test_json_detail_matches_redirect_id(client):
    # Create a scan
    r = client.post(
        "/scan",
        data={"raw": "click here", "subject": "S", "sender": "a@b"},
        allow_redirects=False,
    )
    loc = r.headers["location"]  # e.g., /scan/3/view?ok=1
    m = re.search(r"/scan/(\d+)/view", loc)
    assert m, f"could not parse id from {loc}"
    scan_id = int(m.group(1))

    # Call JSON detail
    r2 = client.get(f"/scan/{scan_id}")
    assert r2.status_code == 200
    data = r2.json()
    assert data["id"] == scan_id
    assert data["label"] in {"safe", "spam", "phishing"}
    assert isinstance(data.get("confidence"), float)
    assert isinstance(data.get("reasons", []), list)


def test_request_id_header_present(client):
    r = client.get("/")
    rid = r.headers.get("X-Request-ID")
    assert rid and isinstance(rid, str) and len(rid) > 0
