import re


def _post_scan(client, raw, subject=None, sender=None):
    return client.post(
        "/scan",
        data={"raw": raw, "subject": subject, "sender": sender},
        follow_redirects=False,
    )


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'name="raw"' in r.text  # textarea input present
    assert r.headers.get("X-Request-ID")


def test_history_pagination_basic(client):
    # seed > 20 items so we get at least 3 pages (default page_size=10)
    for i in range(23):
        _post_scan(client, f"message {i} click here" if i % 2 else f"message {i}")

    r1 = client.get("/history?page=1&page_size=10")
    r2 = client.get("/history?page=2&page_size=10")
    r3 = client.get("/history?page=3&page_size=10")

    assert r1.status_code == r2.status_code == r3.status_code == 200
    assert "Page 1 / 3" in r1.text
    assert "Page 2 / 3" in r2.text
    assert "Page 3 / 3" in r3.text
    assert "No scans yet" not in r1.text


def test_debug_scan_count_and_cwd(client):
    r = client.get("/_debug/scan_count")
    assert r.status_code == 200
    data = r.json()
    assert "count" in data and isinstance(data["count"], int)

    r2 = client.get("/_debug/cwd")
    assert r2.status_code == 200
    assert "cwd" in r2.json()
