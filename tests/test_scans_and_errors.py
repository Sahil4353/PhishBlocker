import re
import types


def _post_scan(client, raw, subject=None, sender=None):
    return client.post(
        "/scan",
        data={"raw": raw, "subject": subject, "sender": sender},
        follow_redirects=False,
    )


def test_scans_json_endpoint_roundtrip(client):
    r = _post_scan(
        client,
        "urgent action required, click here to verify your account",
        subject="JsonRound",
        sender="jr@example.com",
    )
    assert r.status_code == 303
    loc = r.headers["location"]
    m = re.search(r"/scan/(\d+)/view", loc)
    assert m, f"could not parse id from {loc}"
    scan_id = int(m.group(1))

    # JSON detail
    r2 = client.get(f"/scan/{scan_id}")
    assert r2.status_code == 200
    data = r2.json()
    assert data["id"] == scan_id
    assert data["label"] in {"safe", "spam", "phishing"}
    assert isinstance(data.get("confidence"), float)
    assert isinstance(data.get("reasons", []), list)


def test_scan_detail_404(client):
    r = client.get("/scan/999999")
    assert r.status_code == 404


def test_request_id_header_on_error_and_template_message(client, monkeypatch):
    """
    Force the DB commit error path in POST /scan and ensure:
    - HTTP 500 returned
    - Response contains the error message from the template
    - X-Request-ID header is present
    """
    # Patch pages.SessionLocal to return a fake session whose commit() raises
    import app.api.routes.pages as pages

    class FakeSession:
        def add(self, *a, **k):
            pass

        def commit(self, *a, **k):
            raise RuntimeError("boom")

        def refresh(self, *a, **k):
            pass

        def rollback(self, *a, **k):
            pass

        def close(self, *a, **k):
            pass

    monkeypatch.setattr(pages, "SessionLocal", lambda: FakeSession())

    r = client.post(
        "/scan", data={"raw": "verify your account"}, follow_redirects=False
    )
    assert r.status_code == 500
    assert "database error while saving scan" in r.text.lower()
    assert r.headers.get("X-Request-ID")
