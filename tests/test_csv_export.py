import csv
from datetime import date
from io import StringIO


def _post_scan(client, raw, subject=None, sender=None):
    return client.post(
        "/scan",
        data={"raw": raw, "subject": subject, "sender": sender},
        follow_redirects=False,
    )


def _read_csv(text: str):
    sio = StringIO(text)
    return list(csv.DictReader(sio))


def test_export_csv_all_columns_and_header(client):
    _post_scan(client, "hello just a note")  # safe
    _post_scan(client, "you are a WINNER of FREE money")  # spam
    _post_scan(
        client, "urgent action required, click here to verify your account"
    )  # phishing

    r = client.get("/export/csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "").lower()
    rows = _read_csv(r.text)
    assert len(rows) >= 3

    hdr = rows[0].keys()
    for col in [
        "id",
        "created_at",
        "subject",
        "sender",
        "label",
        "confidence",
        "reasons",
        "body_preview",
    ]:
        assert col in hdr


def test_export_csv_label_filter(client):
    _post_scan(client, "verify your password now")
    r = client.get("/export/csv?label=phishing")
    assert r.status_code == 200
    rows = _read_csv(r.text)
    assert len(rows) >= 1
    assert all(row["label"] == "phishing" for row in rows)


def test_export_csv_date_range(client):
    today = date.today().isoformat()
    r = client.get(f"/export/csv?date_from={today}&date_to={today}")
    assert r.status_code == 200
    rows = _read_csv(r.text)
    assert isinstance(rows, list)
    if rows:
        assert set(["id", "created_at"]).issubset(rows[0].keys())


def test_export_csv_excel_injection_guard(client):
    # Leading = + - @ should be prefixed in CSV to avoid Excel formula execution
    subj = '=HYPERLINK("http://evil")'
    sndr = "+123@example.com"
    # Start raw with '=' so the cell's first char is risky; keep a phish trigger in text
    body = "=malicious body verify your account"
    # Post exactly this as raw so body_preview also starts with '='
    _post_scan(client, body, subject=subj, sender=sndr)

    r = client.get("/export/csv?label=phishing")
    assert r.status_code == 200
    rows = _read_csv(r.text)
    assert len(rows) >= 1

    victim = None
    for row in rows:
        if "malicious body" in row.get("body_preview", ""):
            victim = row
            break
    assert victim, "Expected to find the row with our malicious body_preview"

    # Guard check: fields should be prefixed with a single quote if they start with = + - @
    assert victim["subject"].startswith("'")
    assert victim["sender"].startswith("'")
    assert victim["body_preview"].startswith("'")
