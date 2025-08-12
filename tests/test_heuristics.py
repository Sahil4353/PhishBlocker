import pytest

from app.services.heuristics import classify_text


def test_phishing_has_priority_over_spam():
    text = (
        "Urgent action required! Click here to verify your account and you are a winner"
    )
    label, conf, reasons = classify_text(text)
    assert label == "phishing"
    assert conf >= 0.70
    assert any("verify your" in r or "urgent action required" in r for r in reasons)


def test_spam_detected_when_no_phish_patterns():
    text = "You are a WINNER of FREE money, work from home!"
    label, conf, reasons = classify_text(text)
    assert label == "spam"
    assert conf >= 0.60
    assert any("winner" in r or "free\\s+money" in r for r in reasons)


def test_safe_when_no_patterns():
    label, conf, reasons = classify_text("hello there, just checking in")
    assert label == "safe"
    assert conf == 0.50
    assert "no suspicious patterns found" in reasons[0].lower()


@pytest.mark.parametrize(
    "hits,expected_min_conf",
    [
        (1, 0.70),
        (2, 0.80),
        (3, 0.90),
    ],
)
def test_confidence_scales_with_phish_hits(hits, expected_min_conf):
    # Build text that triggers N phishing patterns
    parts = [
        "verify your account",
        "urgent action required",
        "click here",
        "confirm your identity",
    ][:hits]
    text = " -- ".join(parts)
    label, conf, _ = classify_text(text)
    assert label == "phishing"
    assert conf >= expected_min_conf
    assert conf <= 0.99


def test_large_input_does_not_crash():
    big = ("hello " * 10000) + " verify your password " + ("world " * 10000)
    label, conf, reasons = classify_text(big)
    assert label == "phishing"
    assert conf >= 0.70
