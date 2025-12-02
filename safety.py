import re

ADVICE = re.compile(r'\b(you should|file a case|hire a lawyer|legal advice)\b', re.I)
PII_PATTERNS = [
    r"\b\d{4}\s\d{4}\s\d{4}\b",         # Aadhaar pattern
    r"\b[A-Z]{5}\d{4}[A-Z]\b",          # PAN
]

BANNED = ["terrorism", "fake notice", "hate speech"]

def check_safety(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    for w in BANNED:
        if w in low:
            return False
    if ADVICE.search(low):
        # You can either block or allow with disclaimer; blocking for now:
        return False
    for p in PII_PATTERNS:
        if re.search(p, text):
            # Redact or block; blocking for MVP:
            return False
    return True
