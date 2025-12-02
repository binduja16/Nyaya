import re
from dateutil import parser

def extract_facts(text: str):
    facts = {
        "Parties": [],
        "Amounts": [],
        "Dates": []
    }

    # --- Fix OCR issue: replace 'n' with ₹ only when it's actually money ---
    # (e.g., n5,00,000 → ₹5,00,000, but "Section 302" stays untouched)
    text = re.sub(r"(?<!section\s)n(?=\d)", "₹", text, flags=re.IGNORECASE)

    # --- Parties (names + organizations + keywords like "FIR", "Bank") ---
    party_candidates = re.findall(
        r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|Shri|Smt\.)?\s?[A-Z][a-z]+\s[A-Z][a-z]+",
        text
    )
    org_candidates = re.findall(
        r"(Police Station|Court|Bank|Authority|Company|Ltd|LLP|Society)",
        text,
        re.IGNORECASE
    )
    facts["Parties"] = list(set(party_candidates + org_candidates))

    # --- Amounts (₹, Rs., INR, digits with lakh/crore, also OCR 'n') ---
    amount_pattern = r"(?:Rs\.?|INR|₹)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|crore))?"
    facts["Amounts"] = re.findall(amount_pattern, text, re.IGNORECASE)

    # --- Dates (dd-mm-yyyy, dd/mm/yyyy, 10 August 2025, etc.) ---
    date_pattern = (
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\d{1,2}\s?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s?\d{2,4})\b"
    )
    raw_dates = re.findall(date_pattern, text, re.IGNORECASE)

    parsed_dates = []
    for d in raw_dates:
        try:
            parsed_dates.append(str(parser.parse(d, dayfirst=True).date()))
        except:
            pass
    facts["Dates"] = parsed_dates

    return facts
