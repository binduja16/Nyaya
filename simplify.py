from langdetect import detect

def simplify_text(text: str) -> str:
    """
    Lightweight summarizer (fallback): first ~3 sentences or 500 chars.
    (Avoids heavy model to keep server snappy.)
    """
    if not text:
        return "No content extracted."
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    # Simple heuristic summary
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    head = ". ".join(sentences[:3]) + "."
    if len(head) < 120 and len(text) > 120:
        head = text[:500] + ("..." if len(text) > 500 else "")
    return head
