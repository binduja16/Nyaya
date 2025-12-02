# ------------------ Petition generation helpers & endpoint ------------------

from pydantic import BaseModel
from typing import List, Optional

class PetitionRequest(BaseModel):
    case_type: str                      # e.g., "Cheating / Loan recovery"
    ipc_sections: Optional[List[str]]   # e.g., ["420", "406"]
    facts: str                          # short summary (3-8 lines)
    petitioner_name: Optional[str] = "Petitioner"
    respondent_name: Optional[str] = "Respondent"
    court_name: Optional[str] = "Hon'ble Court"
    city: Optional[str] = "City"
    reliefs: Optional[List[str]] = None
    language: Optional[str] = "en"      # optional: 'en' or other supported lang

def local_petition_template(req: PetitionRequest) -> str:
    """Create a simple petition draft locally (fallback)."""
    ipc_line = ""
    if req.ipc_sections:
        ipc_line = "The acts complained of attract the following provisions: " + ", ".join(req.ipc_sections) + ".\n\n"
    reliefs_text = ""
    if req.reliefs:
        reliefs_text = "\n".join([f"{idx+1}. {r}" for idx, r in enumerate(req.reliefs)])
    else:
        reliefs_text = "1. Pass such orders as this Hon'ble Court may deem fit and proper.\n"
    draft = f"""IN THE COURT OF {req.court_name}
AT {req.city}

{req.petitioner_name}
    ...Petitioner

vs.

{req.respondent_name}
    ...Respondent

PETITION UNDER RELEVANT PROVISIONS

MOST RESPECTFULLY SHOWETH:

1. That the petitioner is {req.petitioner_name} and resides at [address to be filled].
2. That the respondent is {req.respondent_name} and resides at [address to be filled].

BRIEF FACTS:
{req.facts}

{ipc_line}
CAUSE OF ACTION:
The cause of action arose on the dates and facts stated above and is within the jurisdiction of this Hon'ble Court.

GROUNDS:
1. The respondent has committed acts as narrated above which are contrary to law and attract liability.
2. The petitioner is entitled to the reliefs sought.

RELIEFS SOUGHT:
{reliefs_text}

PRAYER:
In view of the facts and circumstances stated above, it is most respectfully prayed that this Hon'ble Court may be pleased to:
{reliefs_text}

VERIFICATION:
I, {req.petitioner_name}, do hereby verify that the contents of the above petition are true and correct to the best of my knowledge and belief.

Place: {req.city}
Date: [Date]

Petitioner
(Signature)
Name: {req.petitioner_name}

Advocate for Petitioner
(Name & Enrollment No.)
"""
    return draft

def build_gemini_prompt(req: PetitionRequest) -> str:
    """Create a prompt for Gemini to generate a formal petition in Indian format."""
    ipc_part = ""
    if req.ipc_sections:
        ipc_part = "Relevant IPC/Statute sections: " + ", ".join(req.ipc_sections) + ".\n"
    reliefs_part = ""
    if req.reliefs:
        reliefs_part = "Reliefs desired:\n" + "\n".join([f"- {r}" for r in req.reliefs]) + "\n"
    prompt = f"""
You are a legal drafting assistant familiar with Indian court formats.
Generate a formal DRAFT PETITION (polished, 200-350 words) in Indian format (Court heading, Parties, Petition title, Facts, Cause of Action, Grounds, Reliefs, Prayer, Verification).
Case Type: {req.case_type}
{ipc_part}
Facts: {req.facts}
Petitioner: {req.petitioner_name}
Respondent: {req.respondent_name}
City / Court: {req.city} / {req.court_name}
{reliefs_part}
Output ONLY the draft petition text (no JSON, no extra commentary).
"""
    return prompt

@app.post("/api/generate-petition")
async def generate_petition_endpoint(payload: PetitionRequest):
    """
    Generate a draft petition.
    Tries Gemini first; if it fails, uses a local template.
    Returns JSON: { "draft": "<text>" }
    """
    try:
        # Try Gemini (if call_gemini available)
        try:
            prompt = build_gemini_prompt(payload)
            gemini_output = await call_gemini(prompt)
            # Basic sanity check
            if gemini_output and len(gemini_output.split()) > 80:
                draft_text = gemini_output.strip()
            else:
                raise Exception("Gemini returned empty or short output.")
        except Exception as e:
            # Fallback to local template
            draft_text = local_petition_template(payload)

        # Optional translation if requested language != 'en'
        if payload.language and payload.language != 'en':
            try:
                translated = await translation_service.translate_text(draft_text, payload.language)
                draft_text = translated
            except Exception:
                pass

        return JSONResponse({"draft": draft_text, "language": payload.language})
    except Exception as ex:
        logger.error(f"generate_petition_endpoint error: {ex}")
        return JSONResponse({"error": "Failed to generate petition"}, status_code=500)
