"""Prompt engineering templates enforcing hallucination-resistant, grounded generation."""
from typing import Dict, List

from src.config import MAX_HISTORY_TURNS
from src.models import Chunk


def qa_prompt(
    question: str,
    hits: List[Chunk],
    messages: List[Dict[str, str]],
    history_limit: int = MAX_HISTORY_TURNS
) -> str:
    """Constructs a grounded RAG prompt adhering to IEEE evaluation criteria.
    
    Includes:
    - Strict adherence to retrieved context (anti-hallucination instruction).
    - Automatic available vs. missing document cross-referencing rule.
    - Preserved filename and page citations.
    - Multi-turn conversation context window.
    
    Args:
        question: User query string.
        hits: Ranked retrieved context passages.
        messages: Conversation history log.
        history_limit: Number of recent turns to include.
        
    Returns:
        Structured system prompt string ready for LLM consumption.
    """
    context_blocks = []
    for i, h in enumerate(hits):
        src = h.meta.get("source", "Unknown")
        pg = h.meta.get("page", "")
        kind = "User Document" if h.meta.get("kind") == "user_doc" else "Government Scheme"
        context_blocks.append(f"[{i+1}] ({kind} | Source: {src} | {pg})\n{h.text}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

    history_str = ""
    if messages:
        recent = messages[-history_limit:]
        history_lines = []
        for m in recent:
            role = "Citizen" if m.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {m.get('content', '')}")
        history_str = f"Conversation History (Last {len(recent)} turns):\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""You are the AI Citizen Assistant, an authoritative, helpful public assistant designed to help citizens understand government schemes, scholarships, and their personal credentials.

CORE BEHAVIOR RULES:
1. Grounding: Answer strictly using facts supported by the CONTEXT below. Do NOT hallucinate, guess, or invent unverified facts.
2. If the question cannot be answered from the provided Context, state explicitly:
   "This information could not be found in the uploaded documents or scheme dataset."
3. Scheme Comparison Rule: When the citizen asks about applying for a scheme or which documents they need or have:
   - Carefully cross-reference the requirements in the scheme against the citizen's uploaded documents.
   - List which documents are AVAILABLE (mentioning the exact uploaded file and page).
   - List which documents are MISSING.
4. Bilingual Support: Answer in the same language as the user's inquiry (English, Hindi, or mixed Hinglish).
5. Sources & Citations: Always mention the relevant document name and page number for facts stated.

{history_str}CONTEXT:
{context_str}

USER INQUIRY:
{question}

ANSWER:"""
    return prompt


def summary_prompt(user_chunks: List[Chunk]) -> str:
    """Builds a structured 7-section personal profile summary prompt.
    
    Sections:
    1. Personal Information
    2. Educational Background
    3. Skills & Competencies
    4. Certifications & Credentials
    5. Projects & Work Experience
    6. Achievements & Awards
    7. Important Dates & Validity
    
    Args:
        user_chunks: All chunks extracted from the citizen's uploaded documents.
        
    Returns:
        Summary extraction prompt string.
    """
    combined_docs = "\n\n".join(
        f"[File: {c.meta.get('source', 'Unknown')} | {c.meta.get('page', '')}]\n{c.text}"
        for c in user_chunks
    )

    prompt = f"""You are the AI Citizen Assistant. Provide a rigorous, structured personal document summary from the citizen's uploaded credentials.

Extract information ONLY from the provided text and organize it into these SEVEN distinct sections:
1. Personal Information (Name, Contact Information, Address, Government Identifiers)
2. Educational Background (Institutions, Degrees/Certificates, Grades/Percentages, Passing Years)
3. Skills & Competencies (Technical skills, Domain skills, Languages known)
4. Certifications & Credentials (Course certifications, Issuing organizations, Credential IDs, Dates)
5. Projects & Work Experience (Roles, Organizations, Key contributions)
6. Achievements & Awards (Honors, Competitions, Scholarships awarded)
7. Important Dates & Validity (Document issue dates, Expiry dates, Dates of birth)

CRITICAL RULE:
For any section where information is not present in the uploaded documents, write exactly:
"Not found in uploaded documents."
Do NOT extrapolate, infer, or hallucinate missing information.

UPLOADED DOCUMENTS CONTENT:
{combined_docs}

STRUCTURED DOCUMENT SUMMARY:"""
    return prompt
