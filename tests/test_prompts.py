"""Unit tests for prompt formatting and constraints."""
from src.models import Chunk
from src.prompt_templates import qa_prompt, summary_prompt


def test_qa_prompt_structure():
    hits = [
        Chunk(
            text="Applicant must submit 10th and 12th marksheet along with caste certificate.",
            meta={"kind": "scheme", "source": "State Scholarship", "page": "Page 2"}
        )
    ]
    messages = [
        {"role": "user", "content": "What documents do I need?"},
        {"role": "assistant", "content": "Checking guidelines."}
    ]

    prompt = qa_prompt("Which documents are required?", hits, messages)
    assert "CORE BEHAVIOR RULES:" in prompt
    assert "State Scholarship" in prompt
    assert "Page 2" in prompt
    assert "Citizen: What documents do I need?" in prompt
    assert "USER INQUIRY:" in prompt


def test_summary_prompt_structure():
    user_chunks = [
        Chunk(
            text="Completed Bachelor of Technology in Computer Science with 8.5 CGPA in 2024.",
            meta={"kind": "user_doc", "source": "resume.pdf", "page": "Page 1"}
        )
    ]

    prompt = summary_prompt(user_chunks)
    assert "SEVEN distinct sections" in prompt
    assert "Not found in uploaded documents." in prompt
    assert "resume.pdf" in prompt
    assert "Page 1" in prompt
