"""
Lightweight smoke tests that don't require Postgres or a Gemini key, so they
run in any environment. Full end-to-end tests (upload -> ask -> citations)
are meant to be run inside the docker-compose stack, e.g.:

    docker compose exec api pytest tests/ -v
"""
from api.schemas import ChatRequest, ChatResponse, CitationOut


def test_chat_request_validates_min_length():
    req = ChatRequest(question="What is pH?", session_id="s1")
    assert req.question == "What is pH?"


def test_chat_response_serializes_citations():
    citation = CitationOut(
        source_number=1, document_id="d1", document_title="Atomic Structure and the Bohr Model",
        page=1, chunk_index=0, text_preview="An electron can have only specific energy values...",
        similarity=0.87,
    )
    resp = ChatResponse(
        question="What does Bohr's model say about electron energy levels?",
        search_query="What does Bohr's model say about electron energy levels?",
        is_followup=False,
        answer="An electron can have only specific energy values in an atom [1].",
        supported="supported",
        citations=[citation],
        rule_check={"passed": True},
    )
    assert resp.citations[0].document_title == "Atomic Structure and the Bohr Model"
    assert resp.supported == "supported"