"""Regression tests for the screenshot-reported lecturer-list bug."""
from pathlib import Path
import json

from lecturer_directory import lecturer_directory_answer, DIRECTORY_JSON, known_lecturer_in_text
from veritas_core import lecturer_list_scope, is_explicit_followup, contextualize_query


def test_broad_university_lecturer_lists_need_scope():
    for question in (
        "siapa aja dosen ui",
        "siapa dosen ui",
        "siapa saja dosen Universitas Indonesia?",
        "daftar dosen UI",
        "dosen UI siapa saja?",
        "sebutkan nama dosen ui",
    ):
        assert lecturer_list_scope(question) == "ui", question


def test_maths_scope_and_other_departments():
    assert lecturer_list_scope("siapa aja dosen matematika UI?") == "matematika"
    assert lecturer_list_scope("daftar dosen Departemen Matematika FMIPA") == "matematika"
    assert lecturer_list_scope("siapa saja dosen FMIPA UI?") == "other"


def test_research_questions_and_named_lecturers_are_not_directory_requests():
    for q in (
        "Dosen Matematika UI yang meneliti machine learning siapa?",
        "siapa dosen ui yang ahli bioinformatika?",
        "siapa dosen UI Zuherman Rustam?",
        "apa keahlian dosen Hendri Murfi?",
        "siapa Prof. Alhadi Bustamam?",
    ):
        assert lecturer_list_scope(q) is None, q


def test_repeated_short_question_does_not_become_a_followup():
    assert not is_explicit_followup("siapa dosen ui")
    assert contextualize_query("siapa dosen ui", "siapa dosen ui") == "siapa dosen ui"
    assert is_explicit_followup("bagaimana dengan fakultas tersebut?")


def test_directory_discloses_partial_coverage_and_cites_verified_pdf_pages():
    answer, sources = lecturer_directory_answer()
    assert "bukan daftar lengkap" in answer.lower()
    assert "Hendri Murfi" in answer
    assert "Zuherman Rustam" in answer
    assert {s["page"] for s in sources} == {1, 2}
    assert all(s["source"] == "dosen_official_UI_2026.pdf" for s in sources)
    d = json.loads(DIRECTORY_JSON.read_text(encoding="utf8"))
    assert len(d["people"]) == 6
    assert len({p["name"] for p in d["people"]}) == 6
    assert all(p.get("source_url", "").startswith("https://math.ui.ac.id/") for p in d["people"])


def test_missing_directory_never_invents_professor_list(tmp_path: Path):
    text, sources = lecturer_directory_answer(tmp_path / "no.json")
    assert sources == []
    assert "belum" in text.lower()


def test_exact_name_routing_can_use_verified_profiles():
    assert known_lecturer_in_text("Apa riset Hendri Murfi di UI?") == "Hendri Murfi"
    assert known_lecturer_in_text("bandingkan Dipo Aldila dan Hendri Murfi") in {"Dipo Aldila", "Hendri Murfi"}
    assert known_lecturer_in_text("dosen universitas indonesia") == ""
    assert known_lecturer_in_text("Hendri") == ""
