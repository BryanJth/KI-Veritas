from types import SimpleNamespace
from veritas_core import (NO_EVIDENCE, is_list_fakultas_program_sekolah,
    contextualize_query, retrieve, public_sources, build_context, grounded_answer, detect_domain)

class DB:
    def __init__(self, entries): self.entries = entries
    def similarity_search_with_score(self, query, k): return self.entries[:k]

class BrokenDB:
    def similarity_search_with_score(self, query, k): raise RuntimeError('no scored search')


def doc(text, **meta): return SimpleNamespace(page_content=text, metadata=meta)


def test_fakultas_intent_does_not_hijack_program_query():
    assert not is_list_fakultas_program_sekolah('Apa program studi di FMIPA UI?')
    assert not is_list_fakultas_program_sekolah('Ceritakan sekolah di UI')
    assert is_list_fakultas_program_sekolah('Daftar fakultas, program, dan sekolah UI')


def test_retrieval_filters_bad_distances_and_duplicates():
    good = doc('Kampus utama UI berada di Depok.', source='kampus.pdf', page=2, chunk_id='x')
    bad = doc('Informasi yang tidak terkait.')
    dbs = {'umum': DB([(good, .5), (good, .5), (bad, 3.0)])}
    result = retrieve('di mana kampus?', 'umum', dbs.__getitem__, ['umum'], max_distance=1.2)
    assert len(result) == 1
    assert result[0].distance == .5
    assert public_sources(result) == [{'source':'kampus.pdf','page':2,'document_number':1}]
    assert '[Dokumen 1]' in build_context(result)


def test_broken_scored_search_does_not_invent_zero_distance():
    result = retrieve('foo', 'umum', lambda domain: BrokenDB(), ['umum'])
    assert result == []
    assert grounded_answer('Something made up', result) == NO_EVIDENCE


def test_fallback_only_when_no_evidence():
    dbs = {
        'fasilitas': DB([(doc('Perpustakaan UI.', source='ui.pdf',page=7),.4)]),
        'umum': DB([]),
        'dosen': BrokenDB(),
    }
    got = retrieve('Perpustakaan', 'fasilitas', dbs.__getitem__, ['fasilitas','umum','dosen'])
    assert len(got) == 1 and got[0].domain == 'fasilitas'


def test_invalid_page_never_claims_citation():
    db = DB([(doc('Old passage.', source='pdf.pdf',page='?',chunk_id='a'),.8)])
    got = retrieve('old', 'umum', lambda _:db, ['umum'])
    assert public_sources(got) == []


def test_followup_and_domain():
    assert 'FMIPA UI' in contextualize_query('Bagaimana dengan fakultas tersebut?', 'FMIPA UI')
    assert contextualize_query('Apa itu UI?', 'FMIPA UI') == 'Apa itu UI?'
    assert detect_domain('Bagaimana fasilitas perpustakaan?') == 'fasilitas'
    assert detect_domain('Profil dosen matematika') == 'dosen'
