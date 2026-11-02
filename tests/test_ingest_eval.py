from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from data import split_text, page_domains, iter_pdf_chunks, index_pdf
from eval_rag import compute_metrics, aggregate, load_dataset


def test_chunks_overlap_and_limits():
    text = 'Kalimat tentang UI. ' * 130
    chunks = split_text(text, chunk_size=240, chunk_overlap=30)
    assert len(chunks) >= 2
    assert all(len(x) <= 240 for x in chunks)
    with pytest.raises(ValueError): split_text('foo', 150, 150)


def test_domains_explicit_folder_or_mixed_pdf():
    assert page_domains(Path('dosen/test.pdf'), 'Perpustakaan UI') == ['dosen']
    assert 'event' in page_domains(Path('DATASET KA-I.pdf'), 'Hari wisuda UI')
    assert 'umum' in page_domains(Path('DATASET KA-I.pdf'), 'Hari wisuda UI')


def test_pdf_provenance_and_deterministic_ids():
    root = Path(__file__).resolve().parents[1] / 'data'
    pdf = root / 'umum' / 'umum_official_UI_2026.pdf'
    chunks1 = list(iter_pdf_chunks(pdf, root, chunk_size=650, chunk_overlap=80))
    chunks2 = list(iter_pdf_chunks(pdf, root, chunk_size=650, chunk_overlap=80))
    assert chunks1
    assert [c.id for c in chunks1] == [c.id for c in chunks2]
    assert all(c.metadata['page'] >= 1 and c.metadata['source'] == pdf.name for c in chunks1)
    assert all(c.metadata['chunk_id'] == c.id for c in chunks1)


def test_eval_metrics_label_gating():
    metrics = compute_metrics(['x','y','z'], ['y','z'], 3)
    assert metrics['recall_at_k'] == 1
    assert metrics['precision_at_k'] == pytest.approx(2/3)
    assert metrics['reciprocal_rank'] == .5
    summary = aggregate([{'verified': False, 'metrics':metrics}],k=3)
    assert summary['verified_scored_questions'] == 0
    assert 'mean_recall_at_k' not in summary
    with pytest.raises(ValueError): compute_metrics([], [], 3)


def test_template_explicitly_unverified():
    rows = load_dataset(Path(__file__).resolve().parents[1] / 'evaluation/qa_template.jsonl')
    assert rows and all(r['label_status'] == 'unverified' for r in rows)

class FakeCollection:
    def __init__(self):
        self.ids = {}
        self.deleted = []
    def get(self, where, include):
        return {'ids':[key for key, (_, meta) in self.ids.items() if meta['source_path'] == where['source_path']]}
    def delete(self, ids):
        self.deleted.extend(ids)
        for key in ids: self.ids.pop(key, None)
    def add_documents(self, docs, ids):
        for key, doc in zip(ids, docs): self.ids[key] = (doc.page_content, doc.metadata)


def test_incremental_index_keeps_existing_other_files():
    # Stub the LangChain Document dependency to exercise the DB logic offline.
    import sys
    from types import ModuleType
    from unittest.mock import patch
    from data import Chunk
    module = ModuleType('langchain_core')
    documents = ModuleType('langchain_core.documents')
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content, self.metadata = page_content, metadata
    documents.Document = Document
    fake = FakeCollection()
    fake.ids['other'] = ('Other source', {'source_path':'other.pdf'})
    a = Chunk('A', {'source_path':'x.pdf'}, 'a')
    b = Chunk('B', {'source_path':'x.pdf'}, 'b')
    with patch.dict(sys.modules, {'langchain_core':module,'langchain_core.documents':documents}):
        assert index_pdf(fake, [a,b], 'x.pdf') == (2,0)
        assert index_pdf(fake, [a,b], 'x.pdf') == (0,0)
        assert index_pdf(fake, [a], 'x.pdf') == (0,1)
    assert 'other' in fake.ids
    assert fake.deleted == ['b']


def test_answer_quality_requires_human_labels():
    from eval_answers import summarize
    empty = summarize([{'label_status':'unverified','answer_correct':True}])
    assert empty['answer_correct']['rate'] is None
    annotated = summarize([{'label_status':'human_verified','answer_correct':True, 'grounded':False}])
    assert annotated['answer_correct']['rate'] == 1.0
    assert annotated['grounded']['rate'] == 0.0
