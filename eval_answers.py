"""Report explicitly human-annotated answer quality; never fabricate labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

FIELDS = ('answer_correct', 'grounded', 'refusal_appropriate')


def summarize(rows: list[dict]) -> dict:
    verified = [row for row in rows if row.get('label_status') == 'human_verified']
    report = {'total_rows':len(rows),'human_verified_rows':len(verified)}
    for field in FIELDS:
        labelled = [row[field] for row in verified if isinstance(row.get(field), bool)]
        report[field] = {'evaluated':len(labelled), 'rate':sum(labelled)/len(labelled) if labelled else None}
    return report


def main():
    parser = argparse.ArgumentParser(description='Summarize manually annotated RAG answer quality')
    parser.add_argument('--annotations',type=Path,default=Path('evaluation/answer_annotations_template.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('evaluation/answer_results.json'))
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.annotations.read_text(encoding='utf-8').splitlines() if line.strip()]
    report=summarize(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
