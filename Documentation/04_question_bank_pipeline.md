# Question Bank Pipeline

Covers `app/retrieval/page_window.py`, `question_detector.py`, `answer_extractor.py`,
`validator.py`, `escalation.py`, `question_bank_export.py` — the exhaustive
whole-document pipeline: `detect -> answer -> validate -> escalate -> Excel/CSV`.

## Why this exists as a separate design from `rag_query.py`

This project's actual goal is whole-document extraction — every question in the
document, not one answer to one ad-hoc query. An earlier attempt at this exact goal
(a completely separate repo, before this one) "failed drastically" and was
abandoned; this pipeline is a from-scratch rebuild, built and confirmed in
incremental, individually-tested stages rather than as one large change.

**No vector search on the primary path.** `page_window.py` builds P-1/P/P+1
windows directly from the already-known page sequence — deterministic, no Qdrant
involved. Semantic search is for when you don't know where to look; page order is
already known here. Vector search only re-enters via `escalation.py`, and only for
items the deterministic validator actually flags.

## Question detection — page P is the only source of NEW questions

`question_detector.py`'s `detect_questions_on_page_with_context()` sees the target
page's own Markdown, plus a small **boundary peek** — the last ~3 `\n\n`-blocks of
page P-1 and the first ~3 of page P+1, explicitly marked `[CONTEXT ONLY]` in the
prompt. This is a deliberate middle ground between two rejected extremes:

- **Page-alone, no peek** — duplicate-proof by construction, but a question whose
  stem starts on one page and continues onto the next gets truncated or missed.
- **Full P-1/P/P+1 for detection** (an earlier design, tried and reverted) —
  captures spanning questions fully, but reopens the exact same-question-detected-
  twice problem that design was reverted over, and needs real fuzzy-dedup logic
  (near-duplicate LLM text isn't byte-identical) to fix, which was never built.

The peek's rule: never extract a new question whose stem lives *entirely* inside a
`[CONTEXT ONLY]` section — only use it to complete something already anchored on
page P. Symmetric by construction: when P-1 is later processed as its own center
page, *its* next-page-head peek is P's own head, so a question anchored on P-1 that
spills onto P still gets captured in full — just attributed to P-1, never
duplicated on P.

**Known open issue, confirmed reproducing even on a single, simple reference page
(`task_page-0016`), not just complex multi-question-set pages:** a two-tier list
question (a roman-numeral reference list followed by lettered combination choices)
sometimes gets split by gpt-4o-mini into a duplicate parent+subpart pair — e.g. one
run produced both `[2]` (options = the roman-numeral list) and `[2(a)]` (options =
the lettered list) for what is really one question. This is run-to-run — it
doesn't always reproduce, since gpt-4o-mini isn't perfectly deterministic (no
`temperature=0` currently set). A concrete fix was drafted (never emit a
parent-only entry alongside its own subparts; distinguish "a nested reference list
inside one question" from "genuine independently-answerable subparts") but the
edit was paused mid-flight and **is not currently applied**. Worth finishing before
treating detection as fully reliable at scale.

## Answer extraction — verbatim only, same rigor as `rag_query.py`

`answer_extractor.py`'s `extract_answers_for_questions()` builds each detected
question's own P-1/P/P+1 window (`page_window.py`) and asks whether it contains a
direct answer. Reuses the exact "related fact ≠ direct answer" fix already
validated in `rag_query.py` — confirmed against the same failure shape: "what is
the height" is not answered by "height is 0.15 m more than width" with no width
given.

**Confirmed bug, root-caused, currently unfixed:** on `task_page-0020`, a page
reusing the same `(A)/(B)/(C)` labels for two separate, unrelated question blocks,
the model attached an answer from one question set's leftover text to a completely
different question in the other set — both got `exact, confidence=1.00`, and one
of them (asking for "two characteristics") got an answer describing only one. The
prompt's current verbatim/related-fact rules don't cover **duplicate/reused labels
across two distinct question blocks on one page** — a proposed fix (explicit label-
reuse warning + a completeness check: "if asked for N things, an answer providing
fewer than N isn't a complete direct answer") was drafted but not yet applied.
Separately, this specific case also needed the real neighboring page's content to
be available at all — testing single pages in isolation (rather than as part of the
full ingested document) means there's no real P+1 to search, regardless of prompt
quality.

`answer_status` follows the same `exact`/`multi_source`/`inferred`/`not_found` rule
used everywhere else in this project.

## Validation — deterministic, no LLM call

`validator.py` — plain rule-checks producing `good`/`uncertain` + specific reasons:

- Empty/short `question_text`
- Low detection or answer confidence (`< 0.7` by default)
- `answer_status=inferred` — always flagged, per the project-wide extraction rule
- `answer_status=not_found` — not treated as "wrong," flagged as "worth a broader
  search," since the real answer may sit further away than P-1/P/P+1 could see
- `exact`/`multi_source` with an empty `answer_text` (an inconsistent state)
- A cited `source_page` outside what the model was actually shown

## Escalation — targeted RAG, same model

`escalation.py` retries `uncertain` items only, via a real vector search
(`context_builder.build_context`, against that document's own Qdrant collection)
instead of the fixed P-1/P/P+1 window — in case the real answer sits further away
in the document. **Model stays gpt-4o-mini throughout, never escalated to gpt-4o** —
an explicit correction during design: the fix is a better context-finding strategy,
not a smarter model. Only replaces the original result if the retry is strictly
better (fewer flagged reasons, or higher confidence at equal reasons) — never
silently downgrades.

## Export — CSV and real `.xlsx`

`question_bank_export.py` has both `write_question_bank_csv()` and
`write_question_bank_xlsx()` (via `openpyxl` — column widths set, header row
frozen). `scripts/run_full_pipeline.py` writes `.xlsx`; the standalone
`scripts/run_question_bank.py` writes `.csv`.

## Running it

Whole pipeline, one document at a time:

```bash
python scripts/run_question_bank.py --markdown-json output/my_doc.markdown.json
```

Or as part of the full one-command pipeline (`run_pipeline.py` through export):

```bash
python scripts/run_full_pipeline.py --input <file-or-folder>
```

Individual stages, for debugging just one piece:

```bash
python scripts/test_question_detector.py --markdown-json output/my_doc.markdown.json [--page N]
python scripts/test_answer_extractor.py --markdown-json output/my_doc.markdown.json [--page N]
```

## Verified so far

- Edge-case windowing confirmed correct on a real 5-page document (first/last page
  boundaries).
- Full pipeline confirmed end-to-end on `task_page-0016` (single page): 4 questions
  detected, all answered `exact` with correct verbatim text, all validated `good`,
  escalation correctly no-op'd.
- Full pipeline confirmed end-to-end on `task_page-0020` (single page, in
  isolation): surfaced the label-reuse answer-misattribution bug above — a genuine
  finding from real testing, not yet fixed.
- **Not yet run** against the full 30-page `sample_data` folder as one ingested
  document — this is the real test of the boundary-peek design and the P-1/P/P+1
  answer lookup, since isolated single-page tests structurally can't exercise
  cross-page behavior at all (confirmed: `task_page-0020` tested alone has no real
  P+1 to search, which was itself part of what caused its answer-extraction bug).
