"""Fill in advert text and application deadlines, for survivors only.

This runs AFTER the prefilter, and that ordering is the whole point. Fetching a
description for all ~38,000 postings would be 38,000 extra HTTP requests per
run — hours, and rude to the boards. Fetching for the ~570 that survive the
prefilter is a couple of minutes, and they are the only ones a model ever sees.

Two things come out of it:
  * a description for platforms whose list response has none (Workday,
    SmartRecruiters), which is a third of everything reaching the classifier
  * an application deadline, from the ATS where it is structured and from the
    advert text by rule where it is not
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.fetchers import detail, needs_detail
from src.models import Posting
from src.normalise import description_excerpt, extract_deadline
from src.observability import log, log_error


def enrich(postings: list[Posting], sources, workers: int = 8) -> dict:
    """Populate description and deadline in place. Returns counts for logging.

    Never raises: a board that refuses detail requests (UOB and Citi both 403)
    must cost that posting its description, not the run its results.
    """
    by_id = {s.id: s for s in sources}
    stats = {"fetched": 0, "failed": 0, "deadlines": 0, "skipped": 0}

    wanted = []
    for p in postings:
        source = by_id.get(p.source_id)
        if p.description or source is None or not needs_detail(source.platform):
            stats["skipped"] += 1
            continue
        wanted.append((p, source))

    if wanted:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, p, s): p for p, s in wanted}
            for future in as_completed(futures):
                posting = futures[future]
                try:
                    description, deadline = future.result()
                except Exception as e:
                    stats["failed"] += 1
                    log_error("enrich.failed", e, external_id=posting.external_id)
                    continue
                if description:
                    # Read the deadline from the WHOLE advert, then store only
                    # an excerpt. Closing dates live at the end of an advert,
                    # which is exactly what the excerpt discards — so the order
                    # here matters.
                    iso, sentence = extract_deadline(description)
                    if iso:
                        posting.deadline, posting.deadline_text = iso, sentence
                    # Detail endpoints return the full advert: 4,700-6,400
                    # characters, where the adapters that supply text in bulk
                    # cap at ~900. Six of those in one batch is ~34,000
                    # characters of prompt, which buries the other postings and
                    # blows a small free model's context. Same cap for both.
                    posting.description = description_excerpt(description)
                    stats["fetched"] += 1
                if deadline:
                    # Structured ATS date, and it wins: it is exact, where a
                    # date read out of prose is inferred from phrasing.
                    posting.deadline = deadline
                    posting.deadline_text = "(from the ATS closing-date field)"

    # Postings whose text arrived in the list response get the same pass. Note
    # that text is already excerpted, so a deadline stated in a closing
    # paragraph is not visible here — those platforms were measured and do not
    # publish closing dates anyway, so refetching them in full is not worth a
    # request per posting.
    for p in postings:
        if p.deadline or not p.description:
            continue
        iso, sentence = extract_deadline(p.description)
        if iso:
            p.deadline, p.deadline_text = iso, sentence
    stats["deadlines"] = sum(1 for p in postings if p.deadline)

    log("enrich.done", **stats)
    return stats


def _one(posting: Posting, source):
    return detail(source.platform, source.id, source.board, posting)
