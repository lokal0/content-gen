import re
from dataclasses import dataclass

from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.crawler import CrawlResult


@dataclass
class ExtractedKeyword:
    keyword: str
    score: float
    method: str


def _merge_texts(crawl_result: CrawlResult) -> str:
    parts = []
    for page in crawl_result.pages:
        if page.full_text:
            parts.append(page.full_text)
        if page.metadata and isinstance(page.metadata, dict):
            for key in ("description", "og:description", "twitter:description", "keywords"):
                if key in page.metadata:
                    parts.append(str(page.metadata[key]))
        if page.headings:
            for level_headings in page.headings.values():
                parts.extend(level_headings)
    return " ".join(parts)


def _is_valid_keyword(phrase: str) -> bool:
    if len(phrase) < 3 or len(phrase) > 80:
        return False
    if re.search(r"https?://|www\.|\.com|\.org|\.net", phrase):
        return False
    if re.search(r"[{}()\[\]<>=/\\|]", phrase):
        return False
    alpha_chars = sum(1 for c in phrase if c.isalpha())
    if alpha_chars < len(phrase) * 0.5:
        return False
    if re.search(r"\.\.\.", phrase):
        return False
    return True


def extract_tfidf_keywords(crawl_results: list[CrawlResult], top_n: int = 30) -> dict[str, list[ExtractedKeyword]]:
    docs = []
    urls = []
    for cr in crawl_results:
        text = _merge_texts(cr)
        if text.strip():
            docs.append(text)
            urls.append(cr.competitor_url)

    if not docs:
        return {}

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1,
        max_df=1.0 if len(docs) <= 2 else 0.9,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+\-]{1,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    results = {}
    for idx, url in enumerate(urls):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[-top_n * 2 :][::-1]
        keywords = []
        for i in top_indices:
            if row[i] > 0 and _is_valid_keyword(feature_names[i]):
                keywords.append(ExtractedKeyword(
                    keyword=feature_names[i],
                    score=float(row[i]),
                    method="tfidf",
                ))
            if len(keywords) >= top_n:
                break
        results[url] = keywords

    return results


def extract_rake_keywords(crawl_result: CrawlResult, top_n: int = 30) -> list[ExtractedKeyword]:
    text = _merge_texts(crawl_result)
    if not text.strip():
        return []

    rake = Rake(max_length=4, min_length=2)
    rake.extract_keywords_from_text(text)
    ranked = rake.get_ranked_phrases_with_scores()

    seen = set()
    keywords = []
    for score, phrase in ranked:
        phrase_clean = phrase.strip().lower()
        if phrase_clean in seen:
            continue
        if not _is_valid_keyword(phrase_clean):
            continue
        seen.add(phrase_clean)
        keywords.append(ExtractedKeyword(
            keyword=phrase_clean,
            score=float(score),
            method="rake",
        ))
        if len(keywords) >= top_n:
            break
    return keywords


def extract_all_keywords(crawl_results: list[CrawlResult]) -> dict[str, list[ExtractedKeyword]]:
    tfidf_results = extract_tfidf_keywords(crawl_results)

    all_keywords = {}
    for cr in crawl_results:
        url = cr.competitor_url
        tfidf_kws = tfidf_results.get(url, [])
        rake_kws = extract_rake_keywords(cr)
        all_keywords[url] = tfidf_kws + rake_kws

    return all_keywords
