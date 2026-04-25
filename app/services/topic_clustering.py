import asyncio
import logging
from dataclasses import dataclass, field

import hdbscan
import numpy as np
from google import genai

from app.core.config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
EMBEDDING_DIMENSIONS = 768


def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


async def embed_keywords(keywords: list[str]) -> np.ndarray:
    client = _get_gemini_client()
    all_embeddings = []
    loop = asyncio.get_event_loop()

    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i : i + BATCH_SIZE]
        response = await loop.run_in_executor(
            None,
            lambda b=batch: client.models.embed_content(
                model="gemini-embedding-2",
                contents=b,
                config={"output_dimensionality": EMBEDDING_DIMENSIONS},
            ),
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)

    return np.array(all_embeddings)


@dataclass
class TopicCluster:
    id: int
    label: str = ""
    keywords: list[str] = field(default_factory=list)
    keyword_metrics: list[dict] = field(default_factory=list)
    total_search_volume: int = 0
    avg_keyword_difficulty: float = 0.0
    avg_cpc: float = 0.0
    competitor_coverage: dict[str, float] = field(default_factory=dict)
    opportunity_score: float = 0.0


def cluster_keywords(
    keywords: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
) -> list[TopicCluster]:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)

    clusters: dict[int, TopicCluster] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = TopicCluster(id=label)
        clusters[label].keywords.append(keywords[idx])

    return list(clusters.values())


def score_clusters(
    clusters: list[TopicCluster],
    keyword_metrics: dict[str, dict],
    competitor_keywords: dict[str, set[str]],
) -> list[TopicCluster]:
    competitor_urls = list(competitor_keywords.keys())

    for cluster in clusters:
        volumes = []
        difficulties = []
        cpcs = []
        matched_metrics = []

        for kw in cluster.keywords:
            metrics = keyword_metrics.get(kw.lower())
            if not metrics:
                continue
            matched_metrics.append(metrics)
            if metrics.get("searchVolume"):
                volumes.append(metrics["searchVolume"])
            if metrics.get("keywordDifficulty"):
                difficulties.append(metrics["keywordDifficulty"])
            if metrics.get("cpc"):
                cpcs.append(metrics["cpc"])

        cluster.keyword_metrics = matched_metrics
        cluster.total_search_volume = sum(volumes) if volumes else 0
        cluster.avg_keyword_difficulty = float(np.mean(difficulties)) if difficulties else 0.0
        cluster.avg_cpc = float(np.mean(cpcs)) if cpcs else 0.0

        for url in competitor_urls:
            comp_kws = competitor_keywords[url]
            covered = sum(1 for kw in cluster.keywords if kw.lower() in comp_kws)
            cluster.competitor_coverage[url] = covered / len(cluster.keywords) if cluster.keywords else 0.0

        avg_coverage = np.mean(list(cluster.competitor_coverage.values())) if cluster.competitor_coverage else 0.0
        gap_factor = 1.0 - avg_coverage
        difficulty_factor = 1.0 - (cluster.avg_keyword_difficulty / 100.0) if cluster.avg_keyword_difficulty else 0.5

        cluster.opportunity_score = (
            cluster.total_search_volume * gap_factor * difficulty_factor
        )

    clusters.sort(key=lambda c: c.opportunity_score, reverse=True)
    return clusters


async def build_topic_clusters(
    keywords: list[str],
    keyword_metrics: dict[str, dict],
    competitor_keywords: dict[str, set[str]],
    min_cluster_size: int = 3,
) -> list[TopicCluster]:
    if len(keywords) < min_cluster_size:
        logger.warning("Too few keywords (%d) to cluster", len(keywords))
        return []

    logger.info("Embedding %d keywords with Gemini", len(keywords))
    embeddings = await embed_keywords(keywords)

    logger.info("Clustering keywords with HDBSCAN")
    clusters = cluster_keywords(keywords, embeddings, min_cluster_size)
    logger.info("Found %d topic clusters", len(clusters))

    clusters = score_clusters(clusters, keyword_metrics, competitor_keywords)
    return clusters
