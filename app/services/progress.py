import logging
import uuid
from datetime import datetime, timezone

from app.core.database import async_session
from app.models.tables import Submission

logger = logging.getLogger(__name__)

STAGES = [
    "discovering_competitors",
    "crawling",
    "gathering_seo_data",
    "extracting_keywords",
    "enriching_keywords",
    "classifying_intent",
    "embedding_keywords",
    "clustering",
    "agent_researching",
    "agent_writing",
    "completed",
]


async def update_progress(
    job_id: uuid.UUID,
    stage: str,
    detail: str | None = None,
) -> None:
    async with async_session() as db:
        submission = await db.get(Submission, job_id)
        if not submission:
            return

        progress = submission.progress or {"stages": [], "current_stage": None}
        stages_done = progress.get("stages", [])
        now = datetime.now(timezone.utc).isoformat()

        if stage not in [s["name"] for s in stages_done]:
            stages_done.append({
                "name": stage,
                "started_at": now,
                "detail": detail,
            })

        progress["stages"] = stages_done
        progress["current_stage"] = stage
        progress["current_detail"] = detail
        progress["stage_index"] = STAGES.index(stage) if stage in STAGES else -1
        progress["total_stages"] = len(STAGES)

        submission.progress = progress
        await db.commit()
        logger.info("Job %s: %s%s", job_id, stage, f" — {detail}" if detail else "")


async def update_agent_progress(
    job_id: uuid.UUID,
    iteration: int,
    tool_name: str | None = None,
    tool_input_preview: str | None = None,
) -> None:
    detail = f"iteration {iteration}"
    if tool_name:
        detail = f"iteration {iteration} — calling {tool_name}"
        if tool_input_preview:
            detail += f"({tool_input_preview[:60]})"

    stage = "agent_researching" if tool_name else "agent_writing"
    await update_progress(job_id, stage, detail)
