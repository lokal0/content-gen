import asyncio
import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_queues: dict[str, asyncio.Queue] = {}


def get_queue(job_id: uuid.UUID) -> asyncio.Queue:
    key = str(job_id)
    if key not in _queues:
        _queues[key] = asyncio.Queue()
    return _queues[key]


def cleanup_queue(job_id: uuid.UUID):
    key = str(job_id)
    _queues.pop(key, None)


async def emit(job_id: uuid.UUID, event_type: str, data: dict[str, Any]):
    queue = get_queue(job_id)
    await queue.put({"event": event_type, "data": data})


async def emit_stage(job_id: uuid.UUID, stage: str, detail: str | None = None):
    await emit(job_id, "stage", {"stage": stage, "detail": detail})


async def emit_tool_call(job_id: uuid.UUID, name: str, input_data: dict, output: str | None = None):
    import json as _json

    # Try to send parsed output so frontend can render structured data
    parsed_output = None
    if output:
        try:
            parsed = _json.loads(output)
            if isinstance(parsed, list):
                parsed_output = parsed[:5]
            elif isinstance(parsed, dict):
                parsed_output = parsed
        except (ValueError, TypeError):
            pass

    await emit(job_id, "tool_call", {
        "name": name,
        "input": input_data,
        "output_preview": (output or "")[:500],
        "output_parsed": parsed_output,
    })


async def emit_thinking(job_id: uuid.UUID, text: str):
    if text:
        await emit(job_id, "thinking", {"text": text[:1000]})


async def emit_text_chunk(job_id: uuid.UUID, chunk: str):
    if chunk:
        await emit(job_id, "text", {"chunk": chunk})


async def emit_article(job_id: uuid.UUID, article: dict):
    await emit(job_id, "article", article)


async def emit_complete(job_id: uuid.UUID, data: dict):
    await emit(job_id, "complete", data)
    cleanup_queue(job_id)


async def emit_error(job_id: uuid.UUID, message: str):
    await emit(job_id, "error", {"message": message})
    cleanup_queue(job_id)
