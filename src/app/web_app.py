from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.app.services import AtlasService, render_answer_markdown

SERVICE = AtlasService()
DEFAULT_TOP_K_CHUNKS = 12
DEFAULT_NEIGHBOR_RADIUS = 2
DEFAULT_INCLUDE_SUGGESTIONS = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    steer: float = 0.0
    steering_mode: str = "safety_first"
    top_k_chunks: int = 12
    neighbor_radius: int = 2
    include_suggestions: bool = True


class IngestRequest(BaseModel):
    title: str = ""
    source_url: str = Field(..., min_length=3)
    source_type: str = "auto"
    year: Optional[int] = None
    run_relations: bool = True
    incremental: bool = True


def _chat(
    user_message: str,
    history: List[Dict[str, str]],
    steering_mode: str,
) -> tuple[List[Dict[str, str]], str]:
    if not user_message.strip():
        return history, ""
    try:
        payload = SERVICE.chat(
            user_message=user_message,
            steer=0.0,
            steering_mode=steering_mode,
            top_k_chunks=DEFAULT_TOP_K_CHUNKS,
            neighbor_radius=DEFAULT_NEIGHBOR_RADIUS,
            include_suggestions=DEFAULT_INCLUDE_SUGGESTIONS,
        )
        answer_md = render_answer_markdown(payload, resolver=SERVICE.resolver)
    except Exception as e:
        answer_md = f"Error: {e}"
    history = (history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer_md},
    ]
    return history, ""


def _reset_chat() -> List[Dict[str, str]]:
    SERVICE.reset_chat()
    return []


def _ingest(
    title: str,
    source_url: str,
    source_type: str,
    year: Optional[float],
    run_relations: bool,
    incremental: bool,
) -> tuple[str, str, str]:
    yr = int(year) if year is not None else None
    result = SERVICE.ingest_source(
        title=(title or "").strip(),
        source_url=source_url,
        source_type=source_type,
        year=yr,
        run_relations=run_relations,
        incremental=incremental,
    )
    stage_logs = []
    for st in result.get("stage_results", []) or []:
        stage_logs.append(
            f"### {st.get('module')}\n"
            f"- ok: `{st.get('ok')}`\n"
            f"- return_code: `{st.get('return_code')}`\n"
            f"- elapsed_seconds: `{st.get('elapsed_seconds')}`\n"
            f"```\n{st.get('output_tail', '')}\n```"
        )
    logs_md = "\n\n".join(stage_logs) if stage_logs else "No stage logs."
    status = "Ingest succeeded." if result.get("ok") else f"Ingest failed: {result.get('error')}"
    return status, json.dumps(result, indent=2, ensure_ascii=False), logs_md


def _apply_ingest_preset(preset: str) -> tuple[bool, bool]:
    p = (preset or "").strip().lower()
    if p == "full_quality":
        return True, True
    # fast_demo default
    return False, True


def _prefill_prompt_from_goal(goal: str) -> str:
    g = (goal or "").strip().lower()
    templates = {
        "foundations": "What are the core alignment and AI safety problems researchers focus on today?",
        "reward_hacking": "What is reward hacking, and what evidence shows it in modern systems?",
        "oversight_eval": "How do oversight and evaluations detect misalignment before deployment?",
        "interpretability": "Which interpretability methods are most useful for alignment assurance, and why?",
        "deployment": "What practical deployment controls reduce alignment risk in production?",
    }
    return templates.get(g, "")


def build_gradio_ui() -> gr.Blocks:
    with gr.Blocks(title="Alignment Atlas") as demo:
        gr.Markdown("# Alignment Atlas")

        with gr.Tab("Chat"):
            gr.Markdown(
                "Alignment Atlas is a research assistant for AI alignment and safety literature.\n\n"
                "Use it to explore questions across reward hacking, interpretability, oversight, and deployment risk.\n\n"
                "How it works:\n"
                "1) Pick what you want to learn.\n"
                "2) Ask your question in plain language.\n"
                "3) Get evidence-grounded answers from the Atlas corpus (with labeled external sources only when needed)."
            )
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                learning_goal = gr.Dropdown(
                    label="What are you here to learn?",
                    choices=[
                        ("Core alignment fundamentals", "foundations"),
                        ("Reward hacking and specification gaming", "reward_hacking"),
                        ("Oversight and evaluation methods", "oversight_eval"),
                        ("Interpretability for alignment", "interpretability"),
                        ("Safe deployment and operations", "deployment"),
                    ],
                    value="foundations",
                )
            with gr.Row():
                msg = gr.Textbox(label="Message", placeholder="Ask about alignment papers...")
                use_goal_prompt = gr.Button("Use Topic Prompt")
            gr.Examples(
                examples=[
                    ["What are the key disagreement areas in current AI alignment research?"],
                    ["How do papers define and measure deceptive alignment risk?"],
                    ["What are the strongest evidence-backed mitigations for reward hacking?"],
                    ["Where is interpretability helpful, and where does it still fall short?"],
                ],
                inputs=[msg],
                label="Example questions",
            )
            with gr.Row():
                steering_mode = gr.Dropdown(
                    label="How should answers be framed?",
                    choices=[
                        ("Risk and safety implications", "safety_first"),
                        ("Mechanisms and interpretability", "interpretability_first"),
                        ("Practical deployment guidance", "practical_deployment"),
                    ],
                    value="safety_first",
                )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")

            send.click(
                fn=_chat,
                inputs=[msg, chatbot, steering_mode],
                outputs=[chatbot, msg],
            )
            use_goal_prompt.click(
                fn=_prefill_prompt_from_goal,
                inputs=[learning_goal],
                outputs=[msg],
            )
            msg.submit(
                fn=_chat,
                inputs=[msg, chatbot, steering_mode],
                outputs=[chatbot, msg],
            )
            clear.click(fn=_reset_chat, inputs=[], outputs=[chatbot])

        with gr.Tab("Ingest"):
            preset = gr.Dropdown(
                label="Ingest Preset",
                choices=["fast_demo", "full_quality"],
                value="fast_demo",
                info="fast_demo = cheaper/faster. full_quality = runs contradiction/entailment stages too.",
            )
            with gr.Row():
                title = gr.Textbox(label="Paper Title", placeholder="e.g. Concrete Problems in AI Safety")
                source_url = gr.Textbox(label="Paper/Page URL", placeholder="https://...")
            with gr.Row():
                source_type = gr.Dropdown(label="Source Type", choices=["auto", "pdf", "html"], value="auto")
                year = gr.Number(label="Year (optional)", precision=0, value=None)
                run_relations = gr.Checkbox(
                    label="Run contradiction/entailment relation stages (slower, OpenAI-heavy)",
                    value=False,
                )
                incremental = gr.Checkbox(
                    label="Incremental ingest (recommended for Spaces; faster and cheaper for new papers)",
                    value=True,
                )
            preset.change(
                fn=_apply_ingest_preset,
                inputs=[preset],
                outputs=[run_relations, incremental],
            )
            ingest_btn = gr.Button("Ingest Into Graph", variant="primary")
            status = gr.Textbox(label="Status")
            ingest_json = gr.Code(label="Ingest Result JSON", language="json")
            stage_logs = gr.Markdown("Stage logs will appear here.")

            ingest_btn.click(
                fn=_ingest,
                inputs=[title, source_url, source_type, year, run_relations, incremental],
                outputs=[status, ingest_json, stage_logs],
            )
    return demo


def create_api_app() -> FastAPI:
    api = FastAPI(title="Alignment Atlas API", version="0.1.0")

    @api.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, **SERVICE.runtime_info()}

    @api.get("/api/runtime")
    def runtime() -> Dict[str, Any]:
        return SERVICE.runtime_info()

    @api.post("/api/chat")
    async def chat_endpoint(body: ChatRequest) -> Dict[str, Any]:
        return await asyncio.to_thread(
            SERVICE.chat,
            user_message=body.message,
            steer=body.steer,
            steering_mode=body.steering_mode,
            top_k_chunks=body.top_k_chunks,
            neighbor_radius=body.neighbor_radius,
            include_suggestions=body.include_suggestions,
        )

    @api.post("/api/ingest")
    async def ingest_start_endpoint(body: IngestRequest) -> Dict[str, Any]:
        return await asyncio.to_thread(
            SERVICE.start_ingest_job,
            title=body.title,
            source_url=body.source_url,
            source_type=body.source_type,
            year=body.year,
            run_relations=body.run_relations,
            incremental=body.incremental,
        )

    @api.post("/api/ingest/sync")
    async def ingest_sync_endpoint(body: IngestRequest) -> Dict[str, Any]:
        return await asyncio.to_thread(
            SERVICE.ingest_source,
            title=body.title,
            source_url=body.source_url,
            source_type=body.source_type,
            year=body.year,
            run_relations=body.run_relations,
            incremental=body.incremental,
        )

    @api.get("/api/ingest/jobs")
    def list_ingest_jobs(limit: int = 50) -> Dict[str, Any]:
        return {"jobs": SERVICE.list_ingest_jobs(limit=limit)}

    @api.get("/api/ingest/jobs/{job_id}")
    def get_ingest_job(job_id: str) -> Dict[str, Any]:
        job = SERVICE.get_ingest_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Unknown job_id={job_id}")
        return job

    return api


demo = build_gradio_ui()
api_app = create_api_app()
app = gr.mount_gradio_app(api_app, demo, path="/")

