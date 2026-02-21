from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.ingest.stages import StageResult, build_stage

ProgressCallback = Callable[[Dict[str, object]], None]

FULL_STAGES: List[str] = [
    "dedupe_manifest",
    "collect_manifest",
    "download_sources",
    "pdf_to_text",
    "html_to_text",
    "section_chunk",
    "apply_neighbors",
    "embed_chunks",
    "export_chunk_embs",
    "extract_claims",
    "build_kg",
]

RELATION_STAGES: List[str] = [
    "detect_contradictions",
    "merge_relations_into_kg",
]


@dataclass
class PipelineRun:
    ok: bool
    stage_results: List[Dict[str, object]]
    failed_module: str = ""


class IngestPipeline:
    """
    OOP pipeline orchestrator that runs stage classes in sequence.
    """

    def __init__(self, stages: Optional[Sequence[str]] = None):
        self._base_stages = list(stages) if stages is not None else list(FULL_STAGES)

    def resolve_stages(self, *, run_relations: bool = True) -> List[str]:
        stages = list(self._base_stages)
        if run_relations:
            stages.extend(RELATION_STAGES)
        return stages

    def run_full(
        self,
        *,
        run_relations: bool,
        progress_callback: Optional[ProgressCallback] = None,
        claims_env: Optional[Dict[str, str]] = None,
    ) -> PipelineRun:
        stages = self.resolve_stages(run_relations=run_relations)
        stage_results: List[Dict[str, object]] = []
        total = len(stages)
        for idx, module_name in enumerate(stages, 1):
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_stage": module_name,
                        "stage_index": idx,
                        "stage_total": total,
                        "stage_results": stage_results,
                    }
                )
            stage = build_stage(module_name)
            env_overrides = claims_env if module_name in {"src.ingest.04_extract_claims_openai", "extract_claims"} else None
            res: StageResult = stage.run(
                progress_callback=progress_callback,
                stage_index=idx,
                stage_total=total,
                stage_results=stage_results,
                env_overrides=env_overrides,
            )
            rec = {
                "module": res.module,
                "ok": res.ok,
                "return_code": res.return_code,
                "elapsed_seconds": res.elapsed_seconds,
                "output_tail": res.output_tail,
            }
            stage_results.append(rec)
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_stage": module_name,
                        "stage_index": idx,
                        "stage_total": total,
                        "stage_results": stage_results,
                    }
                )
            if not res.ok:
                return PipelineRun(ok=False, stage_results=stage_results, failed_module=module_name)
        return PipelineRun(ok=True, stage_results=stage_results)

    def run_single_stage(
        self,
        module_name: str,
        *,
        progress_callback: Optional[ProgressCallback] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Dict[str, object]]:
        stage = build_stage(module_name)
        res: StageResult = stage.run(progress_callback=progress_callback, env_overrides=env_overrides)
        rec = {
            "module": res.module,
            "ok": res.ok,
            "return_code": res.return_code,
            "elapsed_seconds": res.elapsed_seconds,
            "output_tail": res.output_tail,
        }
        return res.ok, rec

