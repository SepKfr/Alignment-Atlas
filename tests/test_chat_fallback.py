from __future__ import annotations

import unittest
from unittest.mock import patch

from src.app.chat_agent import _should_use_external_fallback
from src.app.services import render_answer_markdown
from src.retrieval.external_fallback import retrieve_external_evidence


class ChatFallbackTests(unittest.TestCase):
    def test_low_quality_triggers_fallback(self) -> None:
        evidence = {
            "chunks": [{"score": 0.12}, {"score": 0.1}],
            "claims": [],
        }
        use_fallback, reason, stats = _should_use_external_fallback(evidence)
        self.assertTrue(use_fallback)
        self.assertTrue(reason)
        self.assertLess(stats["top_score"], 0.2)

    def test_strong_quality_skips_fallback(self) -> None:
        evidence = {
            "chunks": [{"score": 0.82}, {"score": 0.7}, {"score": 0.65}],
            "claims": [{"claim_id": "c1"}, {"claim_id": "c2"}, {"claim_id": "c3"}],
        }
        use_fallback, _, _ = _should_use_external_fallback(evidence)
        self.assertFalse(use_fallback)

    @patch("src.retrieval.external_fallback._fetch_openalex")
    @patch("src.retrieval.external_fallback._fetch_arxiv")
    @patch("src.retrieval.external_fallback._fetch_wikipedia")
    def test_scholarly_first_then_web(self, wiki, arxiv, openalex) -> None:
        openalex.return_value = []
        arxiv.return_value = []
        wiki.return_value = [
            {
                "kind": "external",
                "id": "ext_web_1",
                "section": "external_web",
                "title": "Web Result",
                "snippet": "Web snippet.",
                "url": "https://example.com",
                "source_type": "external_web",
                "provider": "wikipedia",
                "score": 0.3,
            }
        ]
        out = retrieve_external_evidence("reward hacking", min_scholarly_before_web=1)
        self.assertTrue(out["used_general_web"])
        self.assertEqual(out["results"][0]["source_type"], "external_web")

    def test_render_marks_provenance_tags(self) -> None:
        payload = {
            "evidence_status": "Atlas + external fallback",
            "fallback_reason": "Atlas retrieval scores are low for this query.",
            "answer": {
                "title": "Reward Hacking",
                "summary": "Short summary.",
                "key_points": [
                    {
                        "point": "Agents can exploit misspecified objectives.",
                        "citations": [{"kind": "external", "id": "ext_s1"}],
                    }
                ],
                "debates_and_contradictions": [],
                "limitations": [],
                "citations": [
                    {
                        "kind": "external",
                        "id": "ext_s1",
                        "doc_id": "",
                        "section": "external_scholarly",
                        "snippet": "Evidence snippet.",
                        "source_type": "external_scholarly",
                        "url": "https://openalex.org/W1",
                        "title": "Example Paper",
                    }
                ],
            },
        }
        rendered = render_answer_markdown(payload, resolver=None)
        self.assertIn("Evidence status", rendered)
        self.assertIn("[External Scholar]", rendered)
        self.assertIn("Example Paper", rendered)


if __name__ == "__main__":
    unittest.main()
