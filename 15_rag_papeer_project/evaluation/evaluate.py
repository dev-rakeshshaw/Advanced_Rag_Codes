import os
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseEmbeddingModel, OpenRouterModel

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, embeddings

load_dotenv()

# OpenRouter LLM for DeepEval
judge_model = OpenRouterModel(
    model=os.environ["OPENROUTER_CHAT_MODEL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
    temperature=0,
)
EVALUATION_DIR = Path(__file__).resolve().parent
PDF_PATH            = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE        = EVALUATION_DIR / "goldens.json"
MAX_CONTEXTS        = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD    = 0.7
RESULTS_FILE = EVALUATION_DIR / "eval_results.json"

# Reuse the existing cached local Hugging Face embeddings
class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):

    def load_model(self):
        return embeddings

    def embed_text(self, text: str) -> list[float]:
        return self.load_model().embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.load_model().embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return "Local HuggingFace MiniLM"

def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer(model=judge_model)
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[PDF_PATH],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=MAX_CONTEXTS,
            embedder=LocalEmbeddingModel(),
            critic_model=judge_model,
            context_quality_threshold=0.0,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def main() -> None:
    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()

    docs = load_document(PDF_PATH)
    graph = build_graph(db_path="eval_checkpoints.db")

    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge_model),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge_model),
    ]

    test_cases = []
    for pair in pairs:
        session_id = f"evaluation_session_{uuid4()}"
        add_paper(docs, session_id)

        query = pair["input"] + " as per the report in knowledge base"
        answer, retrieval_context = run_rag_query(graph, query, session_id)
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(max_concurrent=3, throttle_value=5),
    )

    summary = []
    for test_result in results.test_results:
        summary.append({
            "input": test_result.input,
            "actual_output": test_result.actual_output,
            "success": test_result.success,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.success,
                    "reason": m.reason,
                }
                for m in test_result.metrics_data
            ],
        })

    RESULTS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False),encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}.")


if __name__ == "__main__":
    main()
