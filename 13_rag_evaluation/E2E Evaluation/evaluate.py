import os
import json
from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import ( answer_relevancy, context_precision, faithfulness, context_recall, NoiseSensitivity )

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"

QA_PAIRS = [
    ("What does sustainable development mean?", "Development that meets today's needs without preventing future generations from meeting their own needs"),
    # ("What are the three pillars of sustainable development?", "Economic growth, social inclusion, and environmental protection"),
    # ("How many global goals did countries agree on in the 2030 Agenda?", "17 Sustainable Development Goals"),
    # ("What is the Paris Agreement's temperature target?", "Limit global warming to 1.5 degrees Celsius above pre-industrial levels"),
    # ("What does a circular economy try to reduce?", "Waste, by keeping products and materials in use for as long as possible"),
    # ("What are two examples of circular economy actions?", "Recycling and repairing products instead of throwing them away"),
    # ("What basic needs are highlighted under social equity?", "Access to clean water, basic education, and healthcare for all people"),
    # ("What role do individuals play in sustainable development?", "Individuals can reduce waste, save energy, and make eco-friendly choices"),
    # ("Why is limiting global warming to 1.5 degrees Celsius important?", "Beyond 1.5 degrees Celsius, risks of floods, droughts, and extreme weather increase significantly"),
    ("What is the target year for the Sustainable Development Goals?", "2030"),
]


def run_evaluation(chain, retriever):
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
    )

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    )

    noise_sensitivity = NoiseSensitivity(llm=evaluator_llm)

    dataset = []

    for question, reference in QA_PAIRS:
        contexts = [
            doc.page_content for doc in retriever.invoke(question)
        ]
        context = "\n\n".join(contexts)
        response = chain.invoke(
            {"context": context, "question": question}
        ).content

        dataset.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": reference,
        })

    results = evaluate(
        dataset=EvaluationDataset.from_list(dataset),
        llm=evaluator_llm,
        embeddings=embeddings,
        metrics=[
            answer_relevancy,
            context_precision,
            faithfulness,
            context_recall,
            noise_sensitivity,
        ],
        run_config=RunConfig(
            timeout=300,
            max_retries=2,
            max_wait=60,
            max_workers=2,
        ),
    )

    df = results.to_pandas()
    df.to_json("evaluation_results.json", orient="records", indent=2, force_ascii=False)

    columns = list(df.columns)
    markdown = [
        "# RAGAS Evaluation Results",
        "",
        f"Total evaluated questions: {len(df)}",
        "",
        "| " + " | ".join(str(c).replace("|", "\\|") for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if value is None:
                cell = ""
            elif isinstance(value, (list, dict)):
                cell = json.dumps(value, ensure_ascii=False)
            else:
                cell = str(value)
            cell = cell.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")
            cells.append(cell)
        markdown.append("| " + " | ".join(cells) + " |")

    with open("evaluation_results.md", "w", encoding="utf-8") as file:
        file.write("\n".join(markdown) + "\n")

    print("Evaluation saved to evaluation_results.json and evaluation_results.md")