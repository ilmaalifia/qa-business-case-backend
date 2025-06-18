import asyncio
import json
import os
from collections import defaultdict

import numpy as np
from app.graph import graph
from app.utils import setup_logger
from dotenv import load_dotenv
from langchain_community.utils.math import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

load_dotenv()
logger = setup_logger(__name__)

SLEEP_DURATION = 60  # seconds
llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )
)
model_embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("DENSE_MODEL"),
)


def avg_semantic_similarity(texts: list[str]) -> float:
    n = len(texts)
    if n <= 1:
        return 1.0

    embeddings = model_embeddings.embed_documents(texts)
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_scores = similarity_matrix[upper_triangle_indices]
    avg_similarity = np.mean(pairwise_scores)
    return avg_similarity


async def main():
    final_result = defaultdict(dict)
    path = os.getenv("EVALUATION_DATASET_PATH", "tests/evaluation/dataset/vpp.json")
    with open(path, "r") as f:
        data = json.load(f)

        for id, questions in data.items():
            answers = []
            for question in questions:
                result = await graph.ainvoke({"question": question})
                answers.append(result.get("answer"))

                # Evaluate RAGAS Citations
                evaluate(
                    dataset=EvaluationDataset.from_list(
                        [
                            {
                                "user_input": question,
                                "retrieved_contexts": [
                                    r.get("snippet", "")
                                    for r in result.get("citations", [])
                                ],
                                "response": result.get("answer"),
                            }
                        ]
                    ),
                    metrics=[
                        Faithfulness(),
                        ResponseRelevancy(),
                        LLMContextPrecisionWithoutReference(),
                    ],
                    llm=llm,
                    experiment_name=f"group_{id}_citations",
                )

                await asyncio.sleep(SLEEP_DURATION)

                # Evaluate RAGAS Additional Sources
                evaluate(
                    dataset=EvaluationDataset.from_list(
                        [
                            {
                                "user_input": question,
                                "retrieved_contexts": [
                                    r.get("snippet", "")
                                    for r in result.get("additional_sources", [])
                                ],
                                "response": result.get("answer"),
                            }
                        ]
                    ),
                    metrics=[
                        LLMContextPrecisionWithoutReference(),
                    ],
                    llm=llm,
                    experiment_name=f"group_{id}_additional_sources",
                )

                await asyncio.sleep(SLEEP_DURATION)

            # Calculate average semantic similarity per group
            avg_semantic_similarity_score = avg_semantic_similarity(answers)
            logger.info(
                f"Group {id} - Average Semantic Similarity: {avg_semantic_similarity_score}"
            )
            final_result[id]["avg_semantic_similarity"] = avg_semantic_similarity_score

    logger.info(f"Average Semantic Similarity: {json.dumps(final_result)}")


if __name__ == "__main__":
    asyncio.run(main())
