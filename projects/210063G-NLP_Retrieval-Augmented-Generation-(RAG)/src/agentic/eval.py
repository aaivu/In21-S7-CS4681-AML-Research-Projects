import pandas as pd
from tqdm import tqdm  # optional: for progress bar
from ragas import EvaluationDataset, evaluate
from ragas.metrics import AnswerCorrectness, ResponseRelevancy, BleuScore, RougeScore
from ragas.llms.base import llm_factory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def evaluate_rag(
    input_csv_path: str,
    output_csv_path: str,
    llm_model=None,
    metrics=None,
    embedding_model=None,
):
    """
    Evaluate a CSV of RAG outputs using RAGas metrics with embeddings internally used by metrics.
    """
    # Load CSV
    df = pd.read_csv(input_csv_path)

    # Filter out rows with null/NaN responses
    original_count = len(df)
    df = df.dropna(subset=["response"])  # Remove rows where response is NaN
    df = df[df["response"].str.strip() != ""]  # Remove rows with empty responses
    filtered_count = len(df)

    print(f"Original dataset: {original_count} rows")
    print(f"After filtering null responses: {filtered_count} rows")
    print(f"Filtered out: {original_count - filtered_count} rows")

    # Prepare dataset
    dataset_list = []
    for _, row in df.iterrows():
        dataset_list.append(
            {
                "user_input": row["user_input"],
                "response": row["response"],
                "reference": row["reference"],
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset_list)

    # Default LLM using new factory (avoids deprecated wrapper)
    if llm_model is None:
        llm_model = llm_factory("gpt-4o-mini")

    # Default embedding model
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    # Default metrics
    if metrics is None:
        metrics = [
            AnswerCorrectness(),
            ResponseRelevancy(),
            BleuScore(),
            RougeScore(),
        ]

    # Run evaluation
    results = evaluate(dataset=evaluation_dataset, metrics=metrics, llm=llm_model)

    # Convert results to pandas DataFrame
    results_df = results.to_pandas()

    # Save updated CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"✅ Updated CSV with metrics saved to: {output_csv_path}")


def generate_rag_responses(input_csv_path: str, output_csv_path: str, query_rag):
    """
    Generates responses for each user_input in a CSV using query_rag() and saves the results.

    Args:
        input_csv_path (str): Path to the input CSV containing 'user_input' column.
        output_csv_path (str): Path to save the updated CSV with 'response' column.
        query_rag (function): Function that takes a string (query) and returns a string (response).
    """
    # Load CSV
    df = pd.read_csv(input_csv_path)

    # Check if user_input column exists
    if "user_input" not in df.columns:
        raise ValueError("❌ The CSV must contain a 'user_input' column.")

    # Create or update response column
    responses = []

    print("⚙️ Generating responses for each user_input...")
    for query in tqdm(df["user_input"], desc="Processing"):
        try:
            answer = query_rag(query)
        except Exception as e:
            print(f"Error generating response for query '{query}': {e}")
            answer = ""
        responses.append(answer)

    df["response"] = responses

    # Save updated CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Responses saved to: {output_csv_path}")
