import pickle
import asyncio

from .eval import generate_rag_responses, evaluate_rag
from .rag import rag, query_rag
from .agent import retrieve_agent


if __name__ == "__main__":

    ########### PRE TESTS ############

    # print(retrieve_agent("Which magazine was started first Arthur's Magazine or First for Women?"))


    ############ INDEXING ############
    # with open("data/all_contexts.pkl", "rb") as f:
    #     knowledge_corpus = pickle.load(f)

    # asyncio.run(rag.ainsert(knowledge_corpus))

    # ############ QUERYING ############

    # input_csv_path = "data/hotpotqa_formatted.csv"
    no_agent_output_csv_path = "data/hotpotqa_no_agent_generated.csv"
    
    # # generte for no-agent RAG
    # generate_rag_responses(input_csv_path, no_agent_output_csv_path, query_rag)

    # # generte for agentic RAG
    agent_output_csv_path = "data/hotpotqa_agent_generated.csv"
    # generate_rag_responses(input_csv_path, agent_output_csv_path, retrieve_agent)

    # ############ EVALUATION ############

    evaluate_rag(agent_output_csv_path, "data/hotpotqa_agent_evaluated.csv")

    """
    agent answer correctness: 0.67
    no-agent answer correctness: 0.54

    agent response relevancy:  0.89
    no-agent response relevancy: 0.82

    """

