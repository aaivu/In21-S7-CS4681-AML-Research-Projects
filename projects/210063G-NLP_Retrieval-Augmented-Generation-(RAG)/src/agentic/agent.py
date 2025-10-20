from typing import TypedDict
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
from textwrap import dedent
from trustcall import create_extractor
from langgraph.graph import StateGraph, END

from .utils import dict_to_str
from .rag import retrieve_rag


MAX_ITERATIONS = 6



########## LLM MODEL ############

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    # model = "gpt-5-nano"
)



########### PLANING NODE ###########

planner_prompt = """
You are a PLANNER, a strategic AI agent responsible for creating an optimal and efficient data retrieval plan to answer a given user query.

Your primary objective is to analyze the user's query, assess the current progress of information retrieval, and determine the most logical next step. This involves breaking down complex queries into simpler, actionable retrieval tasks and iterating until all necessary information has been gathered.

### Core Mandate & Workflow

You will operate in an iterative loop, following these phases:

1.  **Initial Analysis & Plan Formulation:**
    * When no progress has been made, conduct an initial analysis of the query.
    * Classify the query into one of the following types to determine the initial strategy:
        * **Direct Query:** A simple question that can likely be answered with a single, well-formulated search.
        * **Decomposable Query:** A question that contains multiple, independent sub-questions. Your plan should be to retrieve information for each sub-question separately.
        * **Sequential / Multi-Hop Query:** A complex question where the answer to one part is required to formulate the query for the next part. Your plan must outline these sequential steps.

2.  **Iterative Execution & Refinement (The "Critic" Loop):**
    * Examine the `PROGRESS SO FAR`, which includes the history of actions and retrieved data.
    * **Critique the latest retrieved data:** Is it relevant and sufficient? Does it directly answer a part of the query?
    * Based on your critique:
        * **If sufficient information has been gathered:** Conclude the process by setting the `ACTION` to `stop`.
        * **If the data is irrelevant or insufficient:** Refine your plan. This may involve rephrasing the previous query or altering the overall strategy (e.g., changing from a direct query approach to a multi-hop approach).
        * **If the data is relevant and part of a larger plan:** Formulate the next query in the sequence or the next part of the decomposed query.

3. **Self Stop**
   * Examine the `PROGRESS SO FAR`, including all previous thoughts, actions, and retrieved data.
   * If the retrieved data has been **irrelevant or unhelpful** for **three consecutive steps**, initiate a **self-stop**.
   * When performing a self-stop:
      * **THOUGHT:** Clearly explain *why* you are stopping (e.g., “Further retrievals produced repetitive or non-relevant information; additional exploration is unlikely to add value.”).
      * **ACTION:** Set the action explicitly to `"stop"`.


4.  **Output Generation:**
    * **THOUGHT:** Clearly articulate your analysis, critique of the current progress, and the reasoning behind your next action.
    * **ACTION:** State the precise action to be taken. This will either be `retrive the data for query/ies: [your query/ies here]` or `stop`.


---

### Examples


**Example 1: Querying not progessig**

**Question** who is the CEO of Dialog?

**PROGRESS SO FAR**
This is the Start. Analyze the Query and think and plan the next step.

**THOUGHT:** The user is asking for a single piece of information regarding the CEO of Dialog Group. This is a direct query that can be resolved with a straightforward search for the current CEO of the company. I will proceed to formulate a query to retrieve this information.
**ACTION:** retrive the data for query: CEO of Dialog Group
**RETRIVED DATA:**
current CEO of Dialog Group : Dummy Retrived data for current CEO of Dialog Group from local


**THOUGHT:** The retrieved data contains a placeholder response, implying that the actual current CEO of Dialog Group has not been retrieved successfully. Since this information is not sufficient or relevant, I need to refine my query to ensure I retrieve accurate and specific information about the CEO. I may need to specify the format of the query or check for updates on the company in alternative sources.
**ACTION:** retrive the data for query: current CEO of Dialog Group (2023)
**RETRIVED DATA:**
What is the name of the current CEO of Dialog Group as of 2023? : Dummy Retrived data for What is the name of the current CEO of Dialog Group as of 2023? from local


**THOUGHT:** The retrieved data still comprises placeholders, indicating that I am not accessing suitable or up-to-date sources to obtain accurate information about the current CEO of Dialog Group. It's essential to adjust my strategy. I will refine the query to include possible additional terms that might yield more accurate results by broadening the search parameters or using different phrasing. For instance, trying a more general search with common keywords could help discover relevant updates about the company's leadership.
**ACTION:** retrive the data for query: who is the current CEO of Dialog Group 2023
**RETRIVED DATA:**
who is the current CEO of Dialog Group : Dummy Retrived data for who is the current CEO of Dialog Group from global
Dialog Group leadership updates : Dummy Retrived data for Dialog Group leadership updates from global


**THOUGHT:** I have tried three consecative retrivals and all failed, I will initiate a self-stop due to the lack of meaningful progress.
**ACTION:** stop

---

**Example 2: Simple Direct Query**

**QUESTION:** "who is the CEO of Dialog Axiata PLC?"

**PROGRESS SO FAR**
This is the Start. Analyze the Query and think and plan the next step.

**THOUGHT:** The user is asking for a single piece of information: the name of the CEO of a specific company. This is a direct query. I can formulate a simple search to retrieve this information.
**ACTION:** retrive the data for query: CEO of Dialog Axiata PLC
**RETRIEVED DATA:** The current Group Chief Executive of Dialog Axiata PLC is Supun Weerasinghe.

**THOUGHT:** The retrieved data directly answers the user's question. All necessary information has been obtained. The process can be concluded.
**ACTION:** stop

---

**Example 3: Decomposable (Comparative) Query**

**QUESTION:** "Compare the battery life and camera specifications of the iPhone 15 Pro Max and the Samsung Galaxy S24 Ultra."

**PROGRESS SO FAR**
This is the Start. Analyze the Query and think and plan the next step.

**THOUGHT:** This query requires comparing two products across two different metrics (battery life and camera specs). This is a decomposable query. I need to gather four distinct pieces of information. I can gather all the specs at onece since there is no dependancy.
**ACTION:** retive the data for queryies: battery life of iPhone 15 Pro Max , camera specifications of iPhone 15 Pro Max , battery life of Samsung Galaxy S24 Ultra , camera specifications of Samsung Galaxy S24 Ultra
**RETRIEVED DATA:** The iPhone 15 Pro Max has up to 29 hours of video playback. Its camera system includes a 48MP Main, 12MP Ultra Wide, and 12MP 5x Telephoto lens.The Samsung Galaxy S24 Ultra is equipped with a 5,000mAh battery. Its camera setup features a 200MP Wide-angle, 12MP Ultra-wide, 50MP 5x optical zoom, and 10MP 3x optical zoom lens.

**THOUGHT:** I have now retrieved the battery and camera specifications for both devices mentioned in the query. All the necessary data for a comparison has been collected. The process is complete.
**ACTION:** stop

---

**Example 4: Complex Multi-Hop Query**

**QUESTION:** "Who is the current head of government in the country where the company that developed the GPT-3 language model is headquartered?"

**PROGRESS SO FAR**
This is the Start. Analyze the Query and think and plan the next step.

**THOUGHT:** This is a multi-hop query that requires a sequence of discoveries.
1.  First, I need to identify the company that developed GPT-3.
2.  Second, I need to find out where that company is headquartered.
3.  Third, I need to find the current head of government for that country.
I will start with the first step.
**ACTION:** retive the data for query: what company developed the GPT-3 language model
**RETRIEVED DATA:** GPT-3 was created by the company OpenAI.

**THOUGHT:** I have completed the first hop. Now I need to find the headquarters of OpenAI. This is the second step in my plan.
**ACTION:** retive the data for query: where is OpenAI headquartered
**RETRIEVED DATA:** OpenAI is headquartered in San Francisco, California, USA.

**THOUGHT:** I have completed the second hop and identified the country as the USA. The final step is to find the current head of government for the United States.
**ACTION:** retive the data for query: current head of government of the USA
**RETRIEVED DATA:** The current head of government of the United States is the President, Joe Biden.

**THOUGHT:** I have successfully completed all three hops of the query. I have identified the company, its location, and the corresponding head of government. The process is complete.
**ACTION:** stop

---

**Example 5: Query Requiring Criticism and Refinement**

**QUESTION:** "What was the Q2 2024 revenue for the parent company of WhatsApp?"

**PROGRESS SO FAR**
This is the Start. Analyze the Query and think and plan the next step.

**THOUGHT:** This is a multi-hop question. I first need to identify the parent company of WhatsApp, and then find its Q2 2024 revenue.
**ACTION:** retive the data for query: parent company of WhatsApp
**RETRIEVED DATA:** WhatsApp is owned by Meta Platforms, Inc.

**THOUGHT:** I have the parent company: Meta Platforms. Now I need to find its revenue for a specific quarter. A direct query should work.
**ACTION:** retive the data for query: Meta Platforms revenue
**RETRIEVED DATA:** Meta Platforms reported revenue of $134.90 billion for the full year of 2023. For the fourth quarter of 2023, revenue was $40.11 billion.

**THOUGHT:** **Criticism:** The retrieved data is not specific enough. It provides annual and Q4 2023 revenue, but the query explicitly asks for **Q2 2024** revenue. My previous query was too broad. **Refinement:** I need to make a more specific query that includes both the company and the exact time period.
**ACTION:** retive the data for query: Meta Platforms Inc. Q2 2024 revenue report
**RETRIEVED DATA:** For the second quarter of 2024, Meta reported revenue of $42.5 billion.

**THOUGHT:** This retrieval successfully found the specific piece of information requested in the user's query. The process is now complete.
**ACTION:** stop



**QUESTION:**
{query}

**PROGRESS SO FAR**
{progress}
"""


class Thought(BaseModel):
    thought: str = Field(
        ...,
        description=(
            "The internal reasoning or reflection that explains *why* the next action "
            "is chosen. This captures the agent’s cognitive process — how it interprets "
            "the current state, retrieved information, and goals before deciding what to do next."
        )
    )

    action: str = Field(
        ...,
        description=(
            "The concrete operation or next step to execute based on the thought. "
            "For example: performing a retrieval, refining a query, comparing results, "
            "summarizing findings, or stopping the reasoning loop."
        )
    )

    completed: bool = Field(
        ...,
        description=(
            "Indicates whether all necessary and relevant information has been successfully gathered. "
            "Set to True when the reasoning process can confidently conclude and produce a final answer."
            "Indicates that the agent has decided to **gracefully stop on its own** due to a lack of meaningful progress. "
            "Typically triggered after several consecutive iterations where retrieved data is irrelevant or repetitive, "
            "signaling that continued reasoning is unlikely to yield additional value."
        )
    )



def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:

    query = state["query"]
    progress = state.get("progress")
    iteration = state.get("iteration", 0) + 1

    if iteration > MAX_ITERATIONS:
      progress += dedent(f"""
    **SYSTEM:** Max iteration reached.
    **ACTION:** Abroted
    """)
      return {"progress": progress, "completed": True}


    # handle the first history
    if not progress:
      progress =f"""\nThis is the Start. Analyze the Query and think and plan the next step.\n\n"""


    msg = planner_prompt.format(progress=progress, query=query)
    thinking_llm = llm.with_structured_output(Thought)
    thought = thinking_llm.invoke(msg)

    progress += dedent(f"""
    **THOUGHT:** {thought.thought}
    **ACTION:** {thought.action}
    """)

    return {"current_thought":thought , "iteration":iteration, "completed": thought.completed, "progress":progress }




####### EXECUTOR NODE ######

class RefinedKnowledge(BaseModel):
  query:str = Field(..., description = "query related to given knowledge")
  refined_knowledge: str = Field(..., description ="refined knowledge")

knowledge_refining_prompt = """
You are an expert in unstructured knowledge refinement. You are in the middle of
answering a complex question. You are given three text inputs:
  - **INITIAL QUESTION**
  - **TARGET QUESTION**
  - **RETRIEVED KNOWLEDGE**

Your task is to refine the **RETRIEVED KNOWLEDGE** so that it provides the most relevant,
concise, and accurate information needed to answer the **TARGET QUESTION**, while keeping
it coherent with the progress so far.

### INSTRUCTIONS
1. There are two types of questions:
   - **direct**: The answer can be directly extracted from the RETRIEVED KNOWLEDGE.
     → Return a short, direct fact as the refined_knowledge.
   - **analytical**: The answer requires summarizing or synthesizing information.
     → Return a concise, well-organized summary relevant to the TARGET QUESTION.

2. If the RETRIEVED KNOWLEDGE does **not** contain enough information to answer the TARGET QUESTION,
   set refined_knowledge to: `answer not found`

3. Only use information present in the RETRIEVED KNOWLEDGE — never add, assume, or invent facts.

4. The refined_knowledge should:
   - Be logically consistent with the INITIAL QUESTION.
   - Be concise (1–2 sentences for direct, 3–5 for analytical).
   - Maintain factual accuracy and coherence.
   - Exclude irrelevant or repetitive details.

5. Format your final response strictly as a JSON object matching this schema:

{{
  "query": "<TARGET QUESTION>",
  "refined_knowledge": "<refined text or `answer not found`>"
}}


### EXAMPLES

**Example 01 - Direct Answer**

**INITIAL QUESTION:** "What was the Q2 2024 revenue for the parent company of WhatsApp?"
**TARGET QUESTION:** "What is the parent company of WhatsApp?"
**RETRIEVED KNOWLEDGE:** WhatsApp, founded in 2009 by Brian Acton and Jan Koum, was acquired by Meta Platforms Inc., the parent company of Facebook and Instagram.
OUTPUT:
{{
  "query": "What is the parent company of WhatsApp?",
  "refined_knowledge": "The parent company of WhatsApp is Meta Platforms Inc."
}}

**Example 02 - Analytical Answer**

**INITIAL QUESTION:** "Explain WhatsApp's business model and revenue sources."
**TARGET QUESTION:** "How does WhatsApp generate revenue?"
**RETRIEVED KNOWLEDGE:** WhatsApp provides messaging services free of charge to users. It offers WhatsApp Business tools for SMEs and paid APIs for large enterprises to communicate with customers.
OUTPUT:
{{
  "query": "How does WhatsApp generate revenue?",
  "refined_knowledge": "WhatsApp generates revenue mainly through its business-oriented services, including WhatsApp Business and paid enterprise APIs that enable companies to connect with customers."
}}

If the information is missing:
OUTPUT:
{{
  "query": "<TARGET QUESTION>",
  "refined_knowledge": "answer not found"
}}


Now , Handle the following scenario:

**INITIAL QUESTION:** {initial_question}
**TARGET QUESTION:** {target_question}
**RETRIEVED KNOWLEDGE:** {retrieved_knowledge}
OUTPUT:

"""

def knowledge_refiner( initial_question: str , target_question: str , retrieved_knowledge: str)->str:
  formatted_prompt = knowledge_refining_prompt.format(
      initial_question=initial_question,
      target_question=target_question,
      retrieved_knowledge=retrieved_knowledge
  )
  refining_llm = llm.with_structured_output(RefinedKnowledge)
  refined_knowledge = refining_llm.invoke(formatted_prompt)
  return refined_knowledge


class Query(BaseModel):
    query: str = Field(..., description="The query to be executed")
    mode: Literal["local", "global"] = Field(..., description = "mode of retrieval")

class Queries(BaseModel):
    queries: List[Query] = Field(..., description="The list of queries to be executed")

query_extractor_prompt = """
You are an expert query decomposition and refinement agent.

Your goal is to extract **retrievable queries** from the provided THOUGHT and ACTION.
Each query should be a concrete, retrievable statement that helps fulfill the reasoning step.
The final output must be a valid JSON object that adheres to the provided tool schema.

***Query Types***

1. **Direct (Single Query)**
   - The thought/action can be answered directly in a single step.
   - Use **local mode** if it requires facts and pin point knowledge.
   - Use **global mode** if it requires the overviews

2. **Analytical / Comparative / Decomposed (Multi Query)**
   - The reasoning requires multiple distinct data points for comparison or synthesis.
   - Return multiple queries under the `queries` list.

3. **Multi-hop Reasoning (Sequential)**
   - The answer requires step-by-step reasoning where one query’s result informs the next.
   - At each stage, only one query should be generated.


***Instructions***

- Determine whether the reasoning requires one or multiple queries.
- Classify each query as local or global based on information scope.
- Keep queries concise, factual, and directly retrievable.
- Do not include reasoning, only the final queries in valid JSON format.

THOUGHT:
{thought}

ACTION:
{action}

"""

def executor(state: Dict[str, Any]) -> Dict[str, Any]:

    initial_query = state["query"]
    thought = state["current_thought"]
    iteration = state["iteration"]

    final_str=f"**RETRIVED DATA:**"

    extractor_llm = llm.with_structured_output(Queries)

    # use extractor to extract the queries and refine them
    queries = extractor_llm.invoke(query_extractor_prompt.format(thought=thought.thought, action=thought.action)).queries

    for q in queries:
      
      # add mode
      retrieved_data = dict_to_str(retrieve_rag(q.query, q.mode))
      refined_data = knowledge_refiner(initial_query, q.query, retrieved_data).refined_knowledge
      final_str+=f"\n{q.query} : {refined_data}"

    final_str+="\n\n"

    return {"progress": state["progress"]+final_str }



######### ANSWER GENERATOR NODE ##########

class Answer(BaseModel):
  answer: str = Field(..., description = "final answer")


answer_generator_prompt = """
You are a Synthesizer AI. Your sole purpose is to take the user's original question and the full, step-by-step progress log of a data retrieval agent, and generate a single, clear, and user-friendly final answer.

Analyze the provided `PROGRESS SO FAR` log to determine the outcome of the agent's work. Based on the outcome, formulate your response according to one of the scenarios below.

---

### Scenarios & Instructions

**Scenario 1: Successful Data Retrieval**
- **How to identify:** The log ends with an `ACTION: stop` and the final `RETRIEVED DATA` step contains specific, concrete information that answers the user's question.
- **Your Task:**
    1.  Locate the final, successful `RETRIEVED DATA` in the log.
    2.  Synthesize this information into a direct and concise answer.
    3.  Ignore all previous failed attempts, dummy data, and the agent's internal thoughts.
    4.  Present the answer clearly and confidently.

**Scenario 2: Process Aborted by System (Partial Answer Possible)**
- **How to identify:** The log ends with a system message like `ACTION: Abroted` or `SYSTEM: Max iteration reached.`
- **Your Task:**
    1.  First, carefully review the **entire log** for any concrete, non-dummy information that was successfully retrieved in the steps *before* the process was aborted.
    2.  **If you find relevant partial information:** Synthesize that data into a partial answer. You **must** clearly state that this information is incomplete and the process was stopped due to system limits before a full answer could be found.
    3.  **If the log contains no useful information (only dummy data or failed retrievals):** Apologize and state that the process was stopped due to system limits before any conclusive information could be found.

**Scenario 3: Agent Stopped Due to No Progress**
- **How to identify:** The log ends with an `ACTION: stop`, but the agent's final `THOUGHT` indicates it is stopping due to a lack of meaningful progress, and the retrieved data is unhelpful.
- **Your Task:**
    1.  Acknowledge that you were unable to find a reliable answer.
    2.  Politely explain that after several attempts with different strategies, no conclusive information could be found from the available sources.
    3.  Do not present any of the dummy data.

---

### Examples

**Example 1: Successful Retrieval**

**QUESTION:** "What was the Q2 2024 revenue for the parent company of WhatsApp?"
**PROGRESS SO FAR:**
... [log showing steps to find Meta, then its revenue] ...
**RETRIEVED DATA:** For the second quarter of 2024, Meta reported revenue of $42.5 billion.
**THOUGHT:** This retrieval successfully found the specific piece of information requested...
**ACTION:** stop

**FINAL ANSWER:**
The parent company of WhatsApp, Meta Platforms, Inc., reported a revenue of $42.5 billion for the second quarter (Q2) of 2024.

---

**Example 2: Process Aborted with Partial Answer**

**QUESTION:** "Who is the CEO of the company that owns GitHub, and what was their revenue in 2024?"
**PROGRESS SO FAR:**
**THOUGHT:** This is a multi-hop query. First I need to find the company that owns GitHub. Then find its CEO and 2024 revenue.
**ACTION:** retrive the data for query: who owns GitHub
**RETRIEVED DATA:** GitHub was acquired by and is a subsidiary of Microsoft.
**THOUGHT:** I have found the parent company is Microsoft. Now I need to find the CEO of Microsoft.
**ACTION:** retrive the data for query: CEO of Microsoft
**RETRIEVED DATA:** CEO of Microsoft : Dummy Retrived data for CEO of Microsoft from local
**SYSTEM:** Max iteration reached.
**ACTION:** Abroted

**FINAL ANSWER:**
I was able to find some of the information you requested. GitHub is owned by Microsoft.

However, I must clarify that this is a partial answer. The process was stopped due to system limits before I could determine the name of Microsoft's current CEO and their 2024 revenue.

---

**Example 3: Stopped Due to No Progress**

**QUESTION:** "Who is the CEO of Dialog Group?"
**PROGRESS SO FAR:**
... [log showing multiple attempts with only dummy data] ...
**THOUGHT:** The retrieved data is still placeholders... I will initiate a self-stop due to the lack of meaningful progress...
**ACTION:** stop

**FINAL ANSWER:**
I'm sorry, but I was unable to find a reliable answer to your question. After trying several different search strategies, I could not locate conclusive information about the current CEO of Dialog Group from the available sources.

---

### Your Task

Now, generate the final answer for the following query based on the provided log.

**QUESTION:**
{query}

**PROGRESS SO FAR:**
{progress}

**FINAL ANSWER:**
"""

def answer_generator(state: Dict[str, Any]) -> Dict[str, Any]:
  progress = state["progress"]
  query = state["query"]

  answer_llm = llm.with_structured_output(Answer)
  answer = answer_llm.invoke(answer_generator_prompt.format(query=query, progress=progress))

  return {"answer": answer.answer}




class AgentState(TypedDict):
    query: str
    progress: str | None = None
    current_thought: Thought | None = None
    iteration: int = 0
    completed:bool = False
    answer:str | None = None


graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor)
graph.add_node("answer_generator", answer_generator)

# Define transitions
graph.set_entry_point("planner")
graph.add_edge("executor", "planner")

# If critic says done → END, else loop back to planner
def should_continue(state: Dict[str, Any]) -> bool:
    return not state["completed"] and state["iteration"] <= MAX_ITERATIONS

graph.add_conditional_edges("planner", should_continue, {True:"executor", False:"answer_generator"})
graph.add_edge("answer_generator", END)

# Compile the graph
retrieval_agent = graph.compile()


def retrieve_agent(query: str) -> str:
    final_state = retrieval_agent.invoke({"query": query})
    return final_state["answer"]

