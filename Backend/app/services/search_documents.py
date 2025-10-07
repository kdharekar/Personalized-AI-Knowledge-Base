from langchain_community.vectorstores import Chroma
from langchain_openai import  ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# --- Configuration ---
# Must be the same directory used in the indexing service
PERSIST_DIRECTORY = "./vectordb"

# --- RAG Prompt Template ---
# This prompt is engineered to force the LLM to produce a structured JSON output.
RAG_PROMPT_TEMPLATE = """
You are an expert AI assistant for a knowledge base. Your task is to answer user questions based ONLY on the provided document snippets (context).
Your response MUST be in a valid JSON format with the following keys: "answer", "confidence", "missing_info", and "enrichment_suggestion".

**Instructions:**
1.  **Analyze the Context:** Carefully read the provided context below.
2.  **Answer the Question:** Based solely on the context, provide a clear and concise answer to the user's question. If the context does not contain the answer, state that you cannot answer based on the provided information.
3.  **Assess Confidence:** Rate your confidence in the answer on a scale from 0.0 to 1.0, where 1.0 is highly confident. Confidence is high if the context directly and completely answers the question. Confidence is low if the context is only partially relevant or insufficient. If you cannot answer, confidence must be 0.0.
4.  **Identify Missing Information:** If the context is insufficient, explicitly state what specific information is missing that would be needed to fully answer the question. If the context is sufficient, this must be an empty string.
5.  **Suggest Enrichment:** If information is missing, suggest how the knowledge base could be improved (e.g., "Add a document detailing the 'Project Alpha' budget specifications."). If no information is missing, this must be an empty string.
6.  **Handle Irrelevance:** If the documents in the context are not relevant to the question at all, state that the provided information is not relevant to the question, set confidence to 0.0, and suggest what kind of document might be useful.

**Context:**
{context}

**Question:**
{question}

**JSON Output:**
"""

class SearchService:
    def __init__(self):
        """
        Initializes the SearchService by loading the persistent vector store
        and setting up the RAG chain.
        """
        # 1. Initialize the Embedding Function (must match the one used for indexing)
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 2. Load the persistent vector store
        self.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function
        )
        
        # 3. Create a retriever from the vector store
        # This retriever will fetch the top 4 most relevant document chunks
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        # 4. Initialize the LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # 5. Create the RAG Prompt from the template
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # 6. Initialize the JSON Output Parser
        self.json_parser = JsonOutputParser()

        # 7. Build the RAG Chain using LangChain Expression Language (LCEL)
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | self.json_parser
        )
        print("SearchService initialized successfully. Ready to receive queries.")

    def search(self, query: str):
        """
        Performs a search using the RAG chain.

        Args:
            query (str): The user's question.

        Returns:
            dict: A dictionary containing the structured answer from the LLM.
        """
        try:
            print(f"Executing search for query: '{query}'")
            result = self.rag_chain.invoke(query)
            return result
        except Exception as e:
            print(f"An error occurred during search: {e}")
            # In case of a JSON parsing error or other failure, return a structured error
            return {
                "answer": "An error occurred while processing your request.",
                "confidence": 0.0,
                "missing_info": str(e),
                "enrichment_suggestion": "The system may be experiencing an issue. Please try again later."
            }