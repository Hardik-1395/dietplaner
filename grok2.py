import os
from dotenv import load_dotenv
from groq import Groq  # Groq SDK
from langchain_core.prompts import PromptTemplate
#from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
#from langchain.schema import AIMessage, HumanMessage
from langchain_core.language_models import LLM
from langchain_core.runnables import RunnableLambda

#from langchain.llms.base import LLM
from pydantic import PrivateAttr, ConfigDict
from typing import List, Optional
from prompts import custom_prompt


load_dotenv()


# === Load FAISS Vector Store ===
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_FAISS_PATH = "vectorstore/dietplanner_db_faiss"
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


# === Wrap Groq Chat API into LangChain-compatible class ===


class GroqLLM(LLM):
   # model_config = ConfigDict(arbitrary_types_allowed=True)  # ← new style

    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.4
    _client: Groq = PrivateAttr()  # This is how we define a private client field

    def __init__(self, api_key: str, model_name: Optional[str] = None, temperature: float = 0.6):
        super().__init__()
        self._client = Groq(api_key=api_key) 
        self.model = model_name or self.model
        self.temperature = temperature

    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
    # 🔥 Ensure prompt is always string
        if not isinstance(prompt, str):
            if hasattr(prompt, "to_string"):
                prompt = prompt.to_string()
            elif isinstance(prompt, dict):
                prompt = prompt.get("text") or prompt.get("input") or str(prompt)
            else:
                prompt = str(prompt)

        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=800,
            top_p=1,
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    # class Config:
    #     arbitrary_types_allowed = True

# === Instantiate Groq Model ===
llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))

# === QA Chain ===
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=db.as_retriever(search_kwargs={'k': 5}),
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"}
# )


retriever = db.as_retriever(search_kwargs={'k': 5})

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# qa_chain = (
#     {
#         "context": retriever | RunnableLambda(format_docs),
#         "question": lambda x: x["query"],
#     }
#     | custom_prompt
#     | RunnableLambda(lambda x: x.to_string())  # ← convert PromptValue to str
#     | llm
# )


from prompts import structured_query_template
from prompts import user_input
query = structured_query_template.format(**user_input)

retriever = db.as_retriever(search_kwargs={'k': 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

docs = retriever.invoke(query)
context = format_docs(docs)

if not context.strip():
    raise ValueError("No relevant context retrieved from FAISS.")

final_prompt = custom_prompt.format(context=context, question=query)



# === Final Call ===
try:
    response = llm.invoke(final_prompt)
    print("\n📘 FINAL RESPONSE:\n", response)

    print("\n🔗 SOURCES:\n", [doc.metadata for doc in docs])

    docs = retriever.invoke(query)
    print("\n🔗 SOURCES:\n", [doc.metadata for doc in docs])
except Exception as e:
    print(f"❌ Error during query processing: {str(e)}")