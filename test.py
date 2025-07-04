# import os
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv

# load_dotenv()

# # step -1: setup llm (mistral with huggingface)
# HF_TOKEN = os.environ.get("HF_TOKEN")
# if not HF_TOKEN:
#     raise ValueError("HuggingFace token not found in environment variables")

# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         huggingfacehub_api_token=HF_TOKEN,
#         max_new_tokens=512,  # Removed quotes to ensure it's an integer
#     )
#     return llm

# # step -2: connect llm with faiss and create chain
# CUSTOM_PROMPT_TEMPLATE = """
# You are a trusted nutrition advisor specializing in infant and child nutrition. Your goal is to provide accurate, safe, and guideline-compliant dietary recommendations for infants and toddlers, based strictly on official sources such as WHO, ICMR, and FSSAI guidelines.

# Use ONLY the context provided below to generate your response. Do not make up any information. If the answer is not present in the context, respond with: "I'm sorry, I don't have that specific information in the official guidelines."

# ---
# 📘 CONTEXT:
# {context}
# ---

# 👶 USER QUERY:
# {question}

# 💡 YOUR RESPONSE (Follow these rules strictly):
# - Always mention the appropriate age group if relevant.
# - Suggest foods, preparation methods, or nutrients only if they are explicitly mentioned in the context.
# - Highlight safety precautions if any.
# - If applicable, cite the source like [WHO, 2021] or [ICMR, p.23].
# - Keep the tone caring, professional, and concise.
# """

# def set_custom_prompt():
#     prompt = PromptTemplate(
#         template=CUSTOM_PROMPT_TEMPLATE,
#         input_variables=["context", "question"]
#     )
#     return prompt

# # load database
# DB_FAISS_PATH = "vectorstore/db_faiss"
# try:
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.load_local(
#         DB_FAISS_PATH, 
#         embedding_model, 
#         allow_dangerous_deserialization=True
#     )
# except Exception as e:
#     raise RuntimeError(f"Failed to load FAISS database: {str(e)}")

# # create QA chain
# try:
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=load_llm(HUGGINGFACE_REPO_ID),
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={'k': 3}),
#         return_source_documents=True,
#         chain_type_kwargs={
#             'prompt': set_custom_prompt(),
#             "document_variable_name": "context"
#         }
#     )
# except Exception as e:
#     raise RuntimeError(f"Failed to create QA chain: {str(e)}")

# # Now invoke with a single query
# try:
#     user_query = input('Write Query Here: ')
#     response = qa_chain.invoke({'query': user_query})  # Changed 'question' to 'query' to match common practice
#     print('RESULT:', response["result"])
#     print("SOURCE DOCUMENTS:", response["source_documents"])
# except Exception as e:
#     print(f"Error during query processing: {str(e)}")


import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

load_dotenv()

# === Step 1: Setup Hugging Face LLM ===
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in .env")

HUGGINGFACE_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        provider="hf-inference"
    )

# === Step 2: Load FAISS DB ===
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# === Step 3: LangChain Prompt Template (Context + Question only) ===
CUSTOM_PROMPT_TEMPLATE = """
You are a trusted nutrition advisor specializing in infant and child nutrition. Your goal is to provide accurate, safe, and guideline-compliant dietary recommendations for infants and toddlers, based strictly on official sources such as WHO, ICMR, and FSSAI guidelines.

Use ONLY the context provided below to generate your response. Do not make up any information. If the answer is not present in the context, respond with: "I'm sorry, I don't have that specific information in the official guidelines."

---
📘 CONTEXT:
{context}
---

👶 USER QUERY:
{question}

💡 YOUR RESPONSE (Follow these rules strictly):
- Always mention the appropriate age group if relevant.
- Suggest foods, preparation methods, or nutrients only if they are explicitly mentioned in the context.
- Highlight safety precautions if any.
- If applicable, cite the source like [WHO, 2021] or [ICMR, p.23].
- Keep the tone caring, professional, and concise.
"""

custom_prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# === Step 4: Setup QA Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"}
)

# === Step 5: Structured Input for Meal Plan Request ===
user_input = {
    "meal_duration": "1",
    "age": "8",
    "diet": "Vegetarian",
    "diet_notes": "ensure iron bioavailability with vitamin C",
    "allergies": "None",
    "nutrient_focus": "ViatminA",
    "foods_tolerated": "Sweet potato, oats",
    "medical_conditions": "None",
    "cultural_preference": "Indian"
}

# === Step 6: Convert to Structured Query ===
structured_query_template = PromptTemplate(
    input_variables=[
        "meal_duration", "age", "diet", "diet_notes", "allergies",
        "nutrient_focus", "foods_tolerated", "medical_conditions", "cultural_preference"
    ],
    template="""
Generate a {meal_duration}-day meal plan for a {age}-month-old infant with these characteristics:
- Diet type: {diet} ({diet_notes})
- Allergies/intolerances: {allergies}
- Key nutrient focus: {nutrient_focus}
- Current tolerated foods: {foods_tolerated}
- Medical considerations: {medical_conditions}
- Cultural preference: {cultural_preference}
"""
)

query = structured_query_template.format(**user_input)

# === Step 7: Invoke the RAG Chain ===
try:
    response = qa_chain.invoke({ "query": query })
    print("\n📘 FINAL RESPONSE:\n", response["result"])
    print("\n🔗 SOURCES:\n", [doc.metadata for doc in response["source_documents"]])
except Exception as e:
    print(f"❌ Error during query processing: {str(e)}")