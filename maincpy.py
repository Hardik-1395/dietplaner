import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv()


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in .env")
HUGGINGFACE_REPO_ID="HuggingFaceH4/zephyr-7b-beta"

# def load_llm(huggingface_repo_id):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         huggingfacehub_api_token=HF_TOKEN,  # Correct place to put token
#         model_kwargs={
#             "max_new_tokens": 512  # Use integer, not string
#         }
#     )
#     return llm
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        provider="hf-inference"
    )


# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
You are a diet planning assistant specialized in pregnancy nutrition. Use the given context to create a customized daily diet plan for a pregnant mother.
You are a trusted nutrition advisor specializing in pregnancy nutrition. Your goal is to provide accurate, safe, and guideline-compliant dietary recommendations for a pregnant mother, based strictly on official sources such as WHO, ICMR, and FSSAI guidelines.

Use ONLY the context provided below to generate your response. Do not make up any information. If the answer is not present in the context, respond with: "I'm sorry, I don't have that specific information in the official guidelines."


Context:
{context}
----

USER QUERY:
{question}

Instructions:
- Use only the information from the context to generate the response.
- Do not make up anything that is not mentioned in the context.
- If there's not enough context to generate the answer, reply:
  "Sorry, I could not find relevant dietary guidance for this request in the available documents."
- Structure your output exactly as shown below. Do not skip or reorder the sections.

Output Format:

1. Daily Meal Plan
   - Breakfast:
   - Mid-morning Snack:
   - Lunch:
   - Evening Snack:
   - Dinner:
   - Before Bed (if applicable):

2. Recommended Portion Sizes & Key Nutritional Highlights
   - For each meal, briefly mention ideal portion sizes (e.g., 1 bowl, 2 rotis, etc.) and nutrients (e.g., rich in protein, iron, calcium, fiber).

3. Pregnancy-Specific Dietary Guidelines
   - Add 3 to 5 guidelines that apply based on the trimester, allergies, regional preference, or health goals.

Start directly with section 1 and follow all 3 sections in order. Do not include any extra commentary.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=[
            "context","question"
            
        ]
    )
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':5}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE),"document_variable_name": "context"}
)

# # Now invoke with a single query
# user_query=input("Write Query Here: ")
# response=qa_chain.invoke({'query': user_query})
# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])


# Collect user inputs for all prompt variables
trimester = input("Enter trimester (e.g., First, Second, Third): ")
diet_type = input("Enter diet type (e.g., Balanced, High-protein, Low-carb): ")
allergies = input("List any allergies (comma-separated, or 'None'): ")
goals = input("Enter any health goals (e.g., increase iron, manage weight): ")
menu_type = input("Veg or Non-Veg: ")
region = input("Enter your region (e.g., South India, North India): ")

# You can still take a high-level question to simulate user intent
user_query = input("Ask your question (e.g., Suggest a meal plan): ")

# Build the input dictionary
user_inputs = {
    "query": user_query,   # Replace this with your loaded context from vector DB
    "trimester": trimester,
    "diet_type": diet_type,
    "allergies": allergies,
    "goals": goals,
    "menu_type": menu_type,
    "region": region
}
structured_query_template = PromptTemplate(
    input_variables=[
            "trimester",
            "diet_type",
            "allergies",
            "goals",
            "menu_type",
            "region"
    ],
    template="""
Generare a well structured meal plan for the pregnant mother using below key inputs:
Trimester: {trimester}
Diet Type: {diet_type}
Allergies: {allergies}
Health Goals: {goals}
Menu Type: {menu_type}
Region: {region}
"""
)
query = structured_query_template.format(**user_inputs)

# Now invoke the chain
#response = qa_chain.invoke(query_inputs)

# # Output
# print("\nRESULT:\n", response["result"])
# #print("\nSOURCE DOCUMENTS:\n", response["source_documents"])

# if not response["source_documents"]:
#     print("No relevant documents found.")
# else:
#     print("\nRESULT:\n", response["result"])

# Retrieve documents manually (optional but gives you control)
# Retrieve documents manually
docs = db.similarity_search(user_inputs["query"], k=3)
#query = structured_query_template.format(**user_input)
if not docs:
    print("\nSorry, I could not find relevant dietary guidance for this request in the available documents.")
else:
    # # Inject context and run using invoke
    # try:
    #     result = qa_chain.combine_documents_chain.invoke({
    #         "input_documents": docs,
    #         **query_inputs
    #     })
    #     print("\nRESULT:\n", result)
    # except Exception as e:
    #     print("❌ Error:", e)

    try:
        response = qa_chain.invoke({ "query": query })
        print("\n📘 FINAL RESPONSE:\n", response["result"])
        print("\n🔗 SOURCES:\n", [doc.metadata for doc in response["source_documents"]])
    except Exception as e:
        print(f"❌ Error during query processing: {str(e)}")

