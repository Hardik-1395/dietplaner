from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .grok import qa_chain, structured_query_template

meal_router = APIRouter()

class Userconstraint(BaseModel):
    pregnancy_month : str
    allergies : str
    nutrient_focus: str
    medical_condition : str
    cultural_preference : str
    preference : str

@meal_router.get("/test")
async def home():
    return {"message": "Meal Gen API is running"}


@meal_router.post("/generate")
async def generate_meal_plan(payload: Userconstraint):
     try:
        prompt = structured_query_template.format(**payload.dict())
        result = qa_chain.invoke({"query": prompt})
        sources = [
            {"source": doc.metadata.get("source", "Unknown")}
            for doc in result.get("source_documents", [])
        ]
        return JSONResponse(content={"result": result["result"], "sources": sources})
     except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


