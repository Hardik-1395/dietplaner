import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaaMeal · Pregnancy Diet Planner",
    page_icon="🤰",
    layout="centered",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream: #fdf6ee;
    --blush: #f5c5a3;
    --terra: #c4714a;
    --deep:  #2d1a0e;
    --sage:  #7a9e7e;
    --sage-light: #d4e8d6;
    --card:  #fffaf5;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--deep);
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: var(--terra);
    margin-bottom: 0.25rem;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.05rem;
    color: #7a5c45;
    font-weight: 300;
}

/* Section headers */
.section-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: var(--terra);
    margin: 1.5rem 0 0.5rem;
    border-left: 4px solid var(--blush);
    padding-left: 0.6rem;
}

/* Cards around form groups */
.form-card {
    background: var(--card);
    border: 1px solid #eeddd0;
    border-radius: 16px;
    padding: 1.5rem 1.5rem 0.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 12px rgba(196,113,74,0.06);
}

/* Generate button */
div.stButton > button {
    background: linear-gradient(135deg, var(--terra), #d4845a) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 18px rgba(196,113,74,0.3) !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* Meal plan output */
.meal-card {
    background: var(--card);
    border: 1px solid #eeddd0;
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 4px 24px rgba(196,113,74,0.08);
    white-space: pre-wrap;
    font-size: 0.97rem;
    line-height: 1.8;
    color: #1a1a1a !important;
}
.meal-card * {
    color: #1a1a1a !important;
}
.meal-card h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--terra);
    font-size: 1.4rem;
    margin-bottom: 1rem;
}

/* Sources */
.source-tag {
    display: inline-block;
    background: var(--sage-light);
    color: #3a6b3e;
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    margin: 0.2rem;
}

/* Streamlit widget overrides */
.stSelectbox label, .stTextInput label, .stTextArea label, .stSlider label,
p, label, .stMarkdown p {
    font-weight: 500 !important;
    color: #2d1a0e !important;
    opacity: 1 !important;
}
/* Force all label elements to full opacity */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
.stSelectbox label p,
.stTextInput label p,
.stTextArea label p {
    color: #2d1a0e !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div {
    border-radius: 10px !important;
    border-color: #eeddd0 !important;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Hero header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🤰 MaaMeal</h1>
    <p>Personalised pregnancy meal plans powered by nutritional science</p>
</div>
""", unsafe_allow_html=True)

# ─── Load models (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading nutrition knowledge base…")
def load_resources():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        "vectorstore/dietplanner_db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return db, client

db, groq_client = load_resources()

# ─── Prompts (inline so app.py is self-contained) ───────────────────────────
CUSTOM_PROMPT_TEMPLATE = """
Using ONLY the context provided, generate a structured meal plan covering Breakfast, Lunch, and Dinner that meets the user's requirements exactly.
- Include at least 5 food items per meal.
- Do NOT add any additional commentary or sections—return nothing but the meal plan.

---
📘 CONTEXT:
{context}
---

🤰 USER QUERY:
{question}

💡 YOUR RESPONSE (Follow these rules strictly):
MEAL PLAN FORMAT:

Breakfast:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Lunch:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Dinner:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

- Highlight safety precautions such as foods to avoid or hygiene practices.
"""

custom_prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

QUERY_TEMPLATE = """Generate a personalized meal plan for a pregnant woman with the following characteristics:
- Stage of pregnancy: {pregnancy_month}
- Diet type: {diet_type}
- Allergies or intolerances: {allergies}
- Key nutrient focus: {nutrient_focus}
- Cultural preference: {cultural_preference}
- Personal preferences or dislikes: {preference}
- Medical conditions: {medical_conditions}"""

# ─── Groq caller ────────────────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1500,
        top_p=1,
    )
    return completion.choices[0].message.content

# ─── Input form ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Tell us about your pregnancy</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pregnancy_month = st.selectbox(
            "Month of pregnancy",
            options=[str(i) for i in range(1, 10)],
            format_func=lambda x: f"Month {x}  (Trimester {1 if int(x)<=3 else 2 if int(x)<=6 else 3})",
        )
    with col2:
        diet_type = st.selectbox(
            "Diet type",
            ["Vegetarian", "Non-Vegetarian", "Vegan", "Eggetarian", "Jain"],
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Dietary needs & health</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        nutrient_focus = st.selectbox(
            "Key nutrient focus",
            ["Calcium", "Iron", "Folate", "Protein", "Omega-3", "Fibre", "Vitamin D"],
        )
    with col4:
        cultural_preference = st.selectbox(
            "Cultural / regional cuisine",
            ["North Indian", "South Indian", "Bengali", "Gujarati", "Punjabi",
             "Maharashtra", "Pan-Indian", "Continental", "No preference"],
        )

    allergies = st.text_input(
        "Allergies or intolerances",
        placeholder="e.g. lactose, gluten, nuts — or type None",
        value="None",
    )
    medical_conditions = st.text_input(
        "Medical conditions (if any)",
        placeholder="e.g. gestational diabetes, anaemia — or type None",
        value="None",
    )
    preference = st.text_area(
        "Personal preferences or goals",
        placeholder="e.g. avoid spicy food, want to manage weight, prefer light dinners…",
        height=90,
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("✨ Generate My Meal Plan")

# ─── Generation ─────────────────────────────────────────────────────────────
if generate:
    if not preference.strip():
        st.warning("Please add at least one personal preference or goal.")
        st.stop()

    query = QUERY_TEMPLATE.format(
        pregnancy_month=pregnancy_month,
        diet_type=diet_type,
        allergies=allergies.strip() or "None",
        nutrient_focus=nutrient_focus,
        cultural_preference=cultural_preference,
        preference=preference.strip(),
        medical_conditions=medical_conditions.strip() or "None",
    )

    with st.spinner("Curating your personalised meal plan…"):
        retriever = db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        if not context.strip():
            st.error("Could not retrieve relevant nutritional context. Please check your FAISS database.")
            st.stop()

        final_prompt = custom_prompt.format(context=context, question=query)
        response = call_groq(final_prompt)

    # ── Display result ──
    st.markdown(f"""
    <div class="meal-card">
        <h3>🍽️ Your Personalised Meal Plan — Month {pregnancy_month}</h3>
        {response.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)

    # ── Sources ──
    if docs:
        st.markdown("<br>**📚 Sources used from knowledge base:**", unsafe_allow_html=True)
        source_html = ""
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                short = os.path.basename(src) if "/" in src or "\\" in src else src
                source_html += f'<span class="source-tag">📄 {short}</span>'
        st.markdown(source_html, unsafe_allow_html=True)

    st.success("Meal plan generated! Always consult your doctor or dietitian before making dietary changes.")