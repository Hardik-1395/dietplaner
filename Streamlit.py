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
    font-size: 1.1rem;
    color: #4a2b1a !important;
    font-weight: 500 !important;
}

/* Section headers */
.section-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #8b3f1f !important;
    font-weight: 600;
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

/* 🌟 LIGHT YELLOW LABEL BADGES (FIXED VISIBILITY) */
[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stTextInput label p,
.stTextArea label p {

    background-color: #fff7cc !important;   /* light yellow */
    color: #5c4b00 !important;              /* readable dark yellow-brown */

    font-weight: 600 !important;
    font-size: 0.95rem !important;

    padding: 6px 12px !important;
    border-radius: 10px !important;

    display: inline-block !important;
    margin-bottom: 8px !important;

    border: 1px solid #f4e08a !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
/* Dropdown selected value */
div[data-baseweb="select"] span {
    color: #1f140d !important;
    font-weight: 500 !important;
}

/* Dropdown input area */
div[data-baseweb="select"] > div {
    background-color: #fffaf6 !important;
    border: 1px solid #d6a48a !important;
    border-radius: 10px !important;
}

/* Dropdown menu items */
ul[role="listbox"] li {
    color: #2d1a0e !important;
}
ul[role="listbox"] li:hover {
    background-color: #fde8dc !important;
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
.source-title {
    margin-top: 1.5rem;
    font-weight: 600;
    color: #6b3a20;
    font-size: 1.05rem;
}
.source-tag {
    display: inline-block;
    background: var(--sage-light);
    color: #3a6b3e;
    border-radius: 50px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    margin: 0.2rem;
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

# ─── Load resources (cached) ────────────────────────────────────────────────
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

# ─── Prompts ────────────────────────────────────────────────────────────────
CUSTOM_PROMPT_TEMPLATE = """
Using ONLY the context provided, generate a structured meal plan covering Breakfast, Lunch, and Dinner that meets the user's requirements exactly.

Rules:
- Include at least 5 food items per meal.
- Also include total macros for each meal:
  - Calories
  - Protein (g)
  - Carbohydrates (g)
  - Fats (g)
- Do NOT add extra commentary outside the meal plan.
- If exact macro values are uncertain, estimate them using standard serving sizes.

--- 
📘 CONTEXT:
{context}
---

🤰 USER QUERY:
{question}

💡 YOUR RESPONSE FORMAT:

Breakfast:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Breakfast Macros:
- Calories: ___ kcal
- Protein: ___ g
- Carbohydrates: ___ g
- Fats: ___ g

Lunch:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Lunch Macros:
- Calories: ___ kcal
- Protein: ___ g
- Carbohydrates: ___ g
- Fats: ___ g

Dinner:
- Food item 1: portion
- Food item 2: portion
- Food item 3: portion
- Food item 4: portion
- Food item 5: portion

Dinner Macros:
- Calories: ___ kcal
- Protein: ___ g
- Carbohydrates: ___ g
- Fats: ___ g

Safety Notes:
- Mention 2–4 short precautions such as foods to avoid or hygiene practices.
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
- Medical conditions: {medical_conditions}
"""

# ─── Groq caller ────────────────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1800,
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
        st.markdown("""
        <div class="source-title">📚 Sources used from knowledge base:</div>
        """, unsafe_allow_html=True)

        source_html = ""
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                short = os.path.basename(src) if "/" in src or "\\" in src else src
                source_html += f'<span class="source-tag">📄 {short}</span>'
        st.markdown(source_html, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:#fff7cc;
        border:1px solid #f4e08a;
        padding:12px 16px;
        border-radius:10px;
        color:#6b5a00;
        font-weight:500;
        margin-top:14px;">
        ⚠️ Meal plan generated! Always consult your doctor or dietitian before making dietary changes.
    </div>
    """, unsafe_allow_html=True)