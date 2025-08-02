import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from typing import List, Optional
from prompts import custom_prompt, structured_query_template

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pregnancy Diet Planner",
    page_icon="🤰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f0f2f6;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
    }
    /* Fix for text input visibility - make text black */
    .stTextInput > div > div > input {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    .stTextArea > div > div > textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    /* Additional selectors for better coverage */
    input[type="text"], textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    /* Fix placeholder text */
    input::placeholder, textarea::placeholder {
        color: #888888 !important;
        font-weight: 400 !important;
    }
    /* Streamlit specific text input styling */
    .stTextInput input, .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    /* Dropdown text styling - make selected text black */
    .stSelectbox > div > div > div > div {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    /* Dropdown options styling */
    .stSelectbox > div > div > div > div > div {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    /* Multi-select styling - original version */
    .stMultiSelect > div > div > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stMultiSelect > div > div > div > div {
        color: #000000 !important;
    }
    /* Red box styling for multi-select tags */
    .stMultiSelect > div > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        #color: #ffffff !important;
        border-radius: 5px !important;
        padding: 0.2rem 0.5rem !important;
        margin: 0.2rem !important;
    }
    /* Styling for the 'x' (remove) button within the tag */
    .stMultiSelect > div > div > div > div > div > button {
        color: #ffffff !important;
    }
    /* Meal plan headings styling */
    h3 {
        font-size: 1.8rem !important;
        color: #1f77b4 !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Fix cursor visibility in text areas */
    .stTextArea textarea {
        caret-color: #000000 !important;
    }
    /* Styling for introduction text */
    .intro-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        border-left: 5px solid #ffd700 !important;
    }
    .intro-text h4 {
        color: #ffffff !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
        text-align: center !important;
    }
    .intro-text p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        margin: 0 !important;
        text-align: center !important;
    }
    /* Increase font size for LLM response content */
    # .meal-plan-content {
    #     font-size: 1.2rem !important;
    #     line-height: 1.6 !important;
    #     color: #ffffff !important;
    # }
#     .meal-plan-content ul, .meal-plan-content ul, .meal-plan-content ol {
#         font-size: 1.2rem !important;
#         line-height: 1.6 !important;
#         margin-bottom: 0.8rem !important;
#     }
#     /* Special styling for meal headings (Breakfast, Lunch, Dinner) */
#     .meal-plan-content h3, .meal-plan-content h4, .meal-plan-content strong {
#         color: #ffd700 !important;
#         font-size: 1.4rem !important;
#         font-weight: 700 !important;
#         margin-top: 1rem !important;
#         margin-bottom: 0.5rem !important;
#         text-transform: uppercase !important;
#         letter-spacing: 1px !important;
#         border-bottom: 2px solid #ffd700 !important;
#         padding-bottom: 0.3rem !important;
#     }
#     /* Fix for breakfast formatting - ensure proper line breaks */
#     .meal-plan-content p {
#         margin-bottom: 0.5rem !important;
#         white-space: pre-line !important;
#     }
# </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤰 Pregnancy Diet Planner</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load FAISS Vector Store
@st.cache_resource
def load_vector_store():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        DB_FAISS_PATH = "vectorstore/dietplanner_db_faiss"
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

# Groq LLM Class
class GroqLLM(LLM):
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.4
    _client: Groq = PrivateAttr()

    def __init__(self, api_key: str, model_name: Optional[str] = None, temperature: float = 0.5):
        super().__init__()
        self._client = Groq(api_key=api_key) 
        self.model = model_name or self.model
        self.temperature = temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=5000,
            top_p=1,
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    class Config:
        arbitrary_types_allowed = True

# Initialize components
@st.cache_resource
def initialize_components():
    try:
        # Load vector store
        db = load_vector_store()
        if db is None:
            return None, None
        
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables")
            return None, None
        
        llm = GroqLLM(api_key=groq_api_key)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt, "document_variable_name": "context"}
        )
        
        return qa_chain, db
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

# Main UI
def main():
    # Initialize components
    qa_chain, db = initialize_components()
    
    if qa_chain is None or db is None:
        st.error("Failed to initialize the application. Please check your configuration.")
        return

    # Sidebar for user inputs
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 User Preferences</h2>', unsafe_allow_html=True)
        
        # Pregnancy month dropdown
        pregnancy_month = st.selectbox(
            "Pregnancy Month",
            options=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            help="Select your current month of pregnancy"
        )
        
        # Diet type dropdown
        diet_type = st.selectbox(
            "Diet Type",
            options=["Vegetarian", "Non-Vegetarian", "Vegan", "Balanced"],
            help="Select your dietary preference"
        )
        
        # Allergies multi-select with smart None handling
        allergies_options = ["None", "Nuts", "Dairy", "Gluten", "Eggs", "Seafood", "Soy", "Shellfish", "Wheat", "Fish"]
        
        # Initialize allergies state
        if 'allergies_selection' not in st.session_state:
            st.session_state.allergies_selection = ["None"]
        
        # Handle allergies selection
        selected_allergies = st.multiselect(
            "Allergies/Intolerances",
            options=allergies_options,
            default=st.session_state.allergies_selection,
            key="allergies_multiselect",
            help="Select any food allergies or intolerances (you can select multiple)"
        )
        
        # Smart None handling for allergies
        if selected_allergies != st.session_state.allergies_selection:
            if "None" in selected_allergies and len(selected_allergies) > 1:
                # If None is selected with other options, keep only None
                selected_allergies = ["None"]
                st.session_state.allergies_selection = ["None"]
                st.rerun()
            elif "None" not in selected_allergies and len(selected_allergies) == 0:
                # If nothing is selected, default to None
                selected_allergies = ["None"]
                st.session_state.allergies_selection = ["None"]
                st.rerun()
            else:
                # Update session state
                st.session_state.allergies_selection = selected_allergies
        
        # Nutrient focus multi-select with smart Balanced handling
        nutrient_options = ["Iron", "Calcium", "Protein", "Folic Acid", "Omega-3", "Vitamin D", "Vitamin B12", "Zinc", "Balanced"]
        
        # Initialize nutrients state
        if 'nutrients_selection' not in st.session_state:
            st.session_state.nutrients_selection = ["Balanced"]
        
        # Handle nutrients selection
        selected_nutrients = st.multiselect(
            "Key Nutrient Focus",
            options=nutrient_options,
            default=st.session_state.nutrients_selection,
            key="nutrients_multiselect",
            help="Select nutrients you want to focus on (you can select multiple)"
        )
        
        # Smart Balanced handling for nutrients
        if selected_nutrients != st.session_state.nutrients_selection:
            if "Balanced" in selected_nutrients and len(selected_nutrients) > 1:
                # If Balanced is selected with other options, keep only Balanced
                selected_nutrients = ["Balanced"]
                st.session_state.nutrients_selection = ["Balanced"]
                st.rerun()
            elif "Balanced" not in selected_nutrients and len(selected_nutrients) == 0:
                # If nothing is selected, default to Balanced
                selected_nutrients = ["Balanced"]
                st.session_state.nutrients_selection = ["Balanced"]
                st.rerun()
            else:
                # Update session state
                st.session_state.nutrients_selection = selected_nutrients
        
        # Cultural preference dropdown
        cultural_preference = st.selectbox(
            "Cultural Preference",
            options=["North Indian", "South Indian", "East Indian", "West Indian", "International", "No Preference"],
            help="Select your cultural food preference"
        )
        
        # Personal preference text input
        preference = st.text_area(
            "Personal Preferences/Goals",
            placeholder="e.g., help in reducing weight, increase energy, manage morning sickness",
            help="Describe any specific preferences or health goals"
        )
        
        # Generate button
        generate_button = st.button("🍽️ Generate Meal Plan", type="primary")

    # Main content area
    if generate_button:
        with st.spinner("Analyzing your requirements and generating meal plan..."):
            try:
                # Prepare user input
                user_input = {
                    "pregnancy_month": pregnancy_month,
                    "diet_type": diet_type,
                    "allergies": ", ".join(selected_allergies) if selected_allergies else "None",
                    "nutrient_focus": ", ".join(selected_nutrients) if selected_nutrients else "Balanced",
                    "cultural_preference": cultural_preference,
                    "preference": preference if preference else "No specific preferences"
                }
                
                # Generate query
                query = structured_query_template.format(**user_input)
                
                # Get response
                response = qa_chain.invoke({"query": query})
                
                # Display results
                st.markdown('<h3 class="sub-header">🍽️ Your Personalized Meal Plan</h3>', unsafe_allow_html=True)
                
                # Display the meal plan
                #st.markdown("### 🍽️ Meal Plan")
                st.markdown(response["result"])
                
                # Success message
                st.success("✅ Meal plan generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating meal plan: {str(e)}")
                st.info("Please check your inputs and try again.")

    # Instructions
    else:
        st.markdown("""
        <div class="intro-text">
            🤰 Welcome to Your Pregnancy Diet Planner
            <p>Eating well during pregnancy is one of the best ways to care for yourself and your baby.<br>
            This bot helps you <strong>create personalized meal plans</strong> based on your stage of pregnancy, preferences, and nutritional needs.</p>
        </div>
        
        ### 📖 How to Use
        
        1. **Fill in your preferences** in the sidebar on the left
        2. **Select your pregnancy month** and dietary requirements
        3. **Choose any allergies** or intolerances you have
        4. **Pick your nutrient focus** for targeted nutrition
        5. **Select cultural preferences** for familiar foods
        6. **Add personal goals** or specific requirements
        7. **Click 'Generate Meal Plan'** to get your personalized diet plan
        
        ### 🎯 Features
        
        - 🤰 **Pregnancy-specific** meal planning
        - 🥗 **Personalized** based on your preferences
        - 🚫 **Allergy-aware** recommendations
        - 🌍 **Cultural** food preferences
        - 📊 **Nutrient-focused** meal plans
        - 📚 **Evidence-based** from medical documents
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 