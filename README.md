# 🤰 Pregnancy Diet Planner

A Streamlit-based web application that generates personalized meal plans for pregnant women using AI and medical documents.

## 🚀 Features

- **Personalized Meal Planning**: Based on pregnancy month, diet preferences, and health goals
- **Allergy-Aware**: Considers food allergies and intolerances
- **Cultural Preferences**: Supports various Indian and international cuisines
- **Nutrient-Focused**: Targets specific nutrients like iron, calcium, protein
- **Evidence-Based**: Uses medical documents and dietary guidelines
- **User-Friendly Interface**: Clean Streamlit UI with dropdown selections

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Groq API Key** - Get from [Groq Console](https://console.groq.com/)
3. **HuggingFace Token** - Get from [HuggingFace](https://huggingface.co/settings/tokens)

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Run the backend to create vector store**:
   ```bash
   python backend.py
   ```

## 🚀 Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

## 📖 How to Use

1. **Fill in your preferences** in the sidebar:
   - Pregnancy month (1-9)
   - Diet type (Vegetarian/Non-Vegetarian/Vegan/Balanced)
   - Allergies/Intolerances
   - Nutrient focus (Iron/Calcium/Protein/etc.)
   - Cultural preference
   - Personal goals

2. **Click "Generate Meal Plan"** to get your personalized diet plan

3. **View the results**:
   - Personalized meal plan with portions
   - Source documents used
   - Safety precautions

## 📁 Project Structure

```
diet_planner2/
├── streamlit_app.py      # Main Streamlit application
├── backend.py            # Vector store creation
├── grok.py              # Groq LLM integration
├── prompts.py            # Prompt templates
├── main.py              # Original CLI version
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── data/
│   └── diet_docs/      # PDF documents
└── vectorstore/         # FAISS vector database
```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key
- `HF_TOKEN`: Your HuggingFace token

### Model Configuration
- **LLM**: Groq with Llama-4-Scout-17B model
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for document retrieval

## 🎯 Features in Detail

### Input Fields
- **Pregnancy Month**: 1-9 months
- **Diet Type**: Vegetarian, Non-Vegetarian, Vegan, Balanced
- **Allergies**: None, Nuts, Dairy, Gluten, Eggs, Seafood, Multiple
- **Nutrient Focus**: Iron, Calcium, Protein, Folic Acid, Omega-3, Vitamin D, Balanced
- **Cultural Preference**: North/South/East/West Indian, International, No Preference
- **Personal Goals**: Text input for specific requirements

### Output Format
- **Breakfast**: 5 food items with portions
- **Lunch**: 5 food items with portions  
- **Dinner**: 5 food items with portions
- **Safety Precautions**: Foods to avoid and hygiene practices

## 🐛 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY not found"**
   - Check your `.env` file has the correct API key
   - Restart the Streamlit app after adding the key

2. **"Vector store not found"**
   - Run `python backend.py` first to create the vector store
   - Check if `vectorstore/dietplanner_db_faiss` directory exists

3. **"Error loading components"**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check internet connection for model downloads

4. **"No relevant documents found"**
   - Ensure PDF documents are in `data/diet_docs/`
   - Re-run `backend.py` to rebuild the vector store

### Performance Tips
- First run may be slow due to model downloads
- Subsequent runs will be faster due to caching
- Use specific nutrient focus for better results

## 📚 Technical Details

### AI Components
- **Groq LLM**: Fast inference for meal plan generation
- **FAISS Vector Store**: Efficient document retrieval
- **Sentence Transformers**: Document embedding
- **LangChain**: Chain orchestration

### Data Sources
- WHO dietary guidelines
- Indian nutrition guidelines
- Pregnancy-specific nutrition documents
- Regional dietary recommendations

## 🤝 Contributing

Feel free to contribute by:
- Adding more dietary guidelines
- Improving the UI/UX
- Adding new features
- Reporting bugs

## 📄 License

This project is for educational and research purposes.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your API keys are correct
3. Ensure all dependencies are installed
4. Check the console for detailed error messages 