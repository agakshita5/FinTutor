# 💰 FinTutor - Finance AI Assistant

An intelligent financial chatbot that answers your personal finance questions using AI. Ask anything about budgeting, investments, taxes, loans, and financial planning. Get instant, personalized guidance powered by Google Gemini AI and a comprehensive knowledge base of 40K+ financial FAQs.

**[View Live Demo](https://agakshita-fintutor.hf.space)** 

## 📝 What It Does

- **Ask Finance Questions**: Get instant answers about budgeting, investments, taxes, loans, retirement
- **Smart Categorization**: Automatically categorizes your expenses (Food, Transport, Shopping, etc.)
- **Chat History**: Save and revisit past conversations
- **Voice Chat**: Speak your questions using voice input
- **Personalized Guidance**: AI learns from context to provide relevant financial advice

## ✨ Features

- AI-powered responses via Gemini 2.5 Flash
- RAG system with ChromaDB for accurate, context-based answers
- User login with persistent chat history
- Voice input support (Web Speech API)
- Auto-transaction categorization
- Response caching for faster replies
- Session tracking and analytics

## 🛠️ Tech Stack

- **Backend**: Flask, Google Gemini 2.5 Flash API, ChromaDB, Sentence Transformers
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI/ML**: Semantic search with embeddings, RAG (Retrieval-Augmented Generation)
- **Data**: 40K+ verified financial FAQ entries ([Google Drive](https://drive.google.com/file/d/192D2OVvOeOr-KXUEAuKCqVYOSdJ0Ue6t/view?usp=sharing))
- **Auth**: Simple login/signup with JSON storage (easily upgradeable to database)

## 🚀 Quick Start

```bash
# Clone & install
git clone https://github.com/agakshita5/FinTutor.git
cd FinTutor
pip install -r requirements.txt

# Create .env file
echo "AI_API_KEY=your_gemini_key_here" > .env

# Run
python app.py
# Open http://localhost:5000
```
