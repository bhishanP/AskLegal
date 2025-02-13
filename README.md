# **PDF Chatbot with FAISS & LLaMA3**
This is an AI-powered chatbot that allows users to **ask questions** about PDF documents. It uses:  
✅ **FAISS** for vector-based document retrieval  
✅ **LLaMA3 (via Groq API)** for generating responses  
✅ **LangChain** for conversational retrieval  
✅ **Streamlit** for the web interface  

---

## **🛠️ Features**
- Upload PDF documents and generate an **embedded knowledge base**  
- Ask questions about the document and get **context-aware answers**  
- Uses **FAISS vector store** for efficient information retrieval  
- Integrates **LLaMA3** for generating responses  
- Built with **LangChain, Hugging Face Embeddings, and Streamlit**  

---

## **📂 Project Structure**
```
📦 AskLegal
│── 📂 backend
│   ├── chatbot.py         # Chatbot logic using LangChain & Groq API
│   ├── pdf_processor.py   # Processes PDFs and creates FAISS vector store
│── 📂 data
│   ├── 📂 pdfs            # Stores uploaded PDFs
│   ├── 📂 vectorstores    # Stores FAISS indexes
│
├── app.py            # Streamlit UI for chatbot
│── .env                  # API keys (ignored in .gitignore)
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation
│── License              # Project License

```

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/bhishanP/AskLegal.git
cd pdf-chatbot
```

### **2️⃣ Create & Activate Virtual Environment**
```bash
# Create a virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
```

---

## **💡 How to Use**
### **1️⃣ Start the Streamlit App**
```bash
streamlit run frontend/app.py
```

### **2️⃣ Upload a PDF & Create Vector Store**
- Go to the **"Create Vector Store"** tab  
- Upload your PDF and process it  

### **3️⃣ Ask Questions**
- Go to the **"Chatbot"** tab  
- Ask any question related to the uploaded PDF  

---

## **⚙️ Troubleshooting**
### 🔴 **Issue: Model ignores vector store and gives generic answers**  
✔️ **Fix:** Make sure the chatbot is using the correct **retriever** (see `chatbot.py`).  
✔️ Use **domain-specific questions** to verify retrieval.  

### 🔴 **Issue: FAISS vector store not found**  
✔️ **Fix:** Ensure the vector store is successfully created in `data/vectorstores/`.  

### 🔴 **Issue: API key error**  
✔️ **Fix:** Check `.env` file and make sure `GROQ_API_KEY` is set.  

---

## **📝 Future Improvements**
✅ **Support for multiple PDFs**  
✅ **Improve UI with more interactivity**  
✅ **Allow switching between different LLMs (GPT, Mistral, etc.)**  

---

## **🤝 Contributing**
If you’d like to contribute, feel free to fork the repo and submit a pull request. 🚀  

---

## **📜 License**
This project is **open-source** under the MIT License.  