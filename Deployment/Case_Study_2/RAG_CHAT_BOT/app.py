from flask import Flask, render_template, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from llama_cpp import Llama
import os

app = Flask(__name__)

# Paths
BASE_PATH = r"Models"
EMBEDDING_PATH = os.path.join(BASE_PATH, "embedding_model")
VECTORSTORE_PATH = os.path.join(BASE_PATH, "vectorstore")
LLM_PATH = os.path.join(BASE_PATH, "mistral-7b-instruct-v0.1.Q3_K_M.gguf")

# Load embeddings & vectorstore
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_PATH)
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Load GGUF model
llm_raw = Llama(
    model_path=LLM_PATH,
    n_ctx=2048,
    n_threads=8,
    n_batch=64,
    use_mlock=True,
    verbose=True
)

llm = RunnableLambda(
    lambda input, **kwargs: llm_raw(str(input), max_tokens=256)["choices"][0]["text"].strip()
)

prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using only the context provided.
Context:
{context}

Question: {question}
Answer:"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["query"]
    try:
        response = qa_chain.invoke({"query": user_input})
        return jsonify({"answer": response["result"]})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
