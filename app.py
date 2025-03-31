import os
import git
import json
import subprocess
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

# Constants
REPO_URL = "https://github.com/Taskiee/Project-2---23f1001906"
LOCAL_PATH = "./repo"
FOLDERS = ["GA1", "GA2"]
EMBEDDINGS_FILE = "embeddings.json"

# Clone the repository only once
if not os.path.exists(LOCAL_PATH):
    print("Cloning repository...")
    git.Repo.clone_from(REPO_URL, LOCAL_PATH)
else:
    print("Repository already exists.")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_data():
    data = {}
    for folder in FOLDERS:
        folder_path = os.path.join(LOCAL_PATH, folder)
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".txt"):  # Extracting questions
                        q_path = os.path.join(root, file)
                        with open(q_path, "r", encoding="utf-8") as f:
                            question = f.read().strip()
                        
                        # Find corresponding solution file
                        base_name = file.replace(".txt", "")
                        solution_file = None
                        for ext in [".py", ".sh"]:
                            if base_name + ext in files:
                                solution_file = os.path.join(root, base_name + ext)
                                break
                        
                        # Store data
                        data[question] = {
                            "solution_file": solution_file
                        }
    return data

def generate_embeddings(data):
    embeddings = {}
    for question in data:
        vector = embedding_model.encode(question).tolist()
        embeddings[question] = {
            "embedding": vector,
            "solution_file": data[question]["solution_file"]
        }
    return embeddings

def execute_script(file_path):
    try:
        if file_path.endswith(".py"):
            result = subprocess.run(["python3", file_path], capture_output=True, text=True)
        elif file_path.endswith(".sh"):
            result = subprocess.run(["bash", file_path], capture_output=True, text=True)
        else:
            return "Unsupported file type"
        return result.stdout.strip()
    except Exception as e:
        return str(e)

# API Server
app = Flask(__name__)

@app.route("/api/", methods=["POST"])
def handle_request():
    question = request.form.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    # Load embeddings or regenerate if missing
    if not os.path.exists(EMBEDDINGS_FILE):
        extracted_data = extract_data()
        embeddings = generate_embeddings(extracted_data)
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f, indent=4)
    else:
        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)
    
    # Find closest question
    input_embedding = embedding_model.encode(question).tolist()
    best_match = None
    best_similarity = -1
    
    for stored_question, data in embeddings.items():
        similarity = sum(a * b for a, b in zip(input_embedding, data["embedding"]))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = stored_question
    
    if best_match and embeddings[best_match]["solution_file"]:
        solution_file = embeddings[best_match]["solution_file"]
        answer = execute_script(solution_file)
    else:
        answer = "No matching solution found."
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Ensure Flask runs on the correct port for Railway
    port = int(os.environ.get("PORT", 5000))
    
    # Generate embeddings at startup if missing
    if not os.path.exists(EMBEDDINGS_FILE):
        extracted_data = extract_data()
        embeddings = generate_embeddings(extracted_data)
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f, indent=4)
    
    print("Embeddings saved to", EMBEDDINGS_FILE)
    app.run(host="0.0.0.0", port=port)
