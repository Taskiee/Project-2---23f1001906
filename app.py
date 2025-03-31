import os
import json
import subprocess
import git
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis import Redis
from sentence_transformers import SentenceTransformer

# Constants
REPO_URL = "https://github.com/Taskiee/Project-2---23f1001906"
LOCAL_PATH = "./repo"
FOLDERS = ["GA1", "GA2"]
EMBEDDINGS_FILE = "embeddings.json"

# Clone repo only if it doesn't exist
if not os.path.exists(LOCAL_PATH):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, LOCAL_PATH], check=True)
else:
    print("Repository already exists.")

# Load a lighter embedding model (uses less memory)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# Function to extract question-answer pairs
def extract_data():
    data = {}
    for folder in FOLDERS:
        folder_path = os.path.join(LOCAL_PATH, folder)
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".txt"):
                        q_path = os.path.join(root, file)
                        with open(q_path, "r", encoding="utf-8") as f:
                            question = f.read().strip()
                        
                        base_name = file.replace(".txt", "")
                        solution_file = None
                        for ext in [".py", ".sh"]:
                            if base_name + ext in files:
                                solution_file = os.path.join(root, base_name + ext)
                                break
                        
                        data[question] = {"solution_file": solution_file}
    return data

# Generate embeddings and store in a file
def generate_embeddings(data):
    embeddings = {}
    for question in data:
        vector = embedding_model.encode(question).tolist()
        embeddings[question] = {
            "embedding": vector,
            "solution_file": data[question]["solution_file"]
        }
    return embeddings

# Execute a script file (.py or .sh)
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

# Flask API
app = Flask(__name__)

# Set up Redis connection for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = Redis.from_url(redis_url, decode_responses=True)

# Rate limiter using Redis backend
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri=redis_url,
    default_limits=["5 per minute"]
)

@app.route("/api/", methods=["POST"])
@limiter.limit("5 per minute")
def handle_request():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    # Load embeddings
    with open(EMBEDDINGS_FILE, "r") as f:
        embeddings = json.load(f)

    # Find closest matching question
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

# Initialize embeddings and start Flask server
if __name__ == "__main__":
    extracted_data = extract_data()
    embeddings = generate_embeddings(extracted_data)
    
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f, indent=4)
    
    print("Embeddings saved to", EMBEDDINGS_FILE)

    # Use dynamic port for Railway
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
