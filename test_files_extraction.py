from utils import discover_and_load_text_files

print("[INFO] Discovering text-based files in the current project...")
docs = discover_and_load_text_files(root_path="./vanila-react")

for doc in docs:
    print(f"DOC NAME: {doc.name}")