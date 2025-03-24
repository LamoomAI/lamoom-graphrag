import os
from classes import List, FileDocument

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".log", ".cfg", ".json", ".yaml", ".yml", ".csv", ".ini"
}

import os
from pathspec import PathSpec

def load_gitignore_patterns(root_path=".") -> PathSpec:
    """
    Reads .gitignore from root_path (if present) and returns 
    a PathSpec object containing all ignore patterns.
    If .gitignore doesn't exist, returns an empty PathSpec.
    """
    gitignore_path = os.path.join(root_path, ".gitignore")
    if not os.path.isfile(gitignore_path):
        # No .gitignore found, return an empty spec
        return PathSpec.from_lines("gitwildmatch", [])
    
    with open(gitignore_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    spec = PathSpec.from_lines("gitwildmatch", lines)
    return spec


def discover_and_load_text_files(root_path=".") -> List[FileDocument]:
    """
    Recursively discover text-friendly files in 'root_path', skipping anything 
    matched by .gitignore. Creates FileDocument objects for each valid file.
    """
    # 1) Load .gitignore patterns
    spec = load_gitignore_patterns(root_path)

    all_documents = []

    # 2) Walk the directory structure
    for dirpath, dirnames, filenames in os.walk(root_path):
        # First remove any subdirs from 'dirnames' that are ignored by .gitignore
        # We do this in-place so os.walk won't even recurse into them
        subdirs_to_remove = []
        for d in dirnames:
            rel_subdir_path = os.path.relpath(os.path.join(dirpath, d), root_path)
            if spec.match_file(rel_subdir_path):
                subdirs_to_remove.append(d)
        for d in subdirs_to_remove:
            dirnames.remove(d)

        # Now handle files
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()
            rel_file_path = os.path.relpath(os.path.join(dirpath, filename), root_path)

            # # Skip if extension is not allowed
            # if file_ext not in ALLOWED_EXTENSIONS:
            #     continue

            # Skip if .gitignore says so
            if spec.match_file(rel_file_path):
                continue

            full_path = os.path.join(dirpath, filename)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_content = f.read()
                doc = FileDocument(name=rel_file_path, original_text=file_content)
                all_documents.append(doc)
            except Exception as e:
                print(f"[WARN] Could not read file: {full_path}. Error: {e}")

    return all_documents