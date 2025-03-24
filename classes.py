from typing import List, Optional

class FileDocument:
    """
    Each file: 
      - name
      - original_text (the entire file text or large chunk)
      - overview
      - tags
    We'll fill 'overview' and 'tags' by calling the LLM to analyze the file text.
    'tags' will include cause/effect markers if discovered.
    """
    def __init__(self, name: str, original_text: str):
        self.name = name
        self.original_text = original_text
        self.overview: Optional[str] = None
        self.tags: List[str] = []

    def __repr__(self):
        return f"FileDocument(name='{self.name}', tags={self.tags})"


class Subcluster:
    """
    Each subcluster can represent:
      - A single file initially
      - A merged cluster of multiple files
    Contains:
      - subcluster_id
      - file_list
      - combined_overview
      - combined_tags
      - children subclusters (if it’s a merge)
    """
    def __init__(
        self,
        subcluster_id: str,
        file_list: List[str],
        combined_overview: List[str],
        combined_tags: List[str],
        children: Optional[List['Subcluster']] = None,
        level_from_bottom: int = 1
    ):
        self.subcluster_id = subcluster_id
        self.file_list = file_list
        self.combined_overview = combined_overview
        self.combined_tags = list(set(combined_tags))
        self.children: List[Subcluster] = children if children else []
        self.level_from_bottom = level_from_bottom

    def __repr__(self):
        return (f"Subcluster(id={self.subcluster_id}, "
                f"files={self.file_list}, "
                f"tags={self.combined_tags})")