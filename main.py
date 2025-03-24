import json
from lamoom import Lamoom, Prompt
from typing import List
from json_repair import loads
import dotenv
from classes import *
from utils import discover_and_load_text_files

dotenv.load_dotenv(dotenv.find_dotenv())

from pyvis.network import Network

def visualize_subclusters(subclusters, output_html="cluster_graph.html"):
    """
    Creates an interactive visualization of the subcluster hierarchy using pyvis.
    subclusters: list of top-level Subcluster objects (final clusters).
    output_html: path/filename for the generated HTML visualization.
    """

    # Create a directed graph for clarity (parent -> child)
    net = Network(directed=True, height="750px", width="100%", notebook=False)
    
    net.barnes_hut() 

    def add_subcluster_nodes(sc: Subcluster):
        """
        Recursively add nodes/edges for subcluster sc and its children.
        """

        node_id = sc.subcluster_id

        node_label = sc.subcluster_id

        # Node hover title: detailed info (tags, overview)
        bullet_overview = "<br>".join(sc.combined_overview) if isinstance(sc.combined_overview, list) else sc.combined_overview
        node_title = f"""
        <b>Subcluster Name:</b> {sc.subcluster_id} <br>
        <b>Level from Bottom:</b> {sc.level_from_bottom} <br>
        <b>Files:</b> {", ".join(sc.file_list)} <br>
        <b>Tags:</b> {", ".join(sc.combined_tags)} <br>
        <b>Overview:</b> <br>
        {bullet_overview}
        """

        net.add_node(
            node_id,
            label=node_label,
            title=node_title,
            shape="ellipse",
            color="#87CEFA" if sc.level_from_bottom == 1 else "#ADD8E6"
        )

        # Recursively handle children
        for child in sc.children:
            # First ensure the child is added
            add_subcluster_nodes(child)
            # Then link parent -> child
            net.add_edge(node_id, child.subcluster_id, arrowStrikethrough=False)

    # Add each top-level subcluster (in case there are multiple final clusters)
    for sc in subclusters:
        add_subcluster_nodes(sc)

    # Generate the HTML
    net.show(output_html, notebook=False)
    print(f"[INFO] Visualization saved to {output_html}. Open it in your browser to view.")


def call_llm_api(prompt_id: str, context: dict) -> str:
    lm = Lamoom()
    lm.service.clear_cache()
    result = lm.call(prompt_id, context, "openai/o3-mini")
    return result


def call_llm_api_json(prompt_id: str, context: dict) -> dict:
    raw = call_llm_api(prompt_id, context)
    try:
        return loads(raw.content) 
    except:
        return {}


def generate_overview_and_tags_for_file(doc: FileDocument):
    """
    Calls the LLM to produce:
      - overview (1-2 sentences)
      - tags (3-5 tags, including cause-effect if found)
    """
    prompt = f"""
        You are a specialized text analytics model.

        I will provide the text of a file. Please analyze it and return strictly valid JSON with these keys:

        1) "overview": A short list of bullet points (2–4 points) summarizing the file's key topics or findings. 
        - Each bullet point should be concise, focusing on distinct aspects of the document.

        2) "tags": A small list of keywords/phrases (3–7 max). 
        - Include both topical tags (e.g., "finance", "marketing", "HR") 
        - AND cause-effect tags if applicable (formatted as "cause:..." or "effect:..."). 
            For instance, if the text says "A budget cut caused lead generation to drop," 
            use "cause:budget-cut" and "effect:less-leads".

        File Text:
        \"\"\"{doc.original_text}\"\"\"

        Return valid JSON only of the following format:
        {{
            'overview': generated overview,
            'tags': [list of generated tags],
        }}
        """
    response = call_llm_api_json("generate_overview_and_tags_for_file", {"file_text": doc.original_text})
    doc.overview = response.get("overview", "No overview generated.")
    doc.tags = response.get("tags", [])


def run_summarization_and_tagging(file_docs: List[FileDocument]):
    """
    For each file, call the LLM to fill in 'overview' and 'tags' fields.
    """
    for doc in file_docs:
        generate_overview_and_tags_for_file(doc)


####################################################
# 4) Phase 2: Bottom-Up Subcluster Merging
####################################################

def create_initial_subclusters(file_docs: List[FileDocument]) -> List[Subcluster]:
    """
    Each file doc => one subcluster
    """
    subclusters = []
    for i, doc in enumerate(file_docs, start=1):
        sc_id = f"file_{i:02d}"
        sc = Subcluster(
            subcluster_id=sc_id,
            file_list=[doc.name],
            combined_overview=doc.overview or "No overview",
            combined_tags=doc.tags or [],
            children=[]
        )
        subclusters.append(sc)
    return subclusters


def prompt_llm_for_merges(subclusters: List[Subcluster]) -> dict:
    """
    We expect the LLM to return JSON like:
      {
        "mergeDecisions": [
          { "mergeGroup": ["file_01","file_02"], "reason": "both finance" },
          ...
        ],
        "noMergeClusters": ["file_03"]
      }

    If no merges, it might return an empty mergeDecisions list.
    """
    clusters_info = []
    for sc in subclusters:
        clusters_info.append({
            "id": sc.subcluster_id,
            "level_from_bottom": sc.level_from_bottom,
            "tags": sc.combined_tags,
            "overview": sc.combined_overview
        })

    prompt = f"""
        You are a large language model specialized in bottom-up hierarchical clustering.
        We form clusters from the most specific level (level_from_bottom=1) to more general as levels increase.

        Below is a list of subclusters. Each subcluster has:
        - "id": a unique string
        - "level_from_bottom": an integer indicating how far up we are from the bottom (leaf) level
        - "overview": a short bullet-point list summarizing the subcluster’s key topics/cause-effect
        - "tags": an array of domain tags and possibly "cause:..." or "effect:..." references

        **Your Task**:
        1. Identify which subclusters should be merged because they share strong thematic or cause-effect synergy. 
        - For example, if two subclusters both contain "cause:budget-cut" or have overlapping domain tags (finance, marketing), they might merge.
        - The newly formed parent cluster will have level_from_bottom = 1 + max(children's levels).
        - At higher levels, we accept broader merges, but still emphasize cause/effect or domain alignment.
        2. Return strictly valid JSON with the structure:
        {{
            "mergeDecisions": [
                {{ "mergeGroup": ["subclusterID_1", "subclusterID_2", ...], "reason": "Short bullet points explaining synergy" }}
            ],
            "noMergeClusters": [ "subclusterID_X", "subclusterID_Y", ... ]
        }}

        3. Keep merges **incremental** (2-3 subclusters at most) unless there is a clear reason for larger merges.
        4. If no merges make sense, respond with an empty "mergeDecisions" array and optionally fill "noMergeClusters."

        ### Subclusters:
        {json.dumps(clusters_info, indent=2)}

        ### Important:
        - The "reason" in each merge decision can be bullet points or a short explanation.
        - Return **only** the JSON. No extra text or commentary.
        """
    response = call_llm_api_json("merge_clusters", {"subclusters": json.dumps(clusters_info, indent=2)})
    
    return response

def generate_cluster_name(merged_subclusters: List[Subcluster]) -> str:
    """
    Calls the LLM to produce a short, meaningful name for the newly formed cluster.
    We supply the children's overviews/tags so the LLM can infer a concise topic name.
    """
    child_info = []
    for sc in merged_subclusters:
        child_info.append({
            "id": sc.subcluster_id,
            "level_from_bottom": sc.level_from_bottom,
            "tags": list(sc.combined_tags),
            "overview": sc.combined_overview
        })

    prompt = f"""
        We are merging subclusters to form a new parent with level_from_bottom = 1 + max(child levels).
        Each child's data:
        {json.dumps(child_info, indent=2)}

        Given these children, provide a short cluster name (1-4 words) 
        that is broader than any single child's name if needed, 
        reflecting the combined domain or cause/effect patterns at a higher level of abstraction.

        Return JSON: {{ "clusterName": "..." }}
        """

    resp = call_llm_api_json("generate_cluster_name", {'childs_info': json.dumps(child_info, indent=2)})
    return resp.get("clusterName", "GenericCluster")


def generate_combined_overview(subclusters: List[Subcluster]) -> str:
    """
    Calls the LLM to produce a unified overview from multiple child subclusters.
    """
    child_info = []
    for sc in subclusters:
        child_info.append({
            "id": sc.subcluster_id,
            "tags": list(sc.combined_tags),
            "overview": sc.combined_overview
        })
    prompt = f"""
        You are creating a parent cluster at level_from_bottom = X,
        merging the following child subclusters (which are at lower levels).

        Each child has:
        - "id": unique identifier
        - "level_from_bottom": its current level (which is lower than the parent's)
        - "overview": a bullet-point list of topics or cause/effect details
        - "tags": domain or cause/effect tags

        Please produce a new "combinedOverview" in bullet-point form (2–5 points), focusing on:
        1) Key themes or domain overlaps (finance, hr, marketing, etc.)
        2) Cause/effect synergy (e.g., "cause:budget-cut" => "effect:reduced leads") if relevant
        3) Slightly broader perspective if the parent's level is higher. For example, if children mention Q1 finance, you can unify it as "financial performance and budgets," etc.

        Return strictly valid JSON with the format:
        {{
            "combinedOverview": ["bullet point 1", "bullet point 2", ...]
        }}

        ### Child Subclusters:
        {json.dumps(child_info, indent=2)}

        Return **only** the JSON object with "combinedOverview" as an array of bullet points. 
        """

    resp = call_llm_api_json("generate_combined_overview", {'childs_info': json.dumps(child_info, indent=2)})
    return resp.get("combinedOverview", "(merged overview not generated)")


def create_new_subcluster(merge_group_ids: List[str], all_subclusters: List[Subcluster]) -> Subcluster:
    """
    Merge the subclusters in merge_group_ids into one new parent subcluster.
    """
    merged_children = [sc for sc in all_subclusters if sc.subcluster_id in merge_group_ids]
    
    merged_files = []
    merged_tags = []
    for sc in merged_children:
        merged_files.extend(sc.file_list)
        merged_tags.extend(sc.combined_tags)

    combined_ov = generate_combined_overview(merged_children)
    cluster_name = generate_cluster_name(merged_children)
    

    new_subc_id = cluster_name
    new_level = max(child.level_from_bottom for child in merged_children) + 1

    new_subcluster = Subcluster(
        subcluster_id=new_subc_id,
        file_list=merged_files,
        combined_overview=combined_ov,
        combined_tags=merged_tags,
        children=merged_children,
        level_from_bottom=new_level
    )

    return new_subcluster


def merge_iteration(subclusters: List[Subcluster]) -> List[Subcluster]:
    """
    One pass: ask the LLM for merges, create new parent subclusters, 
    remove old children from the active list.
    """
    if len(subclusters) <= 1:
        return subclusters  # nothing to merge

    merges_info = prompt_llm_for_merges(subclusters)
    merge_decisions = merges_info.get("mergeDecisions", [])
    no_merge = merges_info.get("noMergeClusters", [])

    if not merge_decisions:
        # no merges recommended
        return subclusters

    new_subclusters = []
    merged_ids = []

    # create new subclusters for each merge
    for md in merge_decisions:
        group = md["mergeGroup"]
        new_subc = create_new_subcluster(group, subclusters)
        new_subclusters.append(new_subc)
        merged_ids.extend(group)

    # keep subclusters that are not merged
    for sc in subclusters:
        if sc.subcluster_id not in merged_ids and sc.subcluster_id not in no_merge:
            new_subclusters.append(sc)
        elif sc.subcluster_id in no_merge:
            new_subclusters.append(sc)

    return new_subclusters


def print_subcluster_tree(root: Subcluster, indent=0):
    prefix = "  " * indent
    print(f"{prefix}- Subcluster ID: {root.subcluster_id}")
    print(f"{prefix}  Files: {root.file_list}")
    print(f"{prefix}  Tags: {root.combined_tags}")
    print(f"{prefix}  Overview: {root.combined_overview}")
    if root.children:
        print(f"{prefix}  Children:")
        for ch in root.children:
            print_subcluster_tree(ch, indent+1)


if __name__ == "__main__":
    
    # 1) Extract files content
    print("[INFO] Discovering text-based files in the current project...")
    docs = discover_and_load_text_files(root_path="./vanila-react")

    # 2) Summarize & Tag each doc using the LLM
    print("=== Generating Overviews & Tags ===")
    run_summarization_and_tagging(docs)
    for d in docs:
        print(f"File: {d.name}")
        print(f"  Overview: {d.overview}")
        print(f"  Tags: {d.tags}")

    # 3) Convert them into initial subclusters
    subclusters = create_initial_subclusters(docs)

    # 4) Perform iterative merges until stable
    round_count = 0
    while True:
        round_count += 1
        print(f"\n--- Merge Iteration #{round_count} ---")
        new_subclusters = merge_iteration(subclusters)
        if len(new_subclusters) == len(subclusters):
            print("No more merges recommended. Stopping.")
            break
        subclusters = new_subclusters
        if len(subclusters) <= 1:
            print("Reached 1 or 0 subclusters. Stopping.")
            break

    # 5) Print final subcluster tree
    print("\n=== Final Clusters (Tree) ===")
    for sc in subclusters:
        print_subcluster_tree(sc)
        print("----------------------------")
    
    # 6) Generate an html cluster graph
    visualize_subclusters(subclusters, output_html="cluster_graph.html")