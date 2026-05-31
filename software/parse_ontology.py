import json
import os

def parse_audioset_ontology(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    # 1. Map ID to display name and children
    id_to_name = {item['id']: item['name'] for item in data}
    child_to_parents = {} # To find siblings
    all_nodes = {item['id']: item for item in data}
    
    # 2. Identify all names for Phase 1
    all_names = [item['name'] for item in data]
    
    # 3. Identify Leaves (nodes with no children)
    all_parents = set()
    for item in data:
        for child_id in item.get('child_ids', []):
            all_parents.add(item['id'])
            if child_id not in child_to_parents:
                child_to_parents[child_id] = []
            child_to_parents[child_id].append(item['id'])
            
    leaves = [item for item in data if item['id'] not in all_parents]
    
    # 4. Group sibling leaves
    # Siblings are leaves that share at least one parent
    parent_to_leaves = {}
    for leaf in leaves:
        parents = child_to_parents.get(leaf['id'], [])
        for p in parents:
            if p not in parent_to_leaves:
                parent_to_leaves[p] = []
            parent_to_leaves[p].append(leaf['name'])
            
    # Filter groups with at least 2 leaves for interpolation
    interpolation_groups = {id_to_name[p]: names for p, names in parent_to_leaves.items() if len(names) >= 2}
    
    return all_names, interpolation_groups

if __name__ == "__main__":
    ontology_path = "software/data/ontology.json"
    if os.path.exists(ontology_path):
        all_labels, groups = parse_audioset_ontology(ontology_path)
        print(f"Total Labels: {len(all_labels)}")
        print(f"Interpolation Groups: {len(groups)}")
        # Print a few examples
        example_parent = list(groups.keys())[0]
        print(f"Example Group '{example_parent}': {groups[example_parent]}")
    else:
        print("Ontology file not found.")
