from json import load
from qd.qd_algorithms import index as qd_index


def index(query, root_path, table):
    """
    Indexing function for a pqd structure.  Works for all pqd layouts
    :param query:
    :param root_path:
    :param table:
    :return: A list of files accessed by this query
    """
    tree_files = qd_index(query, root_path, table)
    output = set()
    with open('/'.join(root_path.split('/')[:-1]) + 'index.json', 'r') as f:
        file_dict = load(f)
    for obj in tree_files:
        for file in file_dict[obj]:
            output.add(file)
    return list(output)
