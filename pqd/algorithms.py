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
    # The index should be in the parent directory
    split_path = root_path.split('/')
    lower_path = '/'.join(split_path[:-2]) + '/' + split_path[-1]
    tree_files = qd_index(query, lower_path, table)
    output = set()
    with open('/'.join(split_path[:-1]) + '/index.json', 'r') as f:
        file_dict = load(f)
    for obj in tree_files:
        for file in file_dict[obj]:
            output.add(file)
    return list(output)
