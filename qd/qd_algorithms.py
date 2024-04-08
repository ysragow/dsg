from qd_query import Query, Workload
from qd_node import Node, Root
from qd_table import Table
from qd_predicate_subclasses import pred_gen
import numpy as np
import csv
import os


def dataset_gen(name, num_columns=10, num_points=100000, max_value=9999):
    """
    Generate a uniformly distributed dataset of integers
    :param name: the name of the new dataset
    :param num_columns: number of columns in the dataset
    :param num_points: number of points in the dataset
    :param max_value: maximum value of the dataset
    :return: a table object for the dataset
    """
    path = 'data/' + name + '.csv'
    assert not os.path.isfile(path), "This file already exists"
    assert name[0] not in ('1','0'), "Filename cannot begin with a 1 or a 0"
    with open(path, 'w') as file:
        data = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        data.writerow(list(['col' + str(i) for i in range(num_columns)]))
        for i in range(num_points):
            data.writerow(np.random.randint(max_value, size=num_columns))
    return Table(name)


def workload_gen(root, size, selectivity=0.02, alpha=2):
    """
    Generates a workload for a given root node
    :param root: a root node object
    :param size: number of queries in the workload
    :param selectivity: the expected selectivity of a query in this workload
    :param alpha: the alpha parameter for the beta distribution
    :return: a workload for that root node
    """
    rng = np.random.default_rng()
    selectivities = rng.beta(alpha, (alpha-(alpha*selectivity))/selectivity, size=size)
    columns = rng.choice(list(root.table.list_columns()), size=size)
    tops = np.array([root.preds[col][1].num for col in columns])
    bottoms = np.array([root.preds[col][0].num for col in columns])
    ranges = (tops - bottoms)*selectivities
    centers = rng.integers(bottoms, tops, endpoint=True)
    highs = centers + ranges/2
    lows = centers - ranges/2
    queries = []
    for i in range(size):
        queries.append(Query(
            [pred_gen(columns[i]+' >= '+str(lows[i]), root.table),
             pred_gen(columns[i]+' <= '+str(highs[i]), root.table),
             ],
            root.table))
    return Workload(queries)




    