from qd.qd_query import Query, Workload
from qd.qd_node import Node, Root
from qd.qd_column import Column
from qd.qd_table import Table
from qd.qd_predicate import Operator
from qd.qd_predicate_subclasses import pred_gen, Numerical
import numpy as np
import pickle
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
    return Table(name, num_points)


def workload_gen(root, size, selectivity=0.02, alpha=2, allowed_columns=None):
    """
    Generates a workload for a given root node
    :param root: a root node object
    :param size: number of queries in the workload
    :param selectivity: the expected selectivity of a query in this workload
    :param alpha: the alpha parameter for the beta distribution
    :param allowed_columns: a list of which columns are allowed to be predicated on
    :return: a workload for that root node
    """
    if allowed_columns is None:
        allowed_columns = list(root.table.list_columns())
    rng = np.random.default_rng()
    selectivities = rng.beta(alpha, (alpha-(alpha*selectivity))/selectivity, size=size)
    columns = rng.choice(allowed_columns, size=size)
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


def subset_gen(table, size):
    """
    :param table: a table object
    :param size: the size of the subset
    :return: a list of datapoints representing a subset of the table with size elements
    """
    output = []
    with open("data/" + table.name + ".csv") as file:
        data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        data.__next__()
        i = 0
        for row in data:
            num = np.random.randint(0, i+1)
            if i < size:
                output.append(row)
            elif num < size:
                output[num] = row
            i += 1
        return output


def all_predicates(data, table, columns=None):
    """
    :param data: a list of datapoints
    :param table: the table from where the datapoints came
    :param columns: a list of column names that can be predicated on
    :return: a list of all linear predicates separating this datapoints
    """
    predicates = []
    if columns is None:
        columns = table.list_columns()
    for row in data:
        for i in range(len(columns)):
            column = table.get_column(columns[i])
            predicates.append(Numerical(Operator('<'), column, row[column.num]))
    return predicates


def tree_gen(table, workload, rank_fn, subset_size=60, node=None, root=None, save=False, columns=None):
    """
    :param table: a table object
    :param workload: a workload object
    :param rank_fn: a function that operates on the following parameters:
        param data: a list containing some data
        param table: the same table as used prior
        param workload: a workload object
        param predicate: a predicate on the data
        returns a ranking of how well the predicate splits the operation of the workload on the data.
                better predicates rank higher, and the ranking should be 0 if it cannot be used
    :param subset_size: the size of the data subsets
    :param node: the corresponding node of this function.  used only for recursive calls
    :param root: the root node of the tree being built.  used only for recursive calls
    :param save: whether to save the generated tree
    :param columns: the columns that can be predicated on.  if none, then all columns can be predicated on
    :return: a root object corresponding to the root of a completed qd tree
    """
    if node is None and root is None:
        node = Root(table)
        root = node
    subset = subset_gen(table, subset_size)
    preds = all_predicates(subset, table, columns=columns)
    best_pred = preds[0]
    top_score = 0
    for pred in preds:
        score = rank_fn(subset, table, workload, pred)
        if score > top_score:
            best_pred = pred
            top_score = score
    if top_score > 0:
        child_right, child_left = root.split_leaf(node, best_pred)
        workload_right, workload_left, _ = workload.split(best_pred)
        tree_gen(child_right.table, workload_right, rank_fn, subset_size, child_right, root)
        tree_gen(child_left.table, workload_left, rank_fn, subset_size, child_left, root)
    if save:
        with open('trees/' + table.name + '.pickle', 'w') as file:
            pickle.dump(root, file)
    return root


def rank_fn_gen(min_size, multiply_sizes=False):
    """
    :param min_size: the minimum size of a table in this ranking
    :param multiply_sizes: whether to multiply the number of queries on a side by the number of datapoints on that side
    :return: a function ranking parameters
    """
    def rank_fn(data, table, workload, predicate):
        w_right, w_left, w_both = workload.split(predicate)
        d_right = []
        d_left = []
        for row in data:
            if row in predicate:
                d_right.append(row)
            else:
                d_left.append(row)
        if len(d_right)*table.size/len(data) < min_size:
            return 0
        if len(d_left)*table.size/len(data) < min_size:
            return 0
        if multiply_sizes:
            return len(workload)*len(data) - len(w_right)*len(d_right) - len(w_left)*len(d_left)
        else:
            return len(workload) - len(w_both)

    return rank_fn
