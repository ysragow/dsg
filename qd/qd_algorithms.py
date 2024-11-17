from qd.qd_query import Query, Workload
from qd.qd_node import Node, Root
from qd.qd_column import Column
from qd.qd_table import Table
from qd.qd_predicate import Operator
from qd.qd_predicate_subclasses import pred_gen, Numerical, NumComparative, Categorical, intersect
from fastparquet import ParquetFile
import numpy as np
from json import dump, load, loads
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
    if table.storage == 'csv':
        output = []
        with open(table.path) as file:
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
    elif table.storage == 'parquet':
        return ParquetFile(table.path).to_pandas().sample(size)


def all_predicates(data, table, columns=None):
    """
    :param data: a list of datapoints
    :param table: the table from where the datapoints came
    :param columns: a list of column names that can be predicated on
    :return: a list of all linear predicates separating this datapoints
    """
    predicates = []
    columns = table.list_columns()
    if type(data) == list:
        # the data is a list
        for row in data:
            for i in range(len(columns)):
                column = table.get_column(columns[i])
                predicates.append(Numerical(Operator('<'), column, row[column.num]))
    else:
        # the data is a pandas dataframe
        num_columns = []
        date_columns = []
        cat_columns = []
        for c_ in columns:
            c = table.columns[c_]
            if c.numerical:
                if c.ctype == 'DATE':
                    date_columns.append(c)
                else:
                    num_columns.append(c)
            else:
                cat_columns.append(c)

        # Every predicate comparing numbers
        for c_1 in num_columns:
            for c_2 in num_columns:
                if c_1.name != c_2.name:
                    predicates.append(NumComparative(Operator('<'), c_1, c_2))
                    predicates.append(NumComparative(Operator('>'), c_1, c_2))
            # Every numerical predicate on real numbers
            for num in data[c_1.name]:
                predicates.append(Numerical(Operator('<'), c_1, num))

        # Every predicate comparing dates
        for c_1 in date_columns:
            for c_2 in date_columns:
                if c_1.name != c_2.name:
                    predicates.append(NumComparative(Operator('<'), c_1, c_2))
            # Every numerical predicate on dates
            for num in data[c_1.name]:
                predicates.append(Numerical(Operator('<'), c_1, num))

        # Every categorical predicate (with set size of 1)
        for c in cat_columns:
            items = set()
            for item in data[c.name]:
                items.add(str(item))
            for item in items:
                predicates.append(Categorical(Operator('IN'), c, {item}))

    return predicates


def tree_gen(table, workload, rank_fn=None, subset_size=60, node=None, root=None, prev_preds=None, columns=None, block_size=1000):
    """
    :param table: a table object
    :param workload: a workload object
    :param rank_fn: a function that operates on the following parameters: (only for recursive calls)
        param data: a list containing some data
        param table: the same table as used prior
        param workload: a workload object
        param predicate: a predicate on the data
        param prev_preds: previous preds used to split the data up to this point
        returns a ranking of how well the predicate splits the operation of the workload on the data.
                better predicates rank higher, and the ranking should be 0 if it cannot be used
    :param subset_size: the size of the data subsets
    :param node: the corresponding node of this function.  used only for recursive calls
    :param root: the root node of the tree being built.  used only for recursive calls
    :param prev_preds: previous predicates leading up to this one.  used only for recursive calls
    :param columns: the columns that can be predicated on.  if none, then all columns can be predicated on
    :param block_size: the minimum size of a block.  The maximum size is twice the block size.
    :return: a tree as a nested list
    """
    top = False
    output = []
    if node is None and root is None:
        node = Root(table)
        root = node
        prev_preds = []
        top = True
        rank_fn = rank_fn_gen(block_size)
    print("Generating subset...", end='\r')
    valid_splits = False
    while not valid_splits:
        top_score = 0
        if table.size > 2 * block_size:
            subset = subset_gen(table, subset_size)
            print("Generating preds...", end='\r')
            preds = all_predicates(subset, table, columns=columns)
            best_pred = preds[0]
            print("Testing preds...", end='\r')
            for pred in preds:
                # if len(str(pred)) > 100:
                #     print("acting on pred:", str(pred)[:100] + '                                 ', end='\r')
                # else:
                #     print("acting on pred:", pred, '                                 ', end='\r')
                score = rank_fn(subset, table, workload, pred, prev_preds)
                if score > top_score:
                    best_pred = pred
                    top_score = score
        if top_score > 0:
            print('Choosing the following predicate:', best_pred)
            print('Splitting {} into {} and {}...'.format(table.name, table.name + '0', table.name + '1'), end='\r')
            child_right, child_left = root.split_leaf(node, best_pred)
            workload_right, workload_left, _ = workload.split(best_pred, prev_preds)
            dict_right = tree_gen(child_right.table, workload_right, rank_fn, subset_size, child_right, root,
                                  prev_preds + [best_pred], block_size=block_size)
            dict_left = tree_gen(child_left.table, workload_left, rank_fn, subset_size, child_left, root,
                                 prev_preds + [best_pred.flip()], block_size=block_size)
            output.append(str(best_pred))
            output += [dict_right, dict_left]
            valid_splits = True
        elif table.size > (2 * block_size):
            print("Failed to split {} rows; minimum block size is {}  Trying again.".format(table.size, block_size))
        else:
            valid_splits = True
            print("Leaving {} with {} rows".format(table.path, table.size))
    if top:
        print(output)
        with open(table.folder + '/' + table.name + '.json', 'w') as file:
            dump(output, file)
    return output
    # else:
    #     return output
    # if save:
    #     with open('.'.join(table.path.split('.')[:-1]) + 'pickle', 'w') as file:
    #         pickle.dump(root, file)


def rank_fn_gen(min_size, multiply_sizes=False):
    """
    :param min_size: the minimum size of a table in this ranking
    :param multiply_sizes: whether to multiply the number of queries on a side by the number of datapoints on that side
    :return: a function ranking parameters
    """
    def rank_fn(data, table, workload, predicate, prev_preds):
        w_right, w_left, w_both = workload.split(predicate, prev_preds)
        d_right = []
        d_left = []
        if type(data) == list:
            for row in data:
                if row in predicate:
                    d_right.append(row)
                else:
                    d_left.append(row)
        else:
            # the data is a pandas dataframe
            for row in data.itertuples(index=False):
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
            # We add 1 here to show it is better than an invalid split
            return len(workload) - len(w_both) + 1

    return rank_fn


def index(query, table_path, tree=None, verbose=False):
    """
    For a given query, recursively get the relevant files
    :param query: a Query object
    :param table_path: the path to a table
    :param tree: a tree dictionary.  Only used for recursive calls
    :param verbose: whether to print extra stuff, like which ones we're ignoring
    :return: a list of the paths to relevant files in the qd_tree
    """
    split_path = table_path.split('.')
    storage = split_path[-1]
    path_s = '.'.join(split_path[:-1])
    output = []
    # print(path_s + '.json')

    # Load things for the root
    if tree is None:
        with open(path_s + '.json', 'r') as file:
            tree = load(file)
    # if verbose:
    #     print(tree)
    #     print('')

    # Base case
    if len(tree) == 0:
        output.append(path_s + '.' + storage)
        return output

    # Recursive cases

    valid = False
    pred = pred_gen(tree[0], query.table)
    if intersect(query.list_preds() + [pred]):
        output += index(Query(query.list_preds() + [pred], query.table), path_s + '0.' + storage, tree[1], verbose)
        valid = True
    elif verbose:
        print("Not going down to {} because {} does not intersect".format(path_s + '0', query.list_preds() + [pred]))

    if intersect(query.list_preds() + [pred.flip()]):
        output += index(Query(query.list_preds() + [pred.flip()], query.table), path_s + '1.' + storage, tree[2], verbose)
        valid = True
    elif verbose:
        print("Not going down to {} because {} does not intersect".format(path_s + '0', query.list_preds() + [pred.flip()]))

    if not valid:
        print(query.list_preds())
        print(intersect(query.list_preds()))
        print(query.list_preds() + [pred])
        print(query.list_preds() + [pred.flip()])
        raise Exception("Query matches neither predicate")

    return output







