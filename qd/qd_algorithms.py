from qd.qd_query import Query, Workload
from qd.qd_node import Node, Root
from qd.qd_column import Column
from qd.qd_table import Table, table_gen
from qd.qd_predicate import Operator
from qd.qd_predicate_subclasses import pred_gen, Numerical, NumComparative, Categorical, CatComparative, intersect, BigColumnBlock
from fastparquet import ParquetFile
import numpy as np
from json import dump, load, loads
from glob import glob
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


def all_predicates(data, table, columns=None, workload=None):
    """
    :param data: a list of datapoints
    :param table: the table from where the datapoints came
    :param columns: a list of column names that can be predicated on
    :param workload: a workload
    :return: a list of all linear predicates separating this datapoints
    """
    predicates = []
    if workload is not None:
        for q in workload.queries:
            for p in q.list_preds():
                predicates.append(p)
        return predicates

    # If no columns are given to predicate on, then allow for predicating on all columns
    if columns is None:
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


def tree_gen(table, workload, rank_fn=None, subset_size=60, node=None, root=None, prev_preds=None, columns=None, block_size=1000, pred_blocks=None, subset_size_factor=None):
    """
    :param table: a table object
    :param workload: a workload object
    :param rank_fn: a function that operates on the following parameters: (only for recursive calls)
        param data: a list containing some data
        param table: the same table as used prior
        param workload: a workload object
        param predicate: a predicate on the data
        param prev_preds: previous preds used to split the data up to this point
        param q_blocks: an optional argument; blocks
        returns a ranking of how well the predicate splits the operation of the workload on the data.
                better predicates rank higher, and the ranking should be 0 if it cannot be used
    :param subset_size: the size of the data subsets
    :param node: the corresponding node of this function.  used only for recursive calls
    :param root: the root node of the tree being built.  used only for recursive calls
    :param prev_preds: previous predicates leading up to this one.  used only for recursive calls
    :param columns: the columns that can be predicated on.  if none, then all columns can be predicated on
    :param block_size: the minimum size of a block.  The maximum size is twice the block size.
    :param pred_blocks: a list of predicate blocks.  used only for recursive calls
    :return: a tree as a nested list
    """
    top = False
    output = []
    if node is None and root is None:
        print("Initializing...", end='\n')
        node = Root(table)
        root = node
        prev_preds = []
        top = True
        rank_fn = rank_fn_gen(block_size)
        workload = reset(table, workload)
        # pred_blocks = []
        # for q in workload.queries:
        #     block = BigColumnBlock()
        #     for pred in q.list_preds():
        #         if not block.add(pred, false_if_fail_test=True):
        #             # If the query itself is contradictory, it is represented by a block that always returns False
        #             block = BigColumnBlock(always_false=True)
        #             break
        #     prev_preds.append(block)



        print("Done.             ")
    print("Generating subset...", end='\r')
    valid_splits = False
    first_try = True
    while not valid_splits:
        top_score = 0
        if table.size > 2 * block_size:
            if (subset_size_factor is not None) and first_try:
                subset_size = int(table.size * subset_size_factor)
                subset_size = (2 * int(subset_size / 2)) + 2
            subset = subset_gen(table, subset_size)
            print("Generating preds...", end='\r')
            if first_try:
                preds = all_predicates(subset, root.table, columns=columns, workload=workload)
            else:
                preds = all_predicates(subset, root.table, columns=columns)
            print(f"Testing {len(preds)} preds on sample of size {len(subset)}...")
            for pred in preds:
                # if len(str(pred)) > 100:
                #     print("acting on pred:", str(pred)[:100] + '                                 ', end='\r')
                # else:
                #     print("acting on pred:", pred, '                                 ', end='\r')
                print(f"Testing pred: {pred}", end='\r')
                score = rank_fn(subset, table, workload, pred, prev_preds)
                # print(f"Score for pred {pred}: {score}")
                if score > top_score:
                    best_pred = pred
                    top_score = score
        if top_score > 0:
            score = rank_fn(subset, table, workload, best_pred, prev_preds, verbose=True)
            workload_right, workload_left, _ = workload.split(best_pred, prev_preds)
            print(f'Choosing the predicate "{best_pred}".  {len(workload_left.queries)} queries go left, and {len(workload_right.queries)} queries go right.')
            print('Splitting {} into {} and {}...'.format(table.name, table.name + '0', table.name + '1'), end='\r')
            child_right, child_left = root.split_leaf(node, best_pred)
            dict_right = tree_gen(child_right.table,
                                  workload_right,
                                  rank_fn=rank_fn,
                                  subset_size=subset_size,
                                  node=child_right,
                                  root=root,
                                  prev_preds=prev_preds + [best_pred],
                                  columns=columns,
                                  block_size=block_size,
                                  subset_size_factor=subset_size_factor
                                  )
            dict_left = tree_gen(child_left.table,
                                 workload_left,
                                 rank_fn=rank_fn,
                                 subset_size=subset_size,
                                 node=child_left,
                                 root=root,
                                 prev_preds=prev_preds + [best_pred.flip()],
                                 columns=columns,
                                 block_size=block_size,
                                 subset_size_factor=subset_size_factor
                                 )
            output.append(str(best_pred))
            output += [dict_right, dict_left]
            valid_splits = True
        elif table.size > (2 * block_size):
            first_try = False
            print("Failed to split {} rows; minimum block size is {}  Trying again.".format(table.size, block_size))
        else:
            valid_splits = True
            print("Leaving {} with {} rows".format(table.path, table.size))
    if top:
        print(output)
        print("Writing output to " + table.child_folder + '/' + table.name + '.json')
        with open(table.child_folder + '/' + table.name + '.json', 'w') as file:
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
    def pure_block_rank(pred, q_blocks):
        """
        Rank the pred only on how it performs on the blocks
        :param pred: a predicate
        :param q_blocks: a list of blocks representing queries and previous predicates
        :return: a ranking of the pred; higher is BETTER
        """
        neg_pred = pred.flip()
        right_valid = False
        left_valid = False
        both_count = 0
        for block in q_blocks:
            right_test = block.test(pred)
            left_test = block.test(neg_pred)
            right_valid |= right_test
            left_valid |= left_test
            if right_test and left_test:
                both_count += 1
        if right_valid and left_valid:
            return len(q_blocks) - both_count
        return 0

    def pure_workload_rank(pred, workload, prev_preds):
        """
        Rank the pred only on how it performs on the workload and previous preds
        :param pred: a predicate
        :param workload: a workload
        :param prev_preds: previous predicates that apply to the entire workload
        :return: a ranking of the predicate; higher is BETTER
        """
        w_right, w_left, w_both = workload.split(pred, prev_preds)
        if (len(w_right) > 0) and (len(w_left) > 0):
            return len(workload.queries) - len(w_both)
        return 0

    if min_size == 0:
        def rank_fn(data, table, workload, predicate, prev_preds, q_blocks=None, verbose=False):
            if q_blocks is None:
                init_rank = pure_workload_rank(predicate, workload, prev_preds)
            else:
                init_rank = pure_block_rank(predicate, q_blocks)
            if verbose:
                print("This rank function does not care about block sizes")
            return init_rank + 1
        return rank_fn

    def rank_fn(data, table, workload, predicate, prev_preds, q_blocks=None, verbose=False):
        if q_blocks is None:
            init_rank = pure_workload_rank(predicate, workload, prev_preds)
        else:
            init_rank = pure_block_rank(predicate, q_blocks)
        d_right = []
        d_left = []
        neg_pred = predicate.flip()
        if type(data) == list:
            for row in data:
                if row in predicate:
                    d_right.append(row)
                else:
                    d_left.append(row)
        else:
            # the data is a pandas dataframe
            if predicate.comparative:
                # Any comparative
                d_right = data[predicate.op(data[predicate.column.name], data[predicate.col2.name])]
                d_left = data[neg_pred.op(data[neg_pred.column.name], data[neg_pred.col2.name])]
                # for row in data.itertuples(index=False):
                #     if row in predicate:
                #         d_right.append(row)
                #     else:
                #         d_left.append(row)
            elif predicate.column.numerical:
                # Numerical
                d_right = data[predicate.op(data[predicate.column.name], predicate.num)]
                d_left = data[neg_pred.op(data[neg_pred.column.name], neg_pred.num)]
            else:
                # Categorical.  Assume it only has 1 value.
                value = list(predicate.values)[0]
                d_right = data[data[predicate.column.name] == value]
                d_left = data[data[neg_pred.column.name] != value]
        if len(d_right)*table.size/len(data) < min_size:
            if verbose:
                print(f"Only {len(d_right)} rows out of {len(data)} go right, so this is invalid")
            return 0
        if len(d_left)*table.size/len(data) < min_size:
            if verbose:
                print(f"Only {len(d_left)} rows out of {len(data)} go left, so this is invalid")
            return 0
        if multiply_sizes:
            if verbose:
                print("Sizes are being multiplied")
            return len(workload)*len(data) - len(w_right)*len(d_right) - len(w_left)*len(d_left)
        else:
            # We add 1 here to show that init_rank = 0 is better than an invalid split
            if verbose:
                print(f'''The predicate {predicate} scores {init_rank + 1}.
                      The total size of the sample is {len(data)}, with {len(d_left)} rows going left and {len(d_right)} going right.''')
            return init_rank + 1

    return rank_fn


def reset(table, obj):
    """
    Reset an object to match a table
    :param table: A Table object
    :param obj: A predicate, column, query, or workload
    :return: The object with this table set as its table
    """
    if 'ctype' in dir(obj):
        # This is a column
        return table.get_column(obj.name)
    elif 'op' in dir(obj):
        # This is a predicate
        return pred_gen(str(obj), table)
    elif 'predicates' in dir(obj):
        # This is a query
        return Query(list([reset(table, p) for p in obj.list_preds()]), table)
    elif 'queries' in dir(obj):
        # This is a workload
        return Workload(list([reset(table, q) for q in obj.queries]))
    else:
        raise Exception("Invalid object for resetting")


def index(query, root_path, table, tree=None, block=None, verbose=False):
    """
    For a given query, recursively get the relevant files
    :param query: a Query object
    :param root_path: the path to the root object, whether or not it actually exists
    :param table: a table object
    :param tree: a tree dictionary.  Only used for recursive calls
    :param block: a BigColumnBlock object.  Only used for recursive calls
    :param verbose: whether to print extra stuff, like which ones we're ignoring
    :return: a list of the paths to relevant files in the qd_tree
    """
    split_path = root_path.split('.')
    storage = split_path[-1]
    path_s = '.'.join(split_path[:-1])
    output = []
    # print(path_s + '.json')

    # Load things for the root
    if tree is None:
        with open(path_s + '.json', 'r') as file:
            tree = load(file)
        query = reset(table, query)
        block = BigColumnBlock()
        for pred in query.list_preds():
            if not block.test(pred):
                return False
            block.add(pred)
    # if verbose:
    #     print(tree)
    #     print('')

    # Base case
    if len(tree) == 0:
        output.append(path_s + '.' + storage)
        return output
    elif len(tree) == 1:
        return output

    # Recursive cases

    valid = False
    if verbose:
        print("Pred:", tree[0])
    pred = pred_gen(tree[0], table)
    match_right = block.test(pred)
    match_left = block.test(pred.flip())
    left_block = None
    right_block = None
    if match_right & match_left:
        right_block = block.fork(pred)
        left_block = block
        block.add(pred.flip())
    elif match_right:
        right_block = block
        block.add(pred)
    elif match_left:
        left_block = block
        block.add(pred.flip())
    else:
        raise RuntimeError("Something is wrong with predicate check")
    if match_right:
        # Make sure all column object being acted on are the same by resetting the preds
        new_path = path_s + '0.' + storage
        new_query = Query(query.list_preds() + [pred], table)
        output += index(new_query, new_path, table, tree=tree[1], block=right_block, verbose=verbose)
        valid = True
    elif verbose:
        print("Not going down to {} because {} does not intersect".format(path_s + '0', query.list_preds() + [pred]))

    if match_left:
        new_path = path_s + '1.' + storage
        new_query = Query(query.list_preds() + [pred.flip()], table)
        output += index(new_query, new_path, table, tree=tree[2], block=left_block, verbose=verbose)
        valid = True
    elif verbose:
        print("Not going down to {} because {} does not intersect".format(path_s + '0', query.list_preds() + [pred.flip()]))

    if not valid:
        print("WARNING: INVALID QUERY / PRED COMBINATION")
        if not intersect(query.list_preds()):
            print("")
            print("The following query does not intersect itself:")
            print(query)
        else:
            print("")
            print("Pred:", pred)
            print("Query:", query)
            print("Right preds:", query.list_preds() + [pred])
            print("Left preds:", query.list_preds() + [pred.flip()])
        raise Exception("Query matches neither predicate")

    return output


def q_gen_const(path):
    """
    Generates a query object constructor given a path (to a folder, or to a parquet file)
    :param path: A path to a folder containing parquet files or to a parquet file
    :return: A function which takes in lists of predicates strings and outputs a query
    """
    # from os import getcwd
    # print("Working directory:", getcwd())
    if path[-8:] == '.parquet':
        table = table_gen(path)
    else:
        p_paths = glob(path + ('' if path[-1] == '/' else '/') + '*.parquet' )
        assert len(p_paths) > 0, path + " is not a parquet file or a folder containing parquet files"
        table = table_gen(p_paths[0])
    return lambda strs: Query([pred_gen(s, table) for s in strs], table)


def load_workload(path, table):
    """
    Loads a workload from a json file according to a table object
    :param path: path to the workload json
    :param table: a table object
    :return: a workload object
    """
    q_gen = q_gen_const(table.path)
    with open(path, "r") as file:
        wkld_json = load(file)
    return Workload(list([q_gen(q) for q in wkld_json]))
