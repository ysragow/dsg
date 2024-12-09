from qd.qd_table import table_gen, Table
from qd.qd_algorithms import reset
from qd.qd_predicate_subclasses import intersect, NumComparative, Categorical, Numerical, pred_gen
from qd.qd_query import Workload
from qd.qd_predicate import Operator
from qd.qd_node import Root, Node


def all_predicates_workload(table, workload, columns=None):
    """
    Get all the predicates given a table and a workload
    :param table: A qd table object
    :param workload: A qd workload object
    :param columns: The columns you are allowed to predicate on
    :return: All possible predicates
    """
    predicates = []

    # If no columns are given to predicate on, then allow for predicating on all columns
    if columns is None:
        columns = table.list_columns()

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
        comp_values = set()
        for c_2 in num_columns:
            if c_1.name != c_2.name:
                predicates.append(NumComparative(Operator('<'), c_1, c_2))
                predicates.append(NumComparative(Operator('>'), c_1, c_2))
        # Every numerical predicate on real numbers in the workload
        for query in workload.queries:
            for pred in query.predicates[c_1.name]:
                if (not pred.comparative) and (pred.num not in comp_values):
                    predicates.append(Numerical(Operator('<'), c_1, pred.num))
                    predicates.append(Numerical(Operator('>'), c_1, pred.num))
                    comp_values.add(pred.num)

    # Every predicate comparing dates
    for c_1 in date_columns:
        comp_values = set()
        for c_2 in date_columns:
            if c_1.name != c_2.name:
                predicates.append(NumComparative(Operator('<'), c_1, c_2))
        # Every numerical predicate on dates in the workload
        for query in workload.queries:
            for pred in query.predicates[c_1.name]:
                if (not pred.comparative) and (pred.num not in comp_values):
                    predicates.append(Numerical(Operator('<'), c_1, pred.num))
                    predicates.append(Numerical(Operator('>'), c_1, pred.num))
                    comp_values.add(pred.num)

    # Every categorical predicate (with set size of 1)
    for c in cat_columns:
        items = set()
        for query in workload.queries:
            for pred in query.predicates[c.name]:
                if not pred.comparative:
                    for item in pred.values:
                        items.add(item)
        for item in items:
            predicates.append(Categorical(Operator('IN'), c, {item}))

    return predicates


class PNode:
    """
    A node containing a workload.  Its main function is to split and output a tree
    """
    def __init__(self, workload, table, boundaries=None, tree=None):
        """
        Create a PNode object
        :param workload: The workload at this node
        :param table: A table object
        :param boundaries: A list of predicates beyond the table mins and maxes which bound this node
        :param tree: You can initialize this from a tree, as well
        """
        if boundaries is None:
            boundaries = []
        self.workload = workload
        self.table = table
        self.boundaries = boundaries
        if tree is not None:
            if len(tree) != 0:
                pred = pred_gen(tree[0], table)
                self.right_child = PNode(workload, table, boundaries=boundaries + [pred], tree=tree[1])
                self.left_child = PNode(workload, table, boundaries=boundaries + [pred.flip], tree=tree[2])
                self.pred = pred
                return
        self.left_child = None
        self.right_child = None
        self.pred = None

    def is_split(self):
        return self.pred is not None

    def split(self, factor=1, verbose=False, id='r'):
        """
        :param factor: The maximum value of len(both_wkld) / len(self.workload)
        :return: True if the node has been split successfully, false otherwise
        """
        self.pred = None
        self.left_child = None
        self.right_child = None
        if verbose:
            print(f"Making preds for node {id} with workload size: {len(self.workload)}", end='\r')
        all_preds = all_predicates_workload(self.table, self.workload)
        best_split_len = len(self.workload)
        best_pred = None
        if verbose:
            print(f"Testing preds for node {id} with workload size: {len(self.workload)}", end='\r')
        for pred in all_preds:
            right_wkld, left_wkld, both_wkld = self.workload.split(pred, self.boundaries)
            if (len(both_wkld) < best_split_len)\
                    and (len(right_wkld) > 0) \
                    and (len(left_wkld) > 0)\
                    and ((len(both_wkld) / len(self.workload)) < factor):
                best_split_len = len(both_wkld)
                best_pred = pred
        if best_pred is None:
            if verbose:
                print(f"Leaving node {id} with workload size: {len(self.workload)}")
            return False  # cannot be split further
        right_wkld, left_wkld, both_wkld = self.workload.split(best_pred, self.boundaries)
        self.left_child = PNode(left_wkld, self.table, self.boundaries + [best_pred.flip()])
        self.right_child = PNode(right_wkld, self.table, self.boundaries + [best_pred])
        if verbose:
            print(f"Recursing to right child of size {len(right_wkld)} for node {id} with workload size: {len(self.workload)}")
        self.right_child.split(factor=factor, verbose=verbose, id=id + '0')
        if verbose:
            print(f"Recursing to left child of size {len(left_wkld)} for node {id} with workload size: {len(self.workload)}")
        self.left_child.split(factor=factor, verbose=verbose, id=id + '1')
        self.pred = best_pred
        return True

    def get_tree(self):
        # Returns a tree object that can be turned into a json
        if self.is_split():
            return [str(self.pred), self.right_child.get_tree(), self.left_child.get_tree()]
        else:
            return []

    # def index(self, query):
    #     """
    #     Index this node using a query
    #     :param query: a qd query object
    #     :return: A list of leaf nodes intersected by the query
    #     """
    #     if not self.pred:
    #         # This is a leaf node
    #         return [self]
    #     output
    #     if intersect(query.list_preds(), self.boundaries, )

    def statistics(self):
        if self.pred is None:
            size = len(self.workload)
            return {'max': size, 'min': size, 'average': size, 'depth': 1, 'leaves': 1}
        else:
            left_stats = self.left_child.statistics()
            right_stats = self.right_child.statistics()
            max_size = max(left_stats['max'], right_stats['max'])
            min_size = min(left_stats['min'], right_stats['min'])
            leaves = left_stats['leaves'] + right_stats['leaves']
            average = (left_stats['leaves'] * left_stats['average']) + (right_stats['leaves'] * right_stats['average'])
            average /= leaves
            depth = 1 + max(left_stats['depth'], right_stats['depth'])
            return {'max': max_size, 'min': min_size, 'average': average, 'depth': depth, 'leaves': leaves}

    def to_qd(self, base_data_path, folder_path, qd_node=None):
        """
        Turn a PNode tree into a qd tree
        :param base_data_path:
        :param folder_path:
        :param qd_node: a node object from qd.qd_node
        :return: the root of the corresponding qd tree
        """
        # Initialize on the root node
        if qd_node is None:

            # Make the folder, if it doesn't exit
            import os
            from json import dump
            if not os.path.exists(folder_path):
                _ = os.makedirs(folder_path)

            # Save the tree
            rname = '.'.join(base_data_path.split('/')[-1].split('.')[:-1])
            with open(folder_path + "/" + rname + ".json", "w") as file:
                dump(self.get_tree(), file)

            # Initialize the objects
            table = table_gen(base_data_path, child_folder=folder_path)
            qd_node = Root(table)

        # If not a leaf, then split and recurse
        if self.pred is not None:
            qd_node.split(self.pred, check_bounds=False)
            _ = self.right_child.to_qd(None, None, qd_node=qd_node.child_right)
            _ = self.left_child.to_qd(None, None, qd_node=qd_node.child_left)
        return qd_node













