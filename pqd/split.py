from qd.qd_table import table_gen, Table
from qd.qd_algorithms import reset
from qd.qd_predicate_subclasses import intersect, NumComparative, Categorical, Numerical
from qd.qd_query import Workload
from qd.qd_predicate import Operator


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
    def __init__(self, workload, table, boundaries=None):
        """
        Create a PNode object
        :param workload: The workload at this node
        :param table: A table object
        :param boundaries: A list of predicates beyond the table mins and maxes which bound this node
        """
        if boundaries is None:
            boundaries = []
        self.workload = workload
        self.table = table
        self.boundaries = boundaries
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
        if self.is_split():
            return [str(self.pred), self.right_child.get_tree(), self.left_child.get_tree()]
        else:
            return []






