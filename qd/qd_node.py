import csv

from qd.qd_predicate import Predicate, Operator
from qd.qd_predicate_subclasses import Numerical, Categorical, pred_gen, intersect
from qd.qd_query import Query
from qd.qd_table import Table
import pickle
import os


class Node:
    """
    General node class
    - Every node is initialized as a leaf node, with a table equal to None
    - If you want to give it children, then split the node on a predicate
    - The dictionary preds must always contain:
        - For each numerical column, a tuple of a top predicate and a bottom predicate (both strictly less or greater)
        - For each categorical column, a tuple containing the predicate of included items
    """

    # Setup Functions
    def __init__(self, table, preds=None, root=False, split_pred=None):
        """
        :param table: a table object
        :param preds: a dictionary mapping column names to a tuple containing the associated predicates (deprecated)
        :param root: whether this node the root of the QD-tree
        :param split_pred: If this is not the root, this is the predicate on which this node split from its parent.
        Otherwise, it should be None
        """
        self.table = table
        self.name = self.table.name
        self.leaf = True
        self.root = root
        self.child_right = None
        self.child_left = None
        self.preds = {}
        cats, mins, maxes = table.get_boundaries()
        for cname in table.list_columns():
            col = table.get_column(cname)
            if col.numerical:
                max_pred = Numerical(Operator('<='), col, maxes[cname])
                min_pred = Numerical(Operator('>='), col, mins[cname])
                self.preds[cname] = (min_pred, max_pred)
            else:
                self.preds[cname] = (Categorical(Operator('!IN'), col, set()),)
        if not (root or split_pred):
            raise Exception("If this is not the root, a splitting predicate is required")
        self.split_pred = split_pred
        self.is_split = False
        self.check_inv()

    def check_inv(self, recurse=False):
        """
        Excepts if the predicate is not followed
        :param recurse: whether to recurse on children
        """
        for col_name in self.table.columns.keys():
            col = self.table.get_column(col_name)
            assert self.preds.get(col.name, None), "The column " + col.name + " does not have any predicates"
            assert self.root != bool(self.split_pred), "Roots should not have a split_pred, and non-roots require one"
            col_preds = self.preds[col.name]
            if col.numerical:
                assert len(col_preds) == 2, "Numerical columns must have exactly 2 predicates"
                assert col_preds[0].op.symbol in ('>', '>='), "The first predicate for this column must be > or >="
                assert col_preds[1].op.symbol in ('<', '<='), "The second predicate for this column must be < or <="
                assert col_preds[0].column == col, "A bottom predicate for column " + col_preds[0].column.name + "has been assigned to column" + col.name
                assert col_preds[1].column == col, "A top predicate for column " + col_preds[1].column.name + "has been assigned to column" + col.name
                assert 'num' in dir(col_preds[0]), "Bottom predicate for this column must be numerical"
                assert 'num' in dir(col_preds[1]), "Top predicate for this column must be numerical"
                if col_preds[0].num is None:
                    assert col_preds[1].num is None, "Min is None, max is not"
                else:
                    assert col_preds[0].num <= col_preds[1].num, "Contradictory constraints on column " + col.name
            else:
                assert len(col_preds) == 1, "Numerical columns must have exactly 2 predicates"
                # assert col_preds[0].op.symbol == 'IN', "The first predicate for this column must be IN"
                assert col_preds[0].column == col, "A predicate for column " + col_preds[0].column.name + "has been assigned to column" + col.name
                assert 'values' in dir(col_preds[0]), "Predicate for this column must be categorical"
                # assert len(col_preds[0].values) > 0, "Impossible constraint on column " + col.name
        if recurse and self.is_split:
            self.child_right.check_inv(recurse=True)
            self.child_left.check_inv(recurse=True)

    # Intersection Checkers
    def intersect_t(self, query):
        """
        Short for intersect total
        Checks if the query predicates intersect with every predicates of this node
        :param query: a query
        :return: whether it intersects with all of the predicates of this node
        """
        output = True
        for col in self.table.list_columns():
            for pred in query.predicates[col]:
                output &= pred.intersect(self.preds)
        return output

    def intersect_s(self, query):
        """
        Short for intersect split
        Checks if the query predicates intersects with the split of this node
        Useful under the assumption that any parent differs by the split predicate
        :param query: a query
        :return: whether it intersects with the split predicate of this node
        :raise: error if called on a lead node
        """
        output = True
        if self.root:
            raise Exception("This function should not be called on the root node")
        for pred in query.predicates[self.split_pred.column.name]:
            output &= pred.intersect(self.preds)
        return output

    # Splitting the node
    def split(self, pred, check_bounds=True):
        """
        Split the node into two children
        :param pred: the predicate upon which the node is split
        :param check_bounds: check if this predicate can exist with the bounds of the data
        """
        # assert not self.leaf, "This function should not be called on the root node"
        # assert not pred.comparative, "A node cannot split on a comparative predicate"
        assert not self.is_split, "This node has already been split"
        if check_bounds and (not intersect(list(self.preds[pred.column.name]) + [pred])):
            raise Exception(f"The predicate {pred} is not within the parent preds: \n{list(self.preds[pred.column.name])}")
        old_preds = self.preds[pred.column.name]
        preds1 = {}
        preds2 = {}
        pred_alt = pred.flip()
        for col in self.table.list_columns():
            if (col == pred.column.name) and (not pred.comparative):
                # For the relevant column, change one predicate
                if pred.column.numerical:
                    index = 0
                    valid = False
                    for i in range(2):
                        if not pred.op(old_preds[i].num, pred.num):
                            index = i
                            valid = not valid
                    if check_bounds and (not intersect(list(self.preds[pred.column.name]) + [pred])):
                        raise Exception(f"The predicate {pred} is not within the parent preds: \n{list(self.preds[pred.column.name])}")
                    new_preds1 = []
                    new_preds2 = []
                    for i in range(2):
                        if i == index:
                            new_preds1.append(pred)
                            new_preds2.append(old_preds[i])
                        else:
                            new_preds1.append(old_preds[i])
                            new_preds2.append(pred_alt)
                    preds1[col] = tuple(new_preds1)
                    preds2[col] = tuple(new_preds2)
                else:
                    preds1[col] = (pred,)
                    preds2[col] = (pred_alt,)
            else:
                # For every other column, use the same predicate
                preds1[col] = self.preds[col]
                preds2[col] = self.preds[col]
        table1, table2 = self.table.split(pred)
        if not self.root:
            self.table.delete()
        self.split_pred = pred
        self.child_right = Node(table1, preds1, split_pred=pred)
        self.child_left = Node(table2, preds2, split_pred=pred_alt)
        self.is_split = True


class Root(Node):
    """
    Root node class

    Notes on tree json formulation:
    - The root is represented as a list containing:
        - First, the list of starting predicates
        - Then, child0
        - Then, child1
        - Then, a dictionary mapping leaf node names to a list of their predicates
    - Each node (aside from the root) is represented as a list containing:
        - First, the predicate differentiating that node from its parent
        - Then, child0 (if it exists)
        - Then, child1 (if it exists)
    """
    def __init__(self, table):
        if os.path.isfile('.'.join(table.path.split('.')[:-1]) + 'pickle'):
            raise Exception("A tree for this file exists.")
        else:
            # This tree has not been created yet, so we make a new one
            categorical, mins, maxes = table.get_boundaries()
            preds = {}
            for name in table.list_columns():
                col = table.get_column(name)
                index = col.num if (table.storage == 'csv') else col.name
                if col.numerical:
                    preds[name] = (Numerical(Operator('>='), col, mins[index]), Numerical(Operator('<='), col, maxes[index]))
                else:
                    preds[name] = (Categorical(Operator('!IN'), col, set()),)
            super().__init__(table, preds, root=True)
            self.leaves = {self.table.name: self}

    def split_leaf(self, leaf, pred):
        """
        :param leaf: a leaf node of this root
        :param pred: a predicate
        :return: the right and left children of this leaf
        """
        assert leaf.name in self.leaves, "This is not a leaf node"
        leaf.split(pred)
        self.leaves[leaf.child_right.name] = leaf.child_right
        self.leaves[leaf.child_left.name] = leaf.child_left
        del self.leaves[leaf.name]
        return leaf.child_right, leaf.child_left

    def get_data(self, query, use_tree=True, count_invalid_data=False, count_partitions=False, verbosity=0):
        """
        :param query: a query object
        :param use_tree: whether to use the tree structure, or to just take straight from one table
        :param count_invalid_data: whether to keep track of and return the number of invalid data points
        :param count_partitions: whether to count how many partitions each query saw
        :param verbosity: how much detail to print
        :return: if count_partitions:
                    returns the data, followed by the invalid count, followed by the partition count
                 elif count_invalid_data:
                    returns the data as a list, followed by the invalid data count.
                 else:
                    just returns the data.
        """
        bad_data_count = 0
        leaf_count = 0
        output = []
        if use_tree:
            for leaf_name in self.leaves.keys():
                if verbosity == 1:
                    print("Reading {}".format(leaf_name), end="\r")
                leaf = self.leaves[leaf_name]
                if leaf.intersect_t(query):
                    leaf_count += 1
                    with open('data/' + leaf.table.name + '.csv') as file:
                        data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                        data.__next__()
                        data_count = 0
                        for row in data:
                            if verbosity == 2:
                                print("Reading tuple {} of {}".format(data_count,leaf_name), end="\r")
                            data_count += 1
                            if row in query:
                                output.append(row)
                            elif count_invalid_data:
                                bad_data_count += 1
        else:
            with open('data/' + self.table.name + '.csv') as file:
                data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                data.__next__()
                leaf_count = 1
                for row in data:
                    if row in query:
                        output.append(row)
                    elif count_invalid_data:
                        bad_data_count += 1
        if count_partitions:
            return output, bad_data_count, leaf_count
        if count_invalid_data:
            return output, bad_data_count
        return output

    def delete(self):
        for leaf in self.leaves.keys():
            self.leaves[leaf].table.delete()
        self.table.delete()
