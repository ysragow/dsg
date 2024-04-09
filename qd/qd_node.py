from qd_predicate import Predicate, Operator
from qd_predicate_subclasses import Numerical, Categorical
from qd_query import Query
from qd_table import Table


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
    def __init__(self, table, preds, root=False, split_pred=None):
        """
        :param table: a table object
        :param preds: a dictionary mapping column names to a tuple containing the associated predicates
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
        self.preds = preds
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
                assert col_preds[0].num < col_preds[1].num, "Contradictory constraints on column " + col.name
            else:
                assert len(col_preds) == 1, "Numerical columns must have exactly 2 predicates"
                assert col_preds[0].op.symbol == 'IN', "The first predicate for this column must be IN"
                assert col_preds[0].column == col, "A predicate for column " + col_preds[0].column.name + "has been assigned to column" + col.name
                assert 'values' in dir(col_preds[0]), "Predicate for this column must be categorical"
                assert len(col_preds[0].values) > 0, "Impossible constraint on column " + col.name
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
    def split(self, pred):
        """
        Split the node into two children
        :param pred: the predicate upon which the node is split
        """
        assert not self.leaf, "This function should not be called on the root node"
        assert not pred.comparative, "A node cannot split on a comparative predicate"
        assert not self.is_split, "This node has already been split"
        old_preds = self.preds[pred.column.name]
        preds1 = {}
        preds2 = {}
        pred_alt = pred.flip(old_preds[0])
        for col in self.table.column_list():
            if col == pred.column.name:
                # For the relevant column, change one predicate
                if pred.column.numerical:
                    index = 0
                    valid = False
                    for i in range(2):
                        if not pred.op(old_preds[i].num, pred.num):
                            index = i
                            valid = not valid
                    assert valid, "This predicate is not within the parent predicate"
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
        self.child_right = Node(table1, preds1, split_pred=pred)
        self.child_left = Node(table2, preds2, split_pred=pred_alt)
        self.is_split = True


class Root(Node):
    """
    Root node class
    """
    def __init__(self, table):
        categorical, mins, maxes = table.get_boundaries()
        preds = {}
        for name in table.list_columns():
            col = table.get_column(name)
            if col.numerical:
                preds[name] = (Numerical(Operator('>='), col, mins[col.num]), Numerical(Operator('<='), col, maxes[col.num]))
            else:
                preds[name] = (Categorical(Operator('IN'), col, categorical[col.num]),)
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
