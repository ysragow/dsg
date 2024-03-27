from qd_predicate import Predicate


class Node:
    """
    General node class
    - Every node is initialized as a leaf node, with a table equal to None
    - If you want to give it children, then split the node on a predicate
    - The dictionary preds must always contain:
        - For each numerical column, a tuple of a top predicate and a bottom predicate (both strictly less or greater)
        - For each categorical column, a tuple containing the predicate of included items
    """
    def __init__(self, table, preds):
        """
        :param table: a table object
        :param preds: a dictionary mapping column names to a tuple containing the associated predicates
        """
        self.table = table
        self.leaf = True
        self.child1 = None
        self.child2 = None
        self.preds = preds
        self.check_inv()

    def check_inv(self):
        """
        Excepts if the predicate is not followed
        """
        for col in self.table.columns:
            assert self.preds.get(col.name, None), "The column " + col.name + " does not have any predicates"
            col_preds = self.preds[col]
            if col.numerical:
                assert len(col_preds) == 2, "Numerical columns must have exactly 2 predicates"
                assert col_preds[0].op.symbol == '>', "The first predicate for this column must be >"
                assert col_preds[1].op.symbol == '<', "The second predicate for this column must be <"
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


