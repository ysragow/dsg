from qd_predicate import Predicate, Operator
from qd_column import Column
from qd_table import Table
import json


class Categorical(Predicate):
    # A categorical predicate, using = or IN
    def __init__(self, op, column, categories):
        """
        :param op: the operator this predicate is based on
        :param column: the column this predicate breaks on
        :param categories: the set of categories included in this predicate
        """
        super().__init__(op, column)
        self.values = categories
        assert (not column.numerical), "This column cannot be used for a categorical predicate"
        assert op.symbol == 'IN', "Wrong type of predicate"

    def __contains__(self, item):
        return item[self.column.num] in self.values

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: whether that set intersects with the items in our predicate
        """
        output = True
        for pred in preds[self.column.name]:
            output &= self.op(self.values, pred.values)
        return output

    def flip(self, parent_pred=None):
        """
        :param parent_pred: the parent predicate
        :return: predicate: the inverse of this predicate
        """
        assert parent_pred is not None, "This predicate requires a parent predicate"
        assert parent_pred.column == self.column, "Parent predicate column does not match"
        for item in self.values:
            assert item in parent_pred.values, "This predicate is not a subset of the parent predicate"
        output = set()
        for item in parent_pred.values:
            if item not in self.values:
                output.add(item)
        return Categorical(self.op, self.column, output)


class Numerical(Predicate):
    # A numerical predicate, using <, >, =>, <=, or =
    def __init__(self, op, column, num):
        """
        :param op: the operation this predicate is based upon.  must be in 1-4
        :param column: the column this predicate breaks on
        :param num: the number that we measure against the column value
        """
        super().__init__(op, column)
        self.num = num
        assert column.numerical, "This column cannot be used for a numerical predicate"
        assert op.symbol != 'IN', "Wrong type of predicate"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        return self.op(item[self.column.num], self.num)

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping columns to predicates
        :return: whether the predicates intersect
        """
        output = True
        for pred in preds[self.column.name]:
            num = pred.num
            op = pred.op
            output &= self.op(num, self.num) or op(self.num, num) or ((self.op == op) & (self.num == num))
        return output

    def flip(self, parent_pred=None):
        """
        :param parent_pred: extraneous argument, here for inheritance reasons
        :return: predicate: the inverse of this predicate
        """
        return Numerical(self.op.flip(), self.column, self.num)


class CatComparative(Predicate):
    # A comparative predicate between two columns.  Only used in queries, never in nodes
    def __init__(self, op, col1, col2):
        """
        :param op: the code for the operation this predicate is based upon.  must be 5 or less
        :param col1: the column this predicate breaks on
        :param col2: the column this predicate breaks on
        """
        super().__init__(op, col1)
        self.comparative = True
        self.col2 = col2
        assert col1.numerical == col2.numerical, "These columns cannot be compared"
        assert (op.symbol == '='), "Categorical comparison requires equality statement"
        assert (not col1.numerical), "These columns are not categorical"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        return item[self.column.num] == item[self.col2.num]

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: whether the predicates intersect
        """
        output = True
        preds1 = preds[self.column.name]
        preds2 = {self.col2.name: preds[self.col2.name]}
        for pred in preds1:
            output &= pred.intersect(preds2)
        return output


class NumComparative(Predicate):
    # A comparative predicate between two columns.  Only used in queries, never in nodes
    def __init__(self, op, col1, col2):
        """
        :param op: the code for the operation this predicate is based upon.  must be 5 or less
        :param col1: the column this predicate breaks on
        :param col2: the column this predicate breaks on
        """
        super().__init__(op, col1)
        self.col2 = col2
        self.comparative = True
        assert col1.numerical == col2.numerical, "These columns cannot be compared"
        assert col1.numerical, "Wrong type of comparative predicate"
        assert op.symbol != "IN", "This operation cannot be used to compare columns"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        return self.op(item[self.column.num], item[self.col2.num])

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: whether the predicates intersect
        """
        output = True
        plist = list(preds[self.column.name] + preds[self.col2.name])
        ops = [self.op]
        if self.op.symbol == '=':
            # = can be defined as <= and >=
            ops = [Operator('<='), Operator('>=')]
        for i in range(len(ops)):
            p1 = plist[i]
            p2 = plist[-i-1]
            # Either one is strictly greater than the other, or they are equal and all operations allow equality
            output &= (self.op(p1.num, p2.num) and (p1.num != p2.num)) \
                      or\
                      all((self.op(p1.num, p2.num), p1.op(p1.num, p2.num), p2.op(p1.num, p2.num)))
        return output


def pred_gen(pred_string, table):
    """
    :param pred_string: a string of the form "column_name operator value"
    :param table: an instance of the table class
    :return: a predicate based on the string
    """
    # print(pred_string)
    col_name, op_name, value_name = pred_string.split(" ")
    column = table.get_column(col_name)
    assert column is not None, "The column " + col_name + " does not exist in this table."
    op = Operator(op_name)
    if table.get_column(value_name):
        # Instance of a comparative predicate
        column2 = table.get_column(value_name)
        if column.numerical:
            return NumComparative(op, column, column2)
        else:
            return CatComparative(op, column, column2)
    elif value_name.replace('.', '', 1).isdigit() or (value_name[1:].replace('.', '', 1).isdigit() and value_name[0] == '-'):
        # Instance of a numerical predicate
        num = int(value_name) if value_name.isdigit() else float(value_name)
        assert column.numerical, "This is not a numerical column, so it cannot be compared with a number"
        return Numerical(op, column, num)
    elif (value_name[0] == '(') and (value_name[-1] == ')'):
        values = set(json.loads(value_name.replace("'", '"').replace('(', '[').replace(')', ']')))
        # this is not finished!
    else:
        raise Exception("Something's wrong")


