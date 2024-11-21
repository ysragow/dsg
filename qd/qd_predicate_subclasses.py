from qd.qd_predicate import Predicate, Operator
from qd.qd_column import Column
from qd.qd_table import Table
from pyarrow.compute import field, scalar
from pyarrow import timestamp, scalar as pa_scalar
from numpy import datetime64
import json


class Categorical(Predicate):
    # A categorical predicate, using = or IN
    def __init__(self, op, column, categories):
        """
        :param op: the operator this predicate is based on
        :param column: the column this predicate breaks on
        :param categories: the set of categories included in this predicate
        """
        super().__init__(op, column, list(categories))
        self.values = categories
        self.str_right = str(categories)
        assert (not column.numerical), "This column cannot be used for a categorical predicate"
        assert op.symbol in ('IN', '!IN'), "Wrong type of predicate"

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

    def flip(self):
        """
        :return: predicate: the inverse of this predicate
        """
        return Categorical(self.op.flip(), self.column, self.values)

    def to_expression(self):
        """
        :return: A parquet expression representing this predicate
        """
        if self.op.symbol == 'IN':
            return field(self.column.name).isin(list(self.values))
        else:
            output = None
            for value in list(self.values):
                new_exp = (field(self.column.name) != scalar(value))
                if output is None:
                    output = new_exp
                else:
                    output &= new_exp
            return output


class Numerical(Predicate):
    # A numerical predicate, using <, >, =>, <=, or =
    def __init__(self, op, column, num):
        """
        :param op: the operator this predicate is based upon
        :param column: the column this predicate breaks on
        :param num: the number that we measure against the column value
        """
        super().__init__(op, column, num)
        self.str_right = str(num)
        self.num = num
        assert column.numerical, "This column cannot be used for a numerical predicate"
        assert op.symbol != 'IN', "Wrong type of predicate"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        try:
            return self.op(item[self.column.num], self.num)
        except Exception as ex:
            print(self.column, self.column.num, self.num)
            raise ex

    def to_dnf(self):
        """
        :return: this predicate as a DNF expression
        """
        return self.column.name, self.op.symbol, self.num

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

    def to_expression(self):
        """
        :return: A parquet expression representing this predicate
        """
        if self.column.ctype != 'DATE':
            return self.op(field(self.column.name), scalar(self.num))
        else:
            print(self.num)
            return self.op(field(self.column.name), pa_scalar(datetime64(str(self.num)), type=timestamp('s')))


class CatComparative(Predicate):
    # A comparative predicate between two columns.  Only used in queries, never in nodes
    def __init__(self, op, col1, col2):
        """
        :param op: the operator this predicate is based upon
        :param col1: the column this predicate breaks on
        :param col2: the column this predicate breaks on
        """
        super().__init__(op, col1, col2)
        self.str_right = str(col2)
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
        super().__init__(op, col1, col2)
        self.str_right = str(col2)
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
        try:
            return self.op(item[self.column.num], item[self.col2.num])
        except Exception as ex:
            print(item)
            print(self.column, self.column.num, self.col2, self.col2.num)
            raise ex

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
            p2 = plist[-i - 1]
            # Either one is strictly greater than the other, or they are equal and all operations allow equality
            output &= (self.op(p1.num, p2.num) and (p1.num != p2.num)) or all(
                (self.op(p1.num, p2.num), p1.op(p1.num, p2.num), p2.op(p1.num, p2.num)))
        return output

    def to_dnf(self):
        return self.column.name, self.op.symbol, self.col2.name


    def flip(self, parent_pred=None):
        return NumComparative(self.op.flip(), self.column, self.col2)

    def to_expression(self):
        """
        :return: A parquet expression representing this predicate
        """
        return self.op(field(self.column.name), field(self.col2.name))


def intersect(preds, debug=False):
    date_preds = []
    num_preds = []
    cat_preds = []
    for pred in preds:
        if pred.column.numerical:
            if pred.column.ctype == 'DATE':
                date_preds.append(pred)
            else:
                num_preds.append(pred)
        else:
            cat_preds.append(pred)
    if debug:
        print("Categoricals:", cat_intersect(cat_preds, debug))
        print("Dates:", num_intersect(date_preds, debug))
        print("Numbers:", num_intersect(num_preds, debug))
    return cat_intersect(cat_preds) & num_intersect(date_preds) & num_intersect(num_preds)


def cat_intersect(preds, debug=False):
    """
    Checks whether a list of categorical predicates can all be satisfied
    :param preds: A list of numeric predicates
    :param debug: Whether to print anything causing a falsehood
    :return: Whether there exists a point where they intersect
    """
    # No cat comparatives, so no need for an index
    # Instead we map column names to sets of possible outcomes
    col_sets = {}
    col_not_sets = {}
    for pred in preds:
        if pred.op.symbol == '!IN':
            if pred.column.name in col_not_sets:
                for value in pred.values:
                    col_not_sets[pred.column.name].add(value)
            else:
                col_not_sets[pred.column.name] = pred.values.copy()
        elif pred.op.symbol == 'IN':
            if pred.column.name in col_sets:
                current_values = list(col_sets[pred.column.name])
                for value in current_values:
                    if value not in pred.values:
                        col_sets[pred.column.name].remove(value)
            else:
                col_sets[pred.column.name] = pred.values.copy()
        else:
            raise Exception("Invalid Predicate: " + str(pred))
    for column in col_sets.keys():
        if column in col_not_sets:
            for value in col_not_sets[column]:
                if value in col_sets[column]:
                    col_sets[column].remove(value)
        if len(col_sets[column]) == 0:
            if debug:
                print(f"Column {column} cannot take on any value")
            return False
    return True


class ColumnNode:
    """
    Turns a column into a column node
    """
    def __init__(self, c, col_index, debug=False):
        """
        Take a column and turn it into a column node
        :param c: A column
        :param col_index: A dictionary mapping all column names to their column nodes
        :param debug: Whether to print the mins and max of the column
        NOTE: col_index IS SHARED AMONG ALL COLUMN NODES
        """
        self.col_index = col_index

        # Linked list attributes
        self.left = None
        self.right = None

        # Graph node attributes
        # self.greater: a dictionary mapping columns greater than
        # or equal to this one to a True or False to whether it is
        # [ (True) or ( (False)
        # self.smaller: a dictionary mapping columns smaller than
        # or equal to this one to a True or False to whether it is
        # [ (True) or ( (False)
        # constrained to be at least as small as this one
        # self.columns: the columns constrained to be equal contained
        # within this node
        # self.col_set: the names of the columns in self.columns
        self.type = c.ctype in ('DATE', 'INTEGER')
        self.greater = {}
        self.smaller = {}
        self.max = c.max
        self.max_e = True
        self.min = c.min
        self.min_e = True
        if debug:
            print(f"Max for column {c.name}: {c.max}")
            print(f"Min for column {c.name}: {c.min}")
        self.columns = [c.name]
        self.name = c.name
        self.col_set = {c.name}

    def combine(self, other, debug=False):
        # Merge two ColumnNodes.  Return False if it raises a contradiction, else return True
        self.type |= other.type

        # Remove from linked list
        if other.left is not None:
            other.left.right = other.right
        if other.right is not None:
            other.right.left = other.left

        all_columns = set(sum([self.columns, other.columns] + [list(d.keys()) for d in (
            self.greater,
            other.greater,
            self.smaller,
            other.smaller
        )], []))
        for c in all_columns:
            if c in self.columns:
                if c in other.smaller:
                    if not other.smaller[c]:
                        return False
                    del other.smaller[c]
                    del self.greater[other.name]
                if c in other.greater:
                    if not other.greater[c]:
                        return False
                    del other.greater[c]
                    del self.smaller[other.name]
            elif c in other.columns:
                self.columns.append(c)
                self.col_set.add(c)
                self.col_index[c] = self
            elif c in other.greater:
                if c in self.greater:
                    # Only still [ if both are [
                    self.greater[c] &= other.greater[c]
                    del other.greater[c]
                    self.col_index[c].smaller[self.name] = self.smaller[c]
                    del self.col_index[c].smaller[other.name]
                else:
                    self.greater[c] = other.greater[c]
            elif c in other.smaller:
                if c in self.smaller:
                    # Only still [ if both are [
                    self.smaller[c] &= other.smaller[c]
                    del other.smaller[c]
                    self.col_index[c].greater[self.name] = self.smaller[c]
                    del self.col_index[c].greater[other.name]
                else:
                    self.smaller[c] = other.smaller[c]
        return True

    def set_min(self, n, e):
        self.min = n
        self.min_e = e
        
    def set_max(self, n, e):
        self.max = n
        self.max_e = e

    def add_greater(self, c, e):
        self.greater[c] = e

    def add_smaller(self, c, e):
        self.smaller[c] = e

    def set_left(self, other):
        # Only call this when len(self.smaller.keys()) = 0
        self.left = other

    def set_right(self, other):
        # Only call this when len(self.smaller.keys()) = 0
        self.right = other

    def remove(self):
        # Only call this when len(self.smaller.keys()) = 0
        for c in self.greater.keys():
            node = self.col_index[c]
            e = self.greater[c]
            if node.min == self.min:
                node.set_min(self.min, (self.min_e & node.min_e & e))
            elif node.min < self.min:
                node.set_min(self.min, (self.min_e & e))
            # Remove every associated node
            for col in self.columns:
                if col in node.smaller:
                    del node.smaller[col]


def num_intersect(preds, debug=False):
    """
    Checks whether a list of numeric predicates can all be satisfied
    :param preds: A list of numeric predicates
    :param debug: Whether to be in debug mode (additional print statements)
    :return: Whether there exists a point where they intersect
    """
    # This index keeps track of which column names point to which node objects
    # This is subject to change, so every ColumnNode needs to have access to this
    index = {}

    # Hopefully the code should be good enough that the order of adding nodes here doesn't matter
    for pred in preds:
        if pred.column.name not in index:
            index[pred.column.name] = ColumnNode(pred.column, index, debug)
        if pred.comparative:
            if pred.col2.name not in index:
                index[pred.col2.name] = ColumnNode(pred.col2, index, debug)
            if pred.op.symbol == '=':
                if not index[pred.column.name].combine(index[pred.col2.name], debug):
                    return False
            else:
                e = pred.op.symbol in ('<=', '>=')
                is_greater = pred.op.symbol in ('>', '>=')
                greater = pred.column.name if is_greater else pred.col2.name
                smaller = pred.col2.name if is_greater else pred.column.name
                index[greater].add_smaller(smaller, e)
                index[smaller].add_greater(greater, e)
        else:
            if pred.op.symbol == '=':
                index[pred.column.name].set_min(pred.num, True)
                index[pred.column.name].set_max(pred.num, True)
            else:
                e = pred.op.symbol in ('<=', '>=')
                if pred.op.symbol in ('>', '>='):
                    index[pred.column.name].set_min(pred.num, e)
                else:
                    index[pred.column.name].set_max(pred.num, e)

    # Iterate through everything
    zeroes_list_top = None
    active_set = set()
    for col_name in index.keys():
        active_set.add(index[col_name].name)
    for cname in active_set:
        if len(index[cname].smaller.keys()) == 0:
            index[cname].left = zeroes_list_top
            if zeroes_list_top is not None:
                zeroes_list_top.right = index[cname]
            zeroes_list_top = index[cname]
    while len(active_set) > 0:
        # for c in active_set:
        #     print("Active set:")
        #     i = index[c]
        #     print(i.name, i.smaller)
        if zeroes_list_top is not None:
            changed_set = set(index[c].name for c in zeroes_list_top.greater.keys())
            zeroes_list_top.remove()
            active_set.remove(zeroes_list_top.name)
            zeroes_list_top = zeroes_list_top.left
            if zeroes_list_top is not None:
                zeroes_list_top.right = None
            for cname in changed_set:
                if len(index[cname].smaller.keys()) == 0:
                    index[cname].left = zeroes_list_top
                    if zeroes_list_top is not None:
                        zeroes_list_top.right = index[cname]
                    zeroes_list_top = index[cname]
        elif len(active_set) > 0:
            # There's a cycle.  Find it
            cname = None
            found_set = set()
            for cname in active_set:
                break
            found_list = [cname]
            cname = index[cname].name
            while cname not in found_set:
                found_set.add(cname)
                col = index[cname]
                cname = None
                for cname in col.smaller.keys():
                    break
                cname = index[cname].name
                found_list.append(cname)

            # We now have a path leading to a cycle.  Extract the cycle
            cycle_begun = False
            all_e = True
            prev = None
            root = None
            cycle_nodes = []
            for cname in found_list:
                if cycle_begun:
                    current = index[cname]
                    for cname2 in current.columns:
                        if cname2 in prev.smaller:
                            all_e &= prev.smaller[cname2]
                    prev = current
                    if prev.name != root.name:
                        cycle_nodes.append(prev)
                else:
                    if cname == found_list[-1]:
                        cycle_begun = True
                        prev = index[cname]
                        root = index[cname]
            if not all_e:
                if debug:
                    print("A cycle was found, and it did not consist entirely of less than or equals")
                return False
            for node in cycle_nodes:
                root.combine(node, debug)
                active_set.remove(node.name)

            # Look for zeroes again
            for cname in active_set:
                if len(index[cname].smaller.keys()) == 0:
                    if zeroes_list_top is not None:
                        zeroes_list_top.right = index[cname]
                    index[cname].left = zeroes_list_top
                    zeroes_list_top = index[cname]

    # It's over.  Breathe.  Check if things match.
    for col in index.keys():
        c = index[col]
        if c.min > c.max:
            if debug:
                print(f"Column {col} is constrained to have a max of {c.max} and a min of {c.min}")
            return False
        if (c.min == c.max) and not (c.min_e and c.max_e):
            if debug:
                print(f"Column {col} is constrained to have a max of {c.max} and a min of {c.min}")
            return False
    return True


def pred_gen(pred_string, table):
    """
    :param pred_string: a string of the form "column_name operator value"
    :param table: an instance of the table class
    :return: a predicate based on the string
    """
    # print(pred_string)
    col_name, op_name, value_name = pred_string.split(" ", 2)
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
    elif value_name.replace('.', '', 1).isdigit() or (
            value_name[1:].replace('.', '', 1).isdigit() and value_name[0] == '-'):
        # Instance of a numerical (non-date) predicate
        num = int(value_name) if value_name.isdigit() else float(value_name)
        assert column.numerical, "This is not a numerical column, so it cannot be compared with a number"
        return Numerical(op, column, num)
    elif (list([len(s) for s in value_name[:10].split('-')]) == [4, 2, 2]) and value_name[:10].replace('-', '').isdigit():
        num = datetime64(value_name)
        assert column.numerical, "This is not a numerical column, so it cannot be compared with a number"
        return Numerical(op, column, num)
    elif ((value_name[0] == '(') and (value_name[-1] == ')')) or ((value_name[0] == '{') and (value_name[-1] == '}')):
        # Instance of a categorical predicate
        values_list = value_name[1:-1].replace(', ', ',').split(',')
        values = set()
        for v in values_list:
            if v[0] == v[-1] == "'":
                values.add(v[1:-1])
            else:
                values.add(v)
        return Categorical(op, column, values)
    elif (op.symbol == '=') and (value_name[0] == "'") and (value_name[-1] == "'"):
        values = set()
        values.add(value_name[1:-1])
        return Categorical(Operator('IN'), column, values)
    else:
        raise Exception("Something's wrong")
