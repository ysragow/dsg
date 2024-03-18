from qd_predicate import Predicate, Operator

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
        assert(op.num > 4, "Wrong type of predicate")

    def check_set(self, values):
        """
        :param values: A boolean list of items from some category contained in the other predicate
        :return: whether that set intersects with the items in our predicate
        """
        return self.op(self.values, values)


class Numerical(Predicate):
    # A numerical predicate, using <, >, =>, or <=
    def __init__(self, op, column, num):
        """
        :param op: the operation this predicate is based upon.  must be in 1-4
        :param column: the column this predicate breaks on
        :param num: the number that we measure against the column value
        """
        super().__init__(op, column)
        self.num = num
        assert (op.num < 5, "Wrong type of predicate")

    def check_num(self, num, op):
        """
        :param num: number of the other predicate
        :param op: operation of the other predicate
        :return: whether the predicates intersect
        """
        return self.op(num, self.num) or op(self.num, num) or ((self.op == op) & (self.num == num))


class Comparative(Predicate):
    # A comparative predicate between two columns.  Only used in queries, never in nodes
    def __init__(self, op, col1, col2):
        """
        :param op: the code for the operation this predicate is based upon.  must be 5 or less
        :param col1: the column this predicate breaks on
        :param col2: the column this predicate breaks on
        """
        super().__init__(op, col1)
        self.col2 = col2
        assert (op.num < 6, "Wrong type of predicate")

