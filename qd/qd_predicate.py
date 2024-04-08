operations = {
    '>': lambda a, b: a > b,
    '<': lambda a, b: a < b,
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    '=': lambda a, b: a == b,
    'IN': lambda a, b: any([(i in b) for i in a]),
}

operations_opposites = {
    '>': '<=',
    '<': '>=',
    '>=': '<',
    '<=': '>',
}


class Operator:
    """
        General Operator Class
        - code: the string representing the operator
        Operators: >, <, >=, <=, = (numerical), IN
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.func = operations[symbol]

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __call__(self, a, b):
        """
        :param a: first object acted on
        :param b: second object acted on
        :return: whether the operator on a and b is true or not
        """
        return self.func(a, b)

    def flip(self):
        """
        :return: the opposite operation, or itself if this does not apply
        """
        if self.symbol in operations_opposites:
            return Operator(operations_opposites[self.symbol])
        return self


class Predicate:
    """
    General Predicate Class
    - op: operation the predicate is on
    - column: the column upon which this predicate acts

    This skeleton class will never be called.  It does not intersect any other predicate.
    """

    def __init__(self, op, column):
        self.op = op
        self.column = column
        self.comparative = False
        assert (op.symbol == 'IN') != column.numerical, "This operation cannot be used on this column"

    def __contains__(self, item):
        """
        :param item: an item to be tested against this parameter
        :return: whether this item is in this predicate
        """
        return False

    def intersect(self, preds):
        """
        :param preds: a dictionary mapping column names to predicates
        :return: bool: whether the predicates intersect
        """
        return False

    def flip(self):
        """
        :return: predicate: the inverse of this predicate
        """
        return None

