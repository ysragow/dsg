operations = {
    '>': lambda a, b: a > b,
    '<': lambda a, b: a < b,
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    '=': lambda a, b: a == b,
    'IN': lambda a, b: any([(i in b) for i in a]),
    '!IN': lambda a, b: not any([(i in b) for i in a])
}

operations_opposites = {
    '>': '<=',
    '<': '>=',
    '>=': '<',
    '<=': '>',
    'IN': '!IN',
    '!IN': 'IN'
}


class Operator:
    """
        General Operator Class
        - code: the string representing the operator
        Operators: >, <, >=, <=, = (numerical), IN, !IN
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.func = operations[symbol]

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol

    def __call__(self, a, b):
        """
        :param a: first object acted on
        :param b: second object acted on
        :return: whether the operator on a and b is true or not
        """
        try:
            return self.func(a, b)
        except TypeError:
            raise Exception("Cannot compare: " + str(a) + " " + self.symbol + " " + str(b))

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

    def __init__(self, op, column, value):
        self.op = op
        self.column = column
        self.value = value
        self.comparative = False
        self.str_right = ''
        if (op.symbol in ('IN', '!IN')) == column.numerical:
            print('Column name:', column.name)
            print('Predicate symbol:', op.symbol)
            print('Column type:', column.ctype)
            print('Column is numerical:', column.numerical)
            assert False, "This operation cannot be used on this column"

    def to_dnf(self):
        """
        :return: this predicate as a DNF expression
        """
        return self.column.name, self.op.symbol, self.value

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

    def __str__(self):
        return str(self.column) + ' ' + str(self.op) + ' ' + self.str_right

    def __repr__(self):
        return str(self)

