operations = [
    lambda a, b: a > b,
    lambda a, b: a < b,
    lambda a, b: a >= b,
    lambda a, b: a <= b,
    lambda a, b: a == b,
    lambda a, b: any([a[i] & b[i] for i in range(len(a))]),
]


class Operator:
    """
        General Operator Class
        - code: the operator code representing the operator
        Operator Codes:
        - 1: >
        - 2: <
        - 3: >=
        - 4: <=
        - 5: = (numerical)
        - 6: IN
    """
    def __init__(self, code):
        self.num = code
        self.func = operations[code - 1]

    def __eq__(self, other):
        return self.num == other.num

    def __call__(self, a, b):
        """
        :param a: first object acted on
        :param b: second object acted on
        :return: whether the operator on a and b is true or not
        """
        return self.func(a, b)


class Predicate:
    """
    General Predicate Class
    - op: operation the predicate is on
    - column: the column upon which this predicate acts
    - values: list of boolean values (for categorical only)
    - value: singular comparative value (for numerical only)
    """
    values = []
    num = 0
    col2 = None

    def __init__(self, op, column):
        self.op = op
        self.column = column

    def check_set(self, values):
        return False

    def check_num(self, num, op):
        return False

    def intersect(self, pred):
        """
        :param pred: the predicate with which to judge intersection
        :return: bool: whether the predicates intersect
        """
        if self.column != pred.column:
            return False
        elif self.op == 6:
            return self.check_set(pred.values)
        else:
            return self.check_num(pred.num, pred.op)

