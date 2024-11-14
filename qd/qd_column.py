class Column:
    """
    Represents a column in a table
    IMMUTABLE
    """
    def __init__(self, name, num, ctype, cmax=None, cmin=None, cvalues=None):
        """
        :param name: name of the column;
        :param num: the number of this column
        :param ctype: type of the column
        """
        self.name = name
        self.num = num
        self.ctype = ctype
        self.numerical = ctype in ("REAL", "INTEGER", "FLOAT", "DATE")
        if self.numerical:
            assert cmax is not None, "Numerical columns need a cmax"
            assert cmin is not None, "Numerical columns need a cmin"
        # else:
        #     assert cvalues is not None, "Categorical columns need values"
        self.max = cmax
        self.min = cmin
        self.values = cvalues

    def __eq__(self, other):
        """
        :param other: a different column
        :return: whether they represent the same column
        """
        return self.name == other.name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()



