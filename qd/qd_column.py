class Column:
    """
    Represents a column in a table
    IMMUTABLE
    """
    def __init__(self, name, num, ctype):
        """
        :param name: name of the column;
        :param num: the number of this column
        :param ctype: type of the column
        """
        self.name = name
        self.num = num
        self.ctype = ctype
        self.numerical = ctype in ("REAL", "INTEGER", "FLOAT", "DATE")

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



