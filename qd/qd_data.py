import sqlite3

class Column:
    num = 0
    numerical = True
    def __init__(self, num, numerical):
        """
        :param num: integer representing the index of the column;
        :param numerical: whether the column is numerical or categorical
        """
        self.num = num
        self.numerical = numerical

    def __eq__(self, other):
        """
        :param other: a different column
        :return: whether they represent the same column
        """
        return self.num == other.num



