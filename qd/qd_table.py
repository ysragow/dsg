from qd.qd_column import Column
from qd.qd_predicate import Predicate
import csv
import os


class Table:
    """
    Represents a table in a database
    IMMUTABLE
    """
    def __init__(self, tname, size=None, columns=None):
        """
        :param tname: name of the table.  does not include .csv
        """
        self.name = tname
        self.columns = {}
        if columns is not None:
            self.column_list = columns
            self.size = 0
            for i in range(len(columns)):
                self.columns[columns[i]] = Column(columns[i], i, 'REAL')
            return
        with open('data/' + tname + '.csv', 'r') as file:
            data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            columns = data.__next__()
            if size is None:
                size = 0
                for _ in data:
                    size += 1
            self.size = size

        # assume for now that all data is floats!
        self.column_list = columns
        for i in range(len(columns)):
            self.columns[columns[i]] = Column(columns[i], i, 'REAL')

    def delete(self):
        os.remove('data/' + self.name + '.csv')
        # print('deleting ' + self.name)

    def info(self):
        return str(self.columns)

    def get_column(self, column):
        return self.columns.get(column, None)

    def list_columns(self):
        """
        :return: a list containing the name of every column, in order
        """
        return self.column_list.copy()

    def split(self, pred):
        file = open('data/' + self.name + '.csv', 'r')
        file_0 = open('data/' + self.name + '0.csv', 'w')
        file_1 = open('data/' + self.name + '1.csv', 'w')
        data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        data_0 = csv.writer(file_0, quoting=csv.QUOTE_NONNUMERIC)
        data_1 = csv.writer(file_1, quoting=csv.QUOTE_NONNUMERIC)
        data.__next__()
        data_0.writerow(self.column_list)
        data_1.writerow(self.column_list)
        data_0_size = 0
        data_1_size = 0
        for row in data:
            if row in pred:
                data_0.writerow(row)
                data_0_size += 1
            else:
                data_1.writerow(row)
                data_1_size += 1
        file.close()
        file_0.close()
        file_1.close()
        return Table(self.name + '0', data_0_size), Table(self.name + '1', data_1_size)

    def get_boundaries(self):
        """
        Gets the boundaries of the data in this table
        :return: size, categorical, mins, maxes; where
        - categorical is a dictionary mapping from column numbers to sets of all items in that column
        - mins is a dictionary mapping from column numbers to the smallest number in that column
        - maxes is a dictionary mapping from column numbers to the largest number in that column
        """
        size = 0
        maxes = {}
        mins = {}
        categorical = {}
        numerical = {}
        with open('data/' + self.name + '.csv', 'r') as file:
            data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            data.__next__()
            row_1 = data.__next__()
            for name in self.column_list:
                col = self.columns[name]
                if col.numerical:
                    maxes[col.num] = row_1[col.num]
                    mins[col.num] = row_1[col.num]
                    numerical[col.num] = True
                else:
                    categorical[col.num] = {row_1[col.num]}
                    numerical[col.num] = False
            for row in data:
                size += 1
                for i in range(len(row)):
                    if numerical[i]:
                        maxes[i] = max(row[i], maxes[i])
                        mins[i] = min(row[i], mins[i])
                    else:
                        categorical[i].add(row[i])
        return categorical, mins, maxes
