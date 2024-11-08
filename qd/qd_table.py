from qd.qd_column import Column
from qd.qd_predicate import Predicate
from fastparquet import ParquetFile, write as fp_write
from pandas import read_table, to_datetime
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import csv
import os


class Table:
    """
    Represents a table in a database
    IMMUTABLE
    """
    def __init__(self, tname, size=None, columns=None, storage='parquet', folder='.'):
        """
        :param tname: name of the table.  does not include .csv
        :param size: number of rows
        :param columns: dictionary mapping column names to column objects, or list of strings
        :param storage: method of storing the data ('parquet' or 'csv')
        :param folder: path from working directory to folder location.  Default: working directory
        """
        self.name = tname
        self.folder = folder
        self.path = folder + '/' + tname + '.' + storage
        self.columns = {}
        self.storage = storage
        if 'get' in dir(columns):
            # this means that columns is a dict or OrderedDict
            self.columns = columns
        elif columns is not None:
            self.column_list = columns
            self.size = 0
            for i in range(len(columns)):
                self.columns[columns[i]] = Column(columns[i], i, 'REAL')
            return
        if storage == 'csv':
            with open(self.path, 'r') as file:
                data = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
                columns = data.__next__()
                if size is None:
                    size = 0
                    for _ in data:
                        size += 1
                self.size = size
        elif storage == 'parquet':
            data = ParquetFile(self.path)
            columns = data.columns
            self.size = data.count
        else:
            raise Exception("Invalid storage.  Must be 'parquet' or 'csv'")

        # assume for now that all data is floats!
        self.column_list = list(self.columns.keys())
        # for i in range(len(columns)):
        #     self.columns[columns[i]] = Column(columns[i], i, 'REAL')

    def delete(self):
        os.remove(self.path)
        # print('deleting ' + self.name)

    def info(self):
        return str(self.columns)

    def get_column(self, column):
        return self.columns.get(column, None)

    def get_dtypes(self):
        return ParquetFile(self.path).dtypes

    def list_columns(self):
        """
        :return: a list containing the name of every column, in order
        """
        return self.column_list.copy()

    def split(self, pred):
        file_str = '{}/{}{}.{}'.format(self.folder, self.name, '{}', self.storage)
        if self.storage == 'parquet':
            data = ParquetFile(self.path)
            data_0 = data.to_pandas(filters=[pred.to_dnf()], row_filter=True).reset_index(drop=True)
            data_1 = data.to_pandas(filters=[pred.flip().to_dnf()], row_filter=True).reset_index(drop=True)
            data_0 = pa.Table.from_pandas(data_0)
            data_1 = pa.Table.from_pandas(data_1)
            data_0_size = data_0.num_rows
            data_1_size = data_1.num_rows
            pa.parquet.write_table(data_0, file_str.format(0))
            pa.parquet.write_table(data_1, file_str.format(1))
        elif self.storage == 'csv':
            file = open(self.path, 'r')
            file_0 = open(file_str.format(0), 'w')
            file_1 = open(file_str.format(1), 'w')
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
        else:
            raise Exception("invalid storage format")
        table1 = Table(self.name + '0', data_0_size, storage=self.storage, folder=self.folder)
        table2 = Table(self.name + '1', data_1_size, storage=self.storage, folder=self.folder)
        return table1, table2

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
        if self.storage == 'parquet':
            data = ParquetFile(self.path)
            columns = data.columns
            for c in columns:
                if np.issubdtype(data.dtypes[c], np.number):
                    # this is a numeric column, so deal with it accordingly
                    maxes[c] = max(data.statistics['max'][c])
                    mins[c] = min(data.statistics['min'][c])
                else:
                    # this is a string or categorical column
                    pass
        elif self.storage == 'csv':
            with open(self.path, 'r') as file:
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


def table_gen(path):
    """
    Makes a table object from a path to a parquet file
    :param path: a path to a parquet file
    :return: a table object
    """
    file = ParquetFile(path)
    split_path = path.split('/', 2)
    t_name = split_path[-1][:-8]
    columns = {}
    for i in range(len(file.columns)):
        c_name = file.columns[i]
        c_np_type = file.dtypes[c_name]
        if np.issubdtype(c_np_type, int):
            c_type = 'INTEGER'
        elif np.issubdtype(c_np_type, np.number):
            c_type = 'FLOAT'
        elif np.issubdtype(c_np_type, np.datetime64):
            c_type = 'DATE'
        else:
            c_type = 'STRING'
        columns[c_name] = Column(c_name, i, c_type)
    # print(dict([(column.name, column.ctype) for column in columns.values()]))
    output = Table(t_name, columns=columns, storage='parquet', folder='/'.join(split_path[:-1]))
    # print(dict([(column.name, column.ctype) for column in columns.values()]))
    return output


def tbl_to_parquet(file_path, column_names, dtypes):
    """
    Make a .parquet file out of a .tbl file and return a table object
    YOU CAN FIND ALL RELEVANT INFO AT ~/tpch-dbgen/dss.ddl
    :param file_path: the path to the .tbl file
    :param column_names: the names of each column of the .tbl file
    :param dtypes: a dict mapping columns to their data types, except for strings, which should be an integer
    :return: A table object for a parquet file
    """
    strs = {}
    dates = set()
    for c in column_names:
        if isinstance(dtypes[c], int):
            strs[c] = dtypes[c]
            dtypes[c] = str
        elif dtypes[c] == np.datetime64:
            dates.add(c)
            dtypes[c] = str
    dtypes[''] = np.float64
    column_names.append('')
    data = read_table(file_path, sep='|', header=None, names=column_names, dtype=dtypes)
    data.dropna(axis=1, how='all', inplace=True)
    for i in range(len(list(data.columns))):
        c = list(data.columns)[i]
        if c in strs:
            new_c = data.pop(c).astype("|S" + str(strs[c]))
            data.insert(i, c, new_c, False)
        if c in dates:
            new_c = to_datetime(data.pop(c))
            data.insert(i, c, new_c, False)
    # print(data)
    # print(data.dtypes)
    new_path = file_path[:-3] + 'parquet'
    # print(strs)
    # fp_write(new_path, data, fixed_text=strs)
    pa_data = pa.Table.from_pandas(data)
    # print(pa_data.schema)
    # print(pa_data.columns)
    pq.write_table(pa_data, new_path)
    return table_gen(new_path)
