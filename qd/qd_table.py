from qd.qd_column import Column
from qd.qd_predicate import Predicate
from fastparquet import ParquetFile, write as fp_write
from pandas import read_table, to_datetime, Int32Dtype, Int64Dtype
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
    def __init__(self, tname, size=None, columns=None, storage='parquet', folder='.', child_folder=None):
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
        if child_folder is None:
            self.child_folder = folder
        else:
            self.child_folder = child_folder
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
            # columns = data.columns
            self.size = data.count()
            # df = data.to_pandas()
            # self.categorical = {}
            # for c in columns:
            #     if not self.columns[c].numerical:
            #         # this is a categorical column
            #         self.categorical[c] = set([str(i) for i in df[c]])
            # del data
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
        file_str = '{}/{}{}.{}'.format(self.child_folder, self.name.split('/')[-1], '{}', self.storage)
        # print("Child Folder:", self.child_folder)
        columns = self.columns.copy()
        if self.storage == 'parquet':
            # data = ParquetFile(self.path)
            # if pred.comparative:
            #     pd = data.to_pandas().reset_index(drop=True)
            #     data_0 = pd[pred.op(pd[pred.column.name], pd[pred.col2.name])].reset_index(drop=True)
            #     data_1 = pd[pred.op.flip()(pd[pred.column.name], pd[pred.col2.name])].reset_index(drop=True)
            # else:
            #     data_0 = data.to_pandas(filters=[pred.to_dnf()], row_filter=True).reset_index(drop=True)
            #     data_1 = data.to_pandas(filters=[pred.flip().to_dnf()], row_filter=True).reset_index(drop=True)
            # data_0 = pa.Table.from_pandas(data_0)
            # data_1 = pa.Table.from_pandas(data_1)
            data_0 = pa.parquet.read_table(self.path, filters=pred.to_expression())
            data_1 = pa.parquet.read_table(self.path, filters=pred.flip().to_expression())
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
            _ = data.__next__()
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
        # split_path = self.path.split('.')
        # path_s = '.'.join(split_path[:-1])
        # storage = split_path[-1]
        # table1 = table_gen(path_s + '0.' + storage)
        # table2 = table_gen(path_s + '1.' + storage)
        table1 = table_gen(file_str.format(0))
        table2 = table_gen(file_str.format(1))
        print(f"Table 1 ({table1.path}) size:", table1.size)
        print(f"Table 2 ({table2.path}) size:", table2.size)
        print(f"Total table ({self.path}) size:", self.size)
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
        # categorical = self.categorical.copy() # pre-make this for efficiency
        categorical = {}
        numerical = {}
        if self.storage == 'parquet':
            data = ParquetFile(self.path)
            columns = data.columns
            for c in columns:
                if self.columns[c].numerical:
                    # this is a numeric column, so deal with it accordingly
                    maxes[c] = max(data.statistics['max'][c])
                    mins[c] = min(data.statistics['min'][c])
                    if maxes[c] is None:
                        assert mins[c] is None, "Something fishy is happening here..."
                        if self.columns[c].ctype == 'DATE':
                            maxes[c] = np.datetime64('1970-01-01')
                            mins[c] = np.datetime64('1970-01-01')
                        else:
                            maxes[c] = 0
                            mins[c] = 0
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
                        # categorical[col.num] = {row_1[col.num]}
                        numerical[col.num] = False
                for row in data:
                    size += 1
                    for i in range(len(row)):
                        if numerical[i]:
                            maxes[i] = max(row[i], maxes[i])
                            mins[i] = min(row[i], mins[i])
                        else:
                            pass
                            # categorical[i].add(row[i])
        return categorical, mins, maxes


def table_gen(path, child_folder=None):
    """
    Makes a table object from a path to a parquet file
    :param path: a path to a parquet file
    :param child_folder: the path to the folder containing the children of this table
    :return: a table object
    """
    file = ParquetFile(path)
    split_path = path.split('/')
    t_name = '.'.join(split_path[-1].split('.')[:-1])
    columns = {}
    for i in range(len(file.columns)):
        numerical = True
        c_name = file.columns[i]
        c_np_type = file.dtypes[c_name]
        if str(c_np_type) == 'object':
            c_type = 'STRING'
        elif str(c_np_type) in ('Int32', 'Int64'):
            print("Column {} is interpreted as type {} in table {}".format(c_name, c_np_type, t_name))
            c_type = 'STRING'
        elif np.issubdtype(c_np_type, int):
            c_type = 'INTEGER'
        elif np.issubdtype(c_np_type, np.number):
            c_type = 'FLOAT'
        elif np.issubdtype(c_np_type, np.datetime64):
            c_type = 'DATE'
        else:
            c_type = 'STRING'
            numerical = False
        if numerical:
            # Make a numerical column
            cmax = max(file.statistics['max'][c_name])
            if cmax is None:
                if c_type == 'DATE':
                    cmax = np.datetime64('1970-01-01')
                else:
                    cmax = 0
            cmin = min(file.statistics['min'][c_name])
            if cmin is None:
                if c_type == 'DATE':
                    cmin = np.datetime64('1970-01-01')
                else:
                    cmin = 0
            columns[c_name] = Column(c_name, i, c_type, cmax=cmax, cmin=cmin)
        else:
            # Don't worry about the values - CatComparatives seem not to exist in TPC-H
            # So just make a normal column for categoricals
            columns[c_name] = Column(c_name, i, c_type)
        # print("File path: ", path)
        # print("Column: ", c_name)
        # print("Type: {} ({}) ".format(c_np_type, c_np_type.__repr__()))
        # raise e

    # print(dict([(column.name, column.ctype) for column in columns.values()]))
    output = Table(t_name, columns=columns, storage='parquet', folder='/'.join(split_path[:-1]), child_folder=child_folder)
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
