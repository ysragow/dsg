import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pyarrow as pa
from os import mkdir, path, remove
from qd.qd_query import Query, Numerical, Operator
from qd.qd_table import Table
from qd.qd_column import Column
from fastparquet import ParquetFile, write
from pandas import concat

# A = Column('A', 0, 'INTEGER')
# B = Column('B', 1, 'REAL')
ram_max = 1000000000 # The maximum number of rows the ram can handle in a python object

def nid(a, b):
    # Negative integer division.  Returns ceil(a / b)
    return -((-a) // b)


def generate_random_column(size, rounds=10):
    """
    :param size: size of output
    :param rounds: maximum rounds of generation
    :return: an awkwardly randomly generated array with maximum value 2^rounds and minimum value 1
    """
    output = np.ones(size)
    editor = np.ones(size)
    for _ in range(rounds):
        new = np.random.rand(size)
        output += output * editor * new
        editor = editor * np.random.randint(2, size=size)
    return output


def generate_basic_parquet(size, name, start, step):
    """
    Generate a parquet file according to specifications
    :param size: size of the file
    :param name: name of the file
    :param start: upper bound (non-inclusive) on values of column A
    :param step: how much the
    """
    stop = start + (step * size) - (step / 2)
    df = pd.DataFrame({'A': np.arange(start, stop, step),
                       'B': generate_random_column(size)})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, name)


def generate_two_column(size, name, partitions, minmax=None):
    """
    Generates set of two-column parquet files, along with an index of each file on the first column
    :param size: size of the data
    :param name: name of the folder containing the files this will be sent to
    :param minmax: a tuple of 2 numbers denoting the highest and lowest values.  If None, default to (0, size-1)
    :param partitions: number of files in output
    """
    assert size >= partitions
    regen_required = False
    if size / partitions > ram_max:
        regen_required = True
        regen_factor = nid((size // partitions), ram_max)
        init_partitions = partitions
        partitions *= regen_factor
    else:
        regen_factor = 1
        init_partitions = partitions
    print(f"Regen Factor: {regen_factor}")
    if minmax is None:
        step = 1
        start = 0
    else:
        step = size / (minmax[0] - minmax[1])
        start = minmax[0]
    index = ''
    if not path.exists(name):
        mkdir(name)
    par_size = size // partitions
    extras = size % partitions
    for i in range(partitions):
        # index += str(start) + ' '
        this_size = par_size + (i < extras)
        print(f"Generating partition {i}...")
        generate_basic_parquet(this_size, name + '/{}.parquet'.format(i), start, step)
        start += this_size * step
    # index += str(start)
    # with open(name + '/index.txt', 'w') as file:
    #     file.write(index)
    if regen_required:
        for i in range(init_partitions):
            print(f"Regenerating partition {i}...")
            base_num = i * regen_factor
            files_list = []
            for j in range(regen_factor):
                print(f"Loading dataframe for file {name}/{base_num + j}.parquet...", end="\r")
                files_list.append(ParquetFile(name + f'/{base_num + j}.parquet').to_pandas())
                print(f"Deleting {name}/{base_num + j}.parquet...", end="\r")
                remove(f"{name}/{base_num + j}.parquet")
            print("Concatenating files...", end="\r")
            total_file = concat(files_list)
            print("Writing...", end="\r")
            write(name + f"{i}.parquet", total_file)
            print("Done.")



def generate_query(s, upper, lower, bupper=None, blower=None):
    """
    Generate a random query
    :param s: the selectivity
    :param upper: upper limit on absolute range of column A
    :param lower: lower limit on absolute range of column A
    :param bupper: upper limit on predicate range of column B; no predicate if None
    :param blower: lower limit on predicate range of column B; no predicate if None
    :return: a query object
    """
    bottom = lower + np.random.uniform() * (1 - s) * (upper - lower)
    top = bottom + s * (upper - lower)
    preds = [Numerical(Operator('<'), A, top), Numerical(Operator('>='), A, bottom)]
    if bupper is not None:
        preds.append(Numerical(Operator('<'), B, bupper))
    if blower is not None:
        preds.append(Numerical(Operator('>='), B, blower))
    return Query(preds, Table(' ', columns=['A', 'B']))


def get_matching_files(query, name):
    """
    Get the files matching this query from a (rudimentary) parquet index
    :param query: the query we are looking to match
    :param name: the name of the folder containing the parquet files
    :return: a list of file names
    """
    with open(name + '/index.txt', 'r') as file:
        index = file.read()
    indices = [int(i) for i in index.split(' ')].__iter__()
    i_prev = indices.__next__()
    count = 0
    output = []
    for i in indices:
        pred_1 = Numerical(Operator('>='), A, i_prev)
        pred_2 = Numerical(Operator('<'), A, i)
        if pred_1.intersect(query.predicates) and pred_2.intersect(query.predicates):
            output.append('{}/{}.parquet'.format(name, count))
        count += 1
        i_prev = i
    return output
