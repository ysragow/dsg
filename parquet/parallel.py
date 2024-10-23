from multiprocessing import Pool, Process, Queue
from pyarrow import concat_tables as ar_concat, parquet
from time import time
from qd.qd_query import Query
from qd.qd_table import Table
from metaparams import read
from pandas import concat as pa_concat
from fastparquet import ParquetFile


def read_pq(file, filters=None):
    """
    Read a parquet file according to the metaparams
    :param file: Path to a file
    :param filters: Filters
    :return: Some kind of a table gained from filtering on the parquet file
    """
    if read == 'pyarrow':
        return parquet.read_table(file, filters=filters)
    elif read == 'fastparquet':
        # old code
        # return ParquetFile(file).to_pandas(filters=filters, row_filter=True).reset_index(drop=True)

        # new code (NOTE: DOES NOT FILTER BEYOND ROW GROUPS)
        return table_concat(list(ParquetFile(file).iter_row_groups(filters=filters)))
    else:
        raise Exception('Invalid value of read')


def query_to_filters(query):
    """
    :param query: A QD query object
    :return: That query transformed into a list of filters
    """
    return list(pred.to_dnf() for pred in query.list_preds())


def table_concat(tables):
    """
    Concatenates a list of Arrow tables, which may have 0 or 1 elements, into a list of tables with 0 or 1 elements
    :param tables: A list of Arrow tables
    :return: The concatenation of these two tables
    """
    if len(tables) < 2:
        return tables
    else:
        if read == 'pyarrow':
            return [ar_concat(tables)]
        elif read == 'fastparquet':
            return [pa_concat(tables)]
        else:
            raise Exception('Invalid value of read')


def filter_read(items):
    """
    Reads a file using filters.  This only exists because multiprocessing can't pickle non-global functions
    :param items: A list, containing a query object and a file name
    :return: the result of running the query on that file
    """
    filters = items[0]
    file = items[1]
    output = read_pq(file, filters=filters)
    return output


def filter_scan(items):
    """
    Reads a file using filters.  This only exists because multiprocessing can't pickle non-global functions
    :param items: A list, containing a query object and a file name
    :return: the result of running the query on that file
    """
    filters = items[0]
    file = items[1]
    output = read_pq(file, filters=filters)
    return output.shape[0]


def filter_queue(files, filters, q, q2=None, scan=False):
    """
    Reads a file using filters.  Adds the file to a queue.
    :param files: a list of files to be read
    :param filters: a list of filter
    :param q: A queue
    :param q2: A second queue, used for debugging
    :param scan: Whether to scan instead of read
    """
    for file in files:
        if q2 is not None:
            q2.put('beginning to read ' + file)
        output = read_pq(file, filters=filters)
        if q2 is not None:
            q2.put('finished reading ' + file)
        q.put(output.shape[0] if scan else output)


def parallel_read(filters, files, processes, scan=False, timestamps=False, verbose=False, verbose2=False):
    """
    Reads multiple files in parallel
    :param filters: A list of filters
    :param files: A list of parquet file names
    :param processes: The number of processes running at the same time
    :param scan: Whether to scan instead of reading
    :param timestamps: whether to return the total elapsed time
    :param verbose: whether to print things
    :param verbose2: whether to print even more things
    :return: A pyarrow table containing the contents of every file
    """
    start_time = time()
    q = Queue()
    q2 = None
    if verbose2:
        q2 = Queue()
    lists = [[filters, file] for file in files]
    if verbose2:
        print("Generating processes...")
    active_processes = {}
    tables = []
    sorted_files = [list() for _ in range(processes)]

    # Sort the files
    for i in range(len(lists)):
        sorted_files[i % processes].append(lists[i][1])

        # # Only add new ones when the list is not full
        # if verbose:
        #     print("in the for loop")
        # while len(active_processes.keys()) == processes:
        #     prune(active_processes)
        #     print_all(q2)

    # Make and start processes and add them to list of active processes
    for i in range(processes):
        process = Process(target=filter_queue, args=(sorted_files[i], filters, q), kwargs={'q2': q2, 'scan': scan}, name=str(i))

        # Add processes to list of active processes
        process.start()
        active_processes[process.name] = process

    # Wait for remaining processes to complete
    while active_processes:
        prune(active_processes)
        tables = flush(q, tables, scan=scan)
        print_all(q2)

    output = tables[0]

    # Timestamps and return
    end_query = time()
    end_concat = time()
    if verbose:
        num_rows = output if scan else output.shape[0]
        print('Parallel Read with {} processes scanned {} rows in {} seconds'.format(processes, num_rows, end_concat - start_time))
    if timestamps:
        if verbose2:
            print('\n')
            # print("Querying Time: ", end_query - start_time)
            print("Parallel Read")
            print("Processes: ", processes)
            print("Size: ", output if scan else output.shape)
            print("Total Time: ", end_concat - start_time)
            print('\n')
        return end_concat - start_time
    return output


def prune(active_processes):
    to_remove = []
    for key in active_processes.keys():
        if not active_processes[key].is_alive():
            to_remove.append(key)
    for key in to_remove:
        process = active_processes[key]
        # print("Pruning", process.name)
        process.close()
        del active_processes[key]


def flush(q, tables, scan=False):
    """
    Flush values from a queue to a list of tables
    :param q: A queue
    :param tables: A list of tables
    :param scan: Whether to scan instead of reading
    """
    while not q.empty():
        tables.append(q.get())
    return [sum(tables)] if scan else table_concat(tables)


def print_all(q):
    """
    Empty a queue and print everything on it
    :param q: A queue.  If none, nothing happens
    """
    if q is not None:
        while not q.empty():
            print(q.get())


def pooled_read(filters, files, processes, scan=False, timestamps=False, verbose=False, verbose2=False):
    """
    Reads multiple files in parallel
    :param filters: A list of filters
    :param files: A list of parquet file names
    :param processes: Number of  processes
    :param scan: Whether to scan instead of read
    :param timestamps: whether to print timestamps of how long the query takes
    :param verbose: whether to print things
    :param verbose2: whether to print even more things
    :return: A pyarrow table containing the contents of every file
    """
    start_time = time()
    items = [[filters, file] for file in files]
    f = filter_scan if scan else filter_read
    with Pool(processes) as p:
        output = p.map(f, items)
    end_query = time()
    output = sum(output) if scan else table_concat(output)[0]
    end_concat = time()
    if verbose:
        num_rows = output if scan else output.shape[0]
        end_concat = time()
        print('Pooled Read with {} processes scanned {} rows in {} seconds'.format(processes, num_rows, end_concat - start_time))
    if timestamps:
        if verbose2:
            print('\n')
            print("Pooled Read")
            print("Processes: ", processes)
            print("Size: ", output if scan else output.shape)
            print("Total Time: ", end_concat - start_time)
            print('\n')
        return end_concat - start_time
    return output


def regular_read(filters, files, scan=False, timestamps=False, verbose=False):
    """
    Sequentially read every provided file
    :param filters: A list of filters
    :param files: A list of files to read
    :param scan: Whether to scan instead of read
    :param timestamps: whether to print timestamps for reading
    :param verbose: whether to print things
    :return: All the data in the files, concatenated into one
    """
    output = []
    start_time = time()
    for file in files:
        table = read_pq(file, filters=filters)
        output.append(table.shape[0] if scan else table)
    output = sum(output) if scan else table_concat(output)[0]
    end_time = time()
    if verbose:
        num_rows = output if scan else output.shape[0]
        print("Regular Read scanned {} rows in {} seconds".format(num_rows, end_time - start_time))
    if timestamps:
        return end_time - start_time
    return output
