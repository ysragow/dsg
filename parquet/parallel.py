from multiprocessing import Pool, Process, Queue
from pyarrow import concat_tables, parquet
from time import time
from qd.qd_query import Query
from qd.qd_table import Table


def table_concat(tables):
    """
    Concatenates a list of Arrow tables, which may have 0 or 1 elements, into a list of tables with 0 or 1 elements
    :param tables: A list of Arrow tables
    :return: The concatenation of these two tables
    """
    if len(tables) < 2:
        return tables
    else:
        return [concat_tables(tables)]


def filter_read(items):
    """
    Reads a file using filters.  This only exists because multiprocessing can't pickle non-global functions
    :param items: A list, containing a query object and a file name
    :return: the result of running the query on that file
    """
    filters = items[0]
    file = items[1]
    output = parquet.read_table(file, filters=filters)
    return output


def filter_queue(items, q, q2=None):
    """
    Reads a file using filters.  Adds the file to a queue.
    :param q: A queue
    :param q2: A second queue, used for debugging
    :param items: A list, containing a query object and a file name
    """
    filters = items[0]
    file = items[1]
    if q2 is not None:
        q2.put('beginning to read ' + file)
    output = parquet.read_table(file, filters=filters)
    if q2 is not None:
        q2.put('finished reading ' + file)
    q.put(output)


def parallel_read(query, files, processes, timestamps=False, verbose=False):
    """
    Reads multiple files in parallel
    :param query: A QD query object
    :param files: A list of parquet file names
    :param processes: A list of processes
    :param timestamps: whether to return the total elapsed time
    :param verbose: whether to print things
    :return: A pyarrow table containing the contents of every file
    """
    start_time = time()
    q = Queue()
    q2 = None
    if verbose:
        q2 = Queue()
    filters = list(pred.to_dnf() for pred in query.list_preds())
    lists = [[filters, file] for file in files]
    process_list = []
    if verbose:
        print("Generating processes...")
    for i in range(len(lists)):
        p = Process(target=filter_queue, args=(lists[i], q), kwargs={'q2': q2}, name=lists[i][1])
        process_list.append(p)
    if verbose:
        print("Running processes...")
    active_processes = {}
    tables = []

    # Start processes and add them to list of active processes
    for process in process_list:

        # Only add new ones when the list is not full
        if verbose:
            print("in the for loop")
        while len(active_processes.keys()) == processes:
            prune(active_processes)
            flush(q, tables)
            print_all(q2)

        # Add processes to list of active processes
        process.start()
        active_processes[process.name] = process

    # Wait for remaining processes to complete
    while active_processes:
        prune(active_processes)
        flush(q, tables)
        print_all(q2)

    output = tables[0]

    # Timestamps and return
    end_query = time()
    end_concat = time()
    if timestamps:
        print('\n')
        # print("Querying Time: ", end_query - start_time)
        print("Processes: ", processes)
        print("Size: ", output.shape)
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


def flush(q, tables):
    """
    Flush values from a queue to a list of tables
    :param q: A queue
    :param tables: A list of tables
    """
    while not q.empty():
        tables.append(q.get())
    table_concat(tables)


def print_all(q):
    """
    Empty a queue and print everything on it
    :param q: A queue.  If none, nothing happens
    """
    if q is not None:
        while not q.empty():
            print(q.get())


def pooled_read(query, files, processes, timestamps=False):
    """
    Reads multiple files in parallel
    :param query: A QD query object
    :param files: A list of parquet file names
    :param processes: A list of processes
    :param timestamps: whether to print timestamps of how long the query takes
    :return: A pyarrow table containing the contents of every file
    """
    start_time = time()
    filters = list(pred.to_dnf() for pred in query.list_preds())
    items = [[filters, file] for file in files]
    with Pool(processes) as p:
        output = p.map(filter_read, items)
    end_query = time()
    output = table_concat(output)[0]
    end_concat = time()
    if timestamps:
        print('\n')
        # print("Querying Time: ", end_query - start_time)
        print("Processes: ", processes)
        print("Size: ", output.shape)
        print("Total Time: ", end_concat - start_time)
        print('\n')
        return end_concat - start_time
    return output


def regular_read(query, files, timestamps=False):
    """
    Sequentially read every provided file
    :param query: A query
    :param files: A list of files to read
    :param timestamps: whether to print timestamps for reading
    :return: All the data in the files, concatenated into one
    """
    filters = list(pred.to_dnf() for pred in query.list_preds())
    output = []
    start_time = time()
    for file in files:
        output.append(parquet.read_table(file, filters=filters))
    output = table_concat(output)[0]
    if timestamps:
        print("Size: ", output.shape)
        print("Total Time: ", time() - start_time)
    return output
