import pyarrow.parquet as pq
from warnings import filterwarnings
from os import path, mkdir
from parquet import generate_two_column
from params import size as p_size, partitions as p_partitions, name as p_name, write_processes, row_groups, queries
from time import time
from json import dump
from index import index
from parallel import regular_read, table_concat
from multiprocessing import Pool
from sys import argv

filterwarnings('ignore')


def nid(a, b):
    # Negative integer division.  Returns ceil(a / b)
    return -((-a) // b)


def make_query(start, stop):
    return [('A', '>=', start), ('A', '<', stop)]


def read_write(arg):
    query_list, sources, out_file, rg_size = arg
    sorted_column = pq.SortingColumn(0)
    if (row_groups is None) or (row_groups == 'none'):
        query = query_list[0]
        data = regular_read(query, sources)
        bound = query[1][2] - query[0][2]
        pq.write_table(data, out_file, row_group_size=(bound+1), sorting_columns=[sorted_column])
    elif row_groups == 'default':
        data = regular_read(query_list[0], sources)
        pq.write_table(data, out_file)
    elif row_groups == 'optimize':
        # print("Is this on?")
        sel = [(q[1][2] - q[0][2]) for q in queries]
        assert len(sel) == 2, "For now, this will only work if there are 2 queries"
        assert (sel[0] // sel[1]) == (sel[0] / sel[1]), 'The selectivity of the second query must divide the selectivity of the first query'
        data = []
        for query in query_list:
            # print("Reading query {} for file {}".format(query, out_file))
            data.append(regular_read(query, sources))
        data = table_concat(data)[0]
        print("Writing to file " + out_file)
        pq.write_table(data, out_file, row_group_size=rg_size, sorting_columns=[sorted_column])
    elif str(row_groups).isdigit():
        data = regular_read(query_list[0], sources)
        pq.write_table(data, out_file, row_group_size=int(row_groups), sorting_columns=[sorted_column])
    else:
        raise Exception("params.row_groups must be None, 'none', 'default', 'optimize', or an integer")


def generate(name, size, partitions, source=None):
    # Generate the folder if it doesn't exist
    folder = '{}/{}'.format(name, partitions)
    source_folder = name
    if source:
        source_folder = '{}/{}'.format(name, source)
    if not path.exists(name):
        mkdir(name)
    if not path.exists(folder):
        mkdir(folder)

    # Generate the base data if it doesn't exist
    base = name + '/0.parquet'
    if not path.exists(base):
        print("Generating initial data...")
        start_time = time()
        generate_two_column(size, name, 1)
        end_time = time()
        print("Done in {} seconds".format(end_time - start_time))

    # Find the file size (NOTE: WILL NOT NECESSARILY HAVE ACCURATE NUMBER OF PARTITIONS)
    file_size = nid(size, partitions)

    # Initialize the start and stop points
    start = 0
    stop = file_size
    starts = []

    # Loop through the data, reading and writing chunks as you go
    print("Writing data with {} partitions...".format(partitions))
    start_time = time()
    process_tuples = []
    if row_groups == 'optimize':
        # Selectivities statistics
        large_select = queries[0][1][2] - queries[0][0][2]
        small_select = queries[1][1][2] - queries[1][0][2]
        sel_ratio = large_select // small_select

        # Chunk statistics
        file_count = nid(large_select, file_size)
        chunk_size = file_size * file_count
        rg_size = nid(chunk_size, file_count * sel_ratio)
        # print("rg_size: ", rg_size)
        # print("chunk_size: ", chunk_size)
        # print("file_count: ", file_count)
        # print("sel_ratio: ", sel_ratio)

        # Initializations
        start = 0
        stop = rg_size
        all_filters = []

        # While loop through chunks
        while start < size:
            # print('Chunk loop, stop = {}, size = {}, chunk_size = {}'.format(stop, size, chunk_size))
            filters = []
            chunk_start = start
            starts.append(chunk_start)

            # While loop through row groups in a chunk
            while (stop <= (chunk_size + chunk_start)) and (stop <= size):
                # print('Row group loop, stop = {}'.format(stop))
                stop = min(chunk_start + chunk_size, stop)
                filters.append(make_query(start, stop))
                start = stop
                stop = start + rg_size
            all_filters.append(filters)
        for j in range(len(all_filters)):
            chunk_filters = all_filters[j]
            out_filters = []
            num_files = nid(len(chunk_filters * rg_size), file_size)
            file_path = '{}/{}'.format(folder, starts[j])
            if not path.exists(file_path):
                mkdir(file_path)
            for k in range(num_files):
                out_filters.append([])
            for k in range(len(chunk_filters)):
                # The layout schema
                out_filters[((k % sel_ratio) - (k // sel_ratio)) % num_files].append(chunk_filters[k])
            for k in range(len(out_filters)):
                query_list = out_filters[k]
                sources = set()
                if source:
                    for query in query_list:
                        sources = sources.union(set(index(source_folder, query[0][2], query[1][2])))
                else:
                    sources.add(name + '/0.parquet')
                process_tuples.append((query_list, sources, '{}/{}.parquet'.format(file_path, k), rg_size))

    else:
        while start < size:
            file_path = '{}/{}.parquet'.format(folder, start)
            filters = [make_query(start, stop)]
            files = index(source_folder, start, stop)
            process_tuples.append((filters, files, file_path, None))
            # if source:
            #     files = index(source_folder, start, stop)
            #     process_tuples.append((filters, files, file_path, None))
            # else:
            #     pq.write_table(pq.read_table(base, filters=filters[0]), file_path)
        starts.append(start)
        start += file_size
        stop += file_size
    if source:
        with Pool(write_processes) as p:
            p.map(read_write, process_tuples)
    else:
        for tup in process_tuples:
            read_write(tup)
    end_time = time()
    print("Done in {} seconds".format(end_time - start_time))

    # Save the indices as a json
    with open(folder + '/index.json', 'w') as file:
        dump(starts, file)


if __name__ == '__main__':
    # Process arguments
    if len(argv) == 3:
        p_name = argv[1]
        p_size = int(argv[2])
    elif len(argv) == 2:
        if argv[0].isdigit():
            p_size = int(argv[2])
        else:
            p_name = argv[1]

    # Generate files
    for i in range(len(p_partitions)):
        partition_count = p_partitions[i]
        if i > 0:
            prev_count = p_partitions[i-1]
            generate(p_name, p_size, partition_count, source=prev_count)
        else:
            generate(p_name, p_size, partition_count)
