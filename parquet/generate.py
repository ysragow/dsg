import pyarrow.parquet as pq
from warnings import filterwarnings
from os import path, mkdir
from parquet import generate_two_column
from params import size as p_size, partitions as p_partitions, name as p_name, write_processes
from time import time
from json import dump
from index import index
from parallel import regular_read
from multiprocessing import Pool
from sys import argv

filterwarnings('ignore')


def read_write(arg):
    query, sources, out_file = arg
    bound = query[1][2] - query[0][2]
    data = regular_read(query, sources)
    pq.write_table(data, out_file, row_group_size=(bound+1))


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
    file_size = -((-size) // partitions)

    # Initialize the start and stop points
    start = 0
    stop = file_size
    starts = []

    # Loop through the data, reading and writing chunks as you go
    print("Writing data with {} partitions...".format(partitions))
    start_time = time()
    process_tuples = []
    while start < size:
        file_path = '{}/{}.parquet'.format(folder, start)
        filters = [('A', '>=', start), ('A', '<', stop)]
        if source:
            files = index(source_folder, start, stop)
            process_tuples.append((filters, files, file_path))
        else:
            pq.write_table(pq.read_table(base, filters=filters), file_path)
        starts.append(start)
        start += file_size
        stop += file_size
    if source:
        with Pool(write_processes) as p:
            p.map(read_write, process_tuples)
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
