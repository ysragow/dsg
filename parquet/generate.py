import pyarrow.parquet as pq
from warnings import filterwarnings
from os import path, mkdir
from parquet import generate_two_column
from params import size as p_size, partitions as p_partitions, name as p_name

from json import dump

filterwarnings('ignore')


def generate(name, size, partitions):
    # Generate the folder if it doesn't exist
    folder = '{}/{}'.format(name, partitions)
    if not path.exists(name):
        mkdir(name)
    if not path.exists(folder):
        mkdir(folder)

    # Generate the base data if it doesn't exist
    base = name + '/0.parquet'
    if not path.exists(base):
        print("Generating initial data...")
        generate_two_column(size, name, 1)
        print("Done")

    # Find the file size (NOTE: WILL NOT NECESSARILY HAVE ACCURATE NUMBER OF PARTITIONS)
    file_size = -((-size) // partitions)

    # Initialize the start and stop points
    start = 0
    stop = file_size
    starts = []

    # Loop through the data, reading and writing chunks as you go
    print("Writing data with {} partitions...".format(partitions))
    while start < size:
        file_path = '{}/{}.parquet'.format(folder, start)
        filters = [('A', '>=', start), ('A', '<', stop)]
        pq.write_table(pq.read_table(base, filters=filters), file_path)
        start += file_size
        stop += file_size
        starts.append(start)
    print("Done")

    # Save the indices as a json
    with open(folder + '/index.json', 'w') as file:
        dump(starts, file)


if __name__ == '__main__':
    for partition_count in p_partitions:
        generate(p_name, p_size, partition_count)
