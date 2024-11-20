from params import queries, query_objects, name, partitions, layout
from time import time
from json import load, dump
from os import listdir
from qd.qd_algorithms import index as qd_index
from sys import argv
from glob import glob


def index(folder, query_bottom, query_top, timestamps=False, query_obj=None):
    start_time = time()

    assert layout in ("rgm", "qd", "index"), "Invalid layout"

    if layout == 'qd':
        assert query_obj is not None, "A query object is required to index qd trees"
        root_file = None
        potential_files = glob(folder + '/*.json')
        if len(potential_files) == 0:
            raise Exception("The folder " + folder + "does not have any jsons in it")
        for file in potential_files:
            if root_file[-11:] == '/files.json':
                pass
            else:
                root_file = file
        return qd_index(query_obj, root_file[:-4] + 'parquet', verbose=True)

    num_partitions = int(folder.split('/')[-1])

    # Binary search for smallest start less than or equal to bottom
    with open(folder + '/index.json', 'r') as file1:
        starts = load(file1)
    search_top = len(starts) - 1
    search_bottom = 0
    while search_top - search_bottom > 1:
        middle = (search_top + search_bottom) // 2
        if starts[middle] > query_bottom:
            search_top = middle
        else:
            search_bottom = middle

    # Get a list of valid files
    i = search_bottom
    output = []
    while starts[i] < query_top:
        name_to_index = '{}/{}'.format(folder, starts[i])
        if layout == 'rgm':
            # Row group mixing:
            output += ['{}/{}'.format(name_to_index, pfile) for pfile in listdir(name_to_index)]
        elif layout == 'index':
            output.append(name_to_index + '.parquet')
        i += 1
        if i == len(starts):
            break

    if timestamps:
        end_time = time()
        total_time = end_time - start_time
        print("With {} partitions, found {} matching files in {} seconds".format(num_partitions, len(output), total_time))
        with open(folder + '/index_time', 'w') as file1:
            file1.write(str(total_time))

    return output


def main():
    all_files = []
    local_name = name
    for i in range(len(queries)):
        query = queries[i]
        query_obj = query_objects[i]
        print(query)
        print(query_obj)
        q_files = {}
        if layout in ("rgm", "index"):
            q_bottom = query[0][2]
            q_top = query[1][2]
        else:
            q_bottom = 0
            q_top = 0

        # Process args
        if len(argv) == 2:
            local_name = argv[1]
        elif len(argv) == 3:
            q_bottom = int(argv[1])
            q_top = int(argv[2])
        elif len(argv) == 4:
            local_name = argv[1]
            q_bottom = int(argv[2])
            q_top = int(argv[3])
        # Index
        for partition_count in partitions:
            f = '{}/{}'.format(local_name, partition_count)
            q_files[partition_count] = index(f, q_bottom, q_top, timestamps=True, query_obj=query_obj)
        all_files.append(q_files)

    # Put data into files
    for partition_count in partitions:
        f = '{}/{}'.format(local_name, partition_count)
        with open(f + '/files.json', 'w') as file:
            dump([all_files[i][partition_count] for i in range(len(queries))], file)


if __name__ == '__main__':
    main()





