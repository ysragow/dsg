from params import queries, query_objects, name, partitions, layout, table_path, verbosity_2, timestamps
from time import time
from json import load, dump
from os import listdir
from qd.qd_algorithms import index as qd_index, table_gen, reset, pred_gen, intersect
from pqd.algorithms import index as pqd_index
from fastparquet import ParquetFile
from sys import argv
# from glob import glob


def index(folder, query_bottom, query_top, timestamps=False, query_obj=None):
    start_time = time()

    assert layout in ("rgm", "qd", "index", "pqd"), "Invalid layout"

    num_partitions = folder.split('/')[-1]

    if layout in ('qd', 'pqd'):
        assert query_obj is not None, "A query object is required to index qd trees"
        # potential_files = glob(name + '/*.parquet')
        # assert len(potential_files) == 1, f"There should be exactly one parquet file in {name}"
        table = table_gen(table_path)
        query_obj = reset(table, query_obj)
        root_file = folder + '/' + '.'.join(table_path.split('/')[-1].split('.')[:-1]) + '.json'
        # root_file = None
        # potential_files = glob(folder + '/*.json')
        # if len(potential_files) == 0:
        #     raise Exception("The folder " + folder + "does not have any jsons in it")
        # for file in potential_files:
        #     if file[-11:] == '/files.json':
        #         pass
        #     else:
        #         root_file = file
        total_time = time()
        if layout == 'qd':
            output = qd_index(query_obj, root_file[:-4] + 'parquet', table, verbose=verbosity_2)
        else:
            assert layout == 'pqd'
            output = pqd_index(query_obj, root_file[:-4] + 'parquet', table)
        empty_files = []
        non_empty_files = []
        for file in output:
            stats = ParquetFile(file).statistics
            mins = stats['min']
            maxes = stats['max']
            empty = False
            for pred in query_obj.list_preds():
                if (not pred.comparative) and (pred.column.numerical):
                    if (pred.op.symbol in ('>', '=>', '=')) and (not pred.op(maxes[pred.column.name][0], pred.num)):
                        empty = True
                    elif (pred.op.symbol in ('<', '<=', '=')) and (not pred.op(mins[pred.column.name][0], pred.num)):
                        empty = True
            if empty:
                empty_files.append(file)
            else:
                non_empty_files.append(file)
        total_time = time() - total_time
        if timestamps:
            print(f"Query {query_obj} found {len(non_empty_files)} files in {num_partitions} in {total_time} seconds.") #, but it won't find anything in the following {len(empty_files)} files: {', '.join(empty_files)}.")
        return non_empty_files

    num_partitions = int(num_partitions)

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
            q_files[partition_count] = index(f, q_bottom, q_top, timestamps=timestamps, query_obj=query_obj)
        all_files.append(q_files)

    # Put data into files
    for partition_count in partitions:
        f = '{}/{}'.format(local_name, partition_count)
        with open(f + '/files.json', 'w') as file:
            dump([all_files[i][partition_count] for i in range(len(queries))], file)


if __name__ == '__main__':
    main()





