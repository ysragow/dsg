from params import query, name, partitions
from time import time
from json import load, dump


def index(folder, query_bottom, query_top, timestamps=False):
    start_time = time()

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
        output.append('{}/{}.parquet'.format(folder, starts[i]))
        i += 1
        if i == len(starts):
            break

    if timestamps:
        end_time = time()
        total_time = end_time - start_time
        print("With {} partitions, found {} matching files in {} seconds".format(len(starts), len(output), total_time))
        with open(folder + '/index_time', 'w') as file1:
            file1.write(str(total_time))

    return output


if __name__ == '__main__':
    q_bottom = query[0][2]
    q_top = query[1][2]
    for partition_count in partitions:
        f = '{}/{}'.format(name, partition_count)
        files = index(f, q_bottom, q_top, timestamps=True)
        with open(f + '/files.json', 'w') as file:
            dump(files, file)





