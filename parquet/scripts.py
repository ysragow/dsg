from sys import argv
from json import load, dump
import query


def get_average(n, v=False):
    """
    Takes the average over n runs of query.py and writes it to query_times.json
    :param n: the number of times to run query.py
    """
    first = True
    data = None
    name = query.name
    print(name + '/query_times.json')
    for i in range(n):
        print('Querying...')
        with open('query.py', 'r') as q:
            query.main(v)
        print('Reading...')
        with open(name + '/query_times.json', 'r') as q:
            new_data = load(q)
        if first:
            data = new_data
            first = False
        else:
            for q_type in data.keys():
                q_data = data[q_type]
                nq_data = new_data[q_type]
                for num_part in q_data.keys():
                    if q_type == 'regular':
                        q_data[num_part] += nq_data[num_part]
                        # print('Total for {} with {} partitions: {}'.format(q_type, num_part, q_data[num_part]))
                        if i == (n - 1):
                            q_data[num_part] /= n
                    else:
                        p_data = q_data[num_part]
                        np_data = nq_data[num_part]
                        for num_proc in p_data.keys():
                            p_data[num_proc] += np_data[num_proc]
                            # print('Total for {} with {} partitions and {} processes: {}'.format(q_type, num_part, num_proc, p_data[num_proc]))
                            if i == (n - 1):
                                p_data[num_proc] /= n
    print('Writing')
    with open(name + '/query_times.json', 'w') as q:
        dump(data, q)


def get_min(j, path=None):
    best = []
    minimum = None
    if path is None:
        path = []
        with open(j, 'r') as file:
            data = load(file)
    else:
        data = j
    if type(data) == dict:
        for key in data.keys():
            sub_min, sub_best = get_min(data[key], path + [key])
            if minimum is None:
                minimum = sub_min
                best = sub_best
            elif sub_min < minimum:
                minimum = sub_min
                best = sub_best
        return minimum, best
    elif type(data) in (float, int):
        return data, path
    else:
        raise Exception('Invalid data. {} is not a dictionary or a number'.format(data))


if __name__ == '__main__':
    if len(argv) < 2:
        raise Exception("Which function do you want to call?")
    if argv[1] == 'min':
        if len(argv) != 3:
            raise Exception("This function takes 1 argument")
        print(get_min(argv[2]))
    if argv[1] == 'mean':
        if len(argv) not in (3, 4):
            raise Exception("This function takes 1 or 2 arguments argument")
        verbose = False
        if len(argv) == 4:
            if argv[3] == '-v':
                verbose = True
        get_average(int(argv[2]), verbose)


