from sys import argv
from json import load


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

