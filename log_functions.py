import numpy as np


def get_data(f, file_size):
    with open(f, 'r') as file:
        s = file.read()
    s = s.split('\n')[1:]
    s = [r.split(' ') for r in s]
    s = [(int(r[5]), int(r[8])//file_size, float(r[15])) for r in s]
    d =  [[] for _ in range(len(s)//10)]
    for r in s:
        d[r[1]-1].append(r[2])
    return d


def make_table(d):
    d1 = [' & '.join(f'{i + 1} Files') for i in range(len(d))]
    return '\\\\\n\\hline\n'.join(d1)


def regress(d):
    max_proc = max([0] + [len(r) for r in d])
    a_list = [[] for i in range(max_proc)]
    for i in range(len(d)):
        file_count = i + 1
        for j in range(len(d[i])):
            proc_count = j + 1
            a_list[j].append([file_count, -(-file_count // proc_count)])
    d = np.array(d)
    output = []
    for j in range(max_proc):
        proc_count = j + 1
        a = a_list[j]
        b = d[:,j:j+1]
        x, res, rank, s = np.linalg.lstsq(a, b)
        output.append(x.T[0])
        print(f"For {proc_count} processes, the least squares solution is {x}, with residual sum of squares {res.sum()}")
    return np.array(output)

def func_regress(b, func, ignore_1=False):
    a = np.array(list(map(func, range(2 if ignore_1 else 1, len(b) + 1))))
    if ignore_1:
        b = b[1:]
    x, res, rank, s = np.linalg.lstsq(a, b)
    print(f"The least squares solution is {x}, with residual sum of squares {res.sum()}")
    return x
    


    
            
    
