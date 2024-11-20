from qd.qd_query import q_gen_const

# Overall params
name = 'data'

# Write params
write_processes = 10
size = 1000000
layout = 'rgm'
prev = 200

# Indexing params
q_gen = q_gen_const(name)
query_objects = [
    q_gen(["A >= 0", "A < 200000"]),
    q_gen(["A >= 0", "A < 1000000"]),
    q_gen(["A >= 0", "A < 10000"]),
]
queries = list([list([p.to_dnf() for p in q.list_preds()]) for q in query_objects])

# Querying params
scan = True
verbosity_2 = False
timestamps = True
processes = [3]
partitions = [8, 40, 200]
query_types = {'regular', 'pooled'}
