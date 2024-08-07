from qd_table import Table
from qd_node import Root
from qd_query import Workload
from qd_algorithms import dataset_gen, workload_gen, subset_gen, tree_gen, rank_fn_gen
from qd_approx import approx_1, approx_2, approx_3, approx_4a, approx_4b
from qd_predicate_subclasses import Numerical, pred_gen
from numpy import log2

k = 99999  # number of tuples in the dataset
m = 100  # minimum number of tuples in a partition
c = 2  # number of columns predicated on
tc = 10  # total number of columns
wmax = 10000 # maximum value that any tuple can take
wlt = 100  # size of the training workload
wl = 30  # size of the testing workload
sel = 0.02 # selectivity
s_list = (sel, )
compare_no_tree = True
L = log2(k) - log2(m)
expected_read_3 = approx_3(k, m, c, sel, wmax)
expected_read_2 = k*(2*c - (L % c)) / (2 * c * (2**(L//c))) - k*sel
assert(expected_read_2 == approx_2(k, m, c, sel, wmax))
expected_read_1 = k*(m/k)**(1/c)
assert(expected_read_1 == approx_1(k, m, c, sel, wmax))
print("Calculating 4a...")
expected_read_4a = approx_4a(k, m, c, sel, wmax) - k*sel
print("Done\n")
expected_read_4b = approx_4b(k, m, c, sel, wmax) - k*sel


columns = []
for i in range(c):
    columns.append("col" + str(i))
# t = Table('california_housing_train')
# pred = pred_gen('longitude < 114', t)
# print([113] in pred)
# print([115] in pred)
# t.split(pred)
print("Generating Dataset...")
t = dataset_gen('test_dataset_medium', num_columns=tc, num_points=k, max_value=wmax)
print("Done!")
# t = Table('test_dataset_small')
queries = []
for s in s_list:
    r = Root(t)
    w = workload_gen(r, wlt, selectivity=s, allowed_columns=columns)
    q_count = 0
    for q in w.queries:
        # print("\rQuery #{}".format(q_count))
        q_count += 1
        queries.append(q)
w = Workload(queries)
rank_fn = rank_fn_gen(m, multiply_sizes=True)
# print("Generating Tree...")
# tree = tree_gen(t, w, rank_fn, columns=columns)
# print("Done!")


# string_list = []
# for leaf_name in tree.leaves.keys():
#     leaf = tree.leaves[leaf_name]
#     string_list.append(leaf.name + '\n' + str(leaf.preds))
# with open('test/leaf_preds.txt', 'w') as file:
#     file.write('\n\n'.join(string_list))


for s in s_list:
    # w = workload_gen(r, wl, selectivity=s, allowed_columns=columns)
    # total_invalid_tree = 0
    # total_invalid_no_tree = 0
    # total_partitions = 0
    # for q in w.queries:
    #     # print(list([p.column.name + ' ' + p.op.symbol + ' ' + str(p.num) for p in q.list_preds()]))
    #     d1, c1, par = tree.get_data(q, use_tree=True, count_invalid_data=True, count_partitions=True, verbosity=1)
    #     if compare_no_tree:
    #         d0, c0 = tree.get_data(q, use_tree=False, count_invalid_data=True, count_partitions=False)
    #         total_invalid_no_tree += c0
    #         assert set([tuple(d) for d in d0]) == set([tuple(d) for d in d1]), "The query methods do not return the same data at query " + str(q)
    #     total_invalid_tree += c1
    #     total_partitions += par
    # print('\nSelectivity: ' + str(s))
    # if compare_no_tree:
    #     print("Total invalid data scanned without tree: " + str(total_invalid_no_tree))
    # print("Total invalid data scanned with tree: " + str(total_invalid_tree))
    # print("Total partitions accessed with tree: " + str(total_partitions))
    # print('Average invalid data accessed with tree: ' + str(total_invalid_tree/wl))
    print('Expected invalid (by formula 1): ' + str(expected_read_1))
    print('Expected invalid (by formula 2): ' + str(expected_read_2))
    print('Expected Invalid (by formula 3): ' + str(expected_read_3))
    print('Expected Invalid (by formula 4a): ' + str(expected_read_4a))
    print('Expected Invalid (by formula 4b): ' + str(expected_read_4b))
    # print('Expected/Average percent difference: ' + str(100*(1 - (total_invalid_tree/(wl*(expected_read_2 - k*s))))))
# tree.delete()

