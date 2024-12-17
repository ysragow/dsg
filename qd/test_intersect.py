from qd.qd_predicate_subclasses import pred_gen, intersect
from qd.qd_table import table_gen

t = table_gen('../parquet/data/0.parquet')

pred_strs = ['A >= 190000000', 'A < 200000000', 'A >= 100000000', 'A < 990000000', 'A >= 870000000']
preds = list([pred_gen(p, t) for p in pred_strs])

print(intersect(preds))