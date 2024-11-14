from qd.qd_predicate_subclasses import (
    Predicate,
    Operator,
    Numerical,
    Categorical,
    CatComparative,
    NumComparative,
    pred_gen,
    intersect,
)
from qd.qd_table import Table


class Query:
    # contains a list of predicates
    def __init__(self, predicates, table):
        """
        :param predicates: a list of predicates on the query
        :param table: a table object
        """
        self.table = table
        self.predicates = {}
        for column_name in table.columns.keys():
            self.predicates[column_name] = list()
        for predicate in predicates:
            self.predicates[predicate.column.name].append(predicate)
            # if predicate.comparative:
            #     self.predicates[predicate.col2.name].append(predicate)

    def list_preds(self):
        """
        :return: a list containing every predicate for this query
        """
        output = []
        for col in self.table.list_columns():
            for pred in self.predicates[col]:
                output.append(pred)
        return output

    def __contains__(self, item):
        """
        :param item: a row of data
        :return: whether every predicate of the query contains this item
        """
        output = True
        for pred in self.list_preds():
            output &= item in pred
        return output

    def __str__(self):
        return str(self.list_preds())

    def __repr__(self):
        return str(self)


class Workload:
    # contains a list of queries
    def __init__(self, queries):
        """
        :param queries: a list of queries
        """
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __str__(self):
        return str(self.queries)

    def __repr__(self):
        return str(self)

    def split(self, pred, prev_preds):
        """
        :param pred: the predicates upon which to split the workload
        :param prev_preds: the previous predicates in this workload
        :return: right_queries: a workload of queries matching the predicate
                 left_queries: a workload of queries matching the negation of the predicate
                 straddlers: a workload of queries matching both the predicate and its negation
        """
        left_queries = []
        right_queries = []
        straddlers = []
        neg_pred = pred.flip()
        for query in self.queries:
            left_true = False
            right_true = False
            # if pred.intersect(query.predicates):
            #     right_queries.append(query)
            #     right_true = True
            # if neg_pred.intersect(query.predicates):
            #     left_queries.append(query)
            #     left_true = True
            if intersect(query.list_preds() + prev_preds + [pred]):
                right_queries.append(query)
                right_true = True
            if intersect(query.list_preds() + prev_preds + [neg_pred]):
                left_queries.append(query)
                left_true = True
            if right_true & left_true:
                straddlers.append(query)
            if not (right_true or left_true):
                q1 = query.list_preds() + prev_preds + [pred]
                print(q1, intersect(q1))
                q2 = query.list_preds() + prev_preds + [neg_pred]
                print(q2, intersect(q2))
                raise Exception("Something is wrong with predicate check")

        return Workload(right_queries), Workload(left_queries), Workload(straddlers)

