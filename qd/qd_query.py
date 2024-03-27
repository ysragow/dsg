from qd_predicate_subclasses import Predicate, Operator, Comparative, Numerical, Categorical, pred_gen
from qd_data import Table

class Query:
    # contains a list of predicates
    def __init__(self, predicates, table):
        """
        :param predicates: a list of predicates on the query in the form of a string "column_name operator value"
        :param table: a table object
        """
        self.predicates = []
        for predicate in predicates:
            self.predicates.append(pred_gen(predicate, table))




