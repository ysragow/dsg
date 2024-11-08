from qd.qd_query import Query, Operator
from qd.qd_column import Column
from qd.qd_predicate_subclasses import Numerical, Categorical


class VariablePredicate:
    # like a predicate, except the value is a distribution
    def __init__(self, op, column, pred_class, id):
        """
        :param op: the operator of the predicate
        :param column: the column of the predicate
        :param pred_class: which parameter subclass this should belong to
        :param id: a unique ID string for this var_pred
        """
        self.op = op
        self.column = column
        self.pred_class = pred_class
        self.id = id

    def __call__(self, value):
        return self.pred_class(self.op, self.column, (set(value) if self.pred_class == Categorical else value))


class Template:
    # contains a list of predicates, with some wildcard predicates and their distributions
    def __init__(self, static_preds, var_preds, value_distribution, table):
        """
        A template for queries
        :param static_preds: a list of Predicates
        :param var_preds: a list of VariablePredicates
        :param value_distribution: a function that, when called, outputs a dictionary mapping
        :param table: a table object
        var_pred ID strings to a list of values for that var_pred
        """
        self.static_preds = static_preds
        self.var_preds = var_preds
        self.dist = value_distribution
        self.table = table

    def __call__(self):
        dist = self.dist()
        all_preds = self.static_preds + list([pred(dist[pred.id]) for pred in self.var_preds])
        return Query(all_preds, self.table)


keywords = [
    'date',
    'year',
    'interval',
    'between',
    'and',
    'like',
    'or',
]


def var_pred_gen(pred_string, table):
    """
        NOTE: DOES NOT WORK FOR COMPARATIVE PREDICATES (yet).
        I will make a separate function for that if I need to
        :param pred_string: a string of the form "column_name operator value"
        :param table: an instance of the table class
        :return: a predicate based on the string
        """
    # print(pred_string)
    col_name, op_name, value_name = pred_string.split(" ", 2)
    column = table.get_column(col_name)
    assert column is not None, "The column " + col_name + " does not exist in this table."
    op = Operator(op_name)
    type = column.ctype
    # print(type)
    # print(table.get_dtypes()[column.name])
    if type in ('DATE', 'INTEGER', 'FLOAT'):
        return VariablePredicate(op, column, Numerical, pred_string)
    else:
        return VariablePredicate(Operator('IN'), column, Categorical, pred_string)


# def template_gen(preds, var_preds, table):
#     """
#     A list of strings representing the preds, and a list of strings representing the var preds
#     :param preds: a dictionary mapping predicate strings to data types of the predicate
#     :param var_preds: a dictionary mapping variable predicates strings with a ? denoting a variable,
#     to a tuple of the data type of the string, a boolean indicating whether or not it is comparative,
#     and a function outputting values for each variable
#     :param table: a table object
#     :return: a template
#     """
#     static_preds = [pred_gen(pred, table) for pred in preds.keys]
