import numpy as np
from qd_query import Query, Workload, Predicate


def tpch_workload_gen(counts):
    """
    Generate a tpc-h workload
    :param counts: Dictionary mapping tpc-h template names to a number requested
    :return: A Workload object containing the relevant queries
    """
    pass