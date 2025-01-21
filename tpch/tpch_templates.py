from qd.qd_template import Template, VariablePredicate, var_pred_gen
from qd.qd_table import table_gen, Column
from qd.qd_predicate_subclasses import pred_gen
from qd.qd_query import Workload
from numpy import random
import numpy as np


table = table_gen('/home/ysragow/dsg/tpch/tpch_data/tpch.parquet')

containers_1 = ["SM", "LG", "MED", "JUMBO", "WRAP"]
containers_2 = ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"]
modes = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]
segments = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
type_1 = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
type_2 = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
type_3 = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
n_name = ["ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA",
            "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO",
            "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM",
            "UNITED STATES"]
p_name = ["almond", "antique", "aquamarine", "azure", "beige", "bisque",
            "black", "blanched", "blue",
            "blush", "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral",
            "cornflower", "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick",
            "floral", "forest", "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew",
            "hot", "indian", "ivory", "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen",
            "magenta", "maroon", "medium", "metallic", "midnight", "mint", "misty", "moccasin", "navajo",
            "navy", "olive", "orange", "orchid", "pale", "papaya", "peach", "peru", "pink", "plum", "powder",
            "puff", "purple", "red", "rose", "rosy", "royal", "saddle", "salmon", "sandy", "seashell", "sienna",
            "sky", "slate", "smoke", "snow", "spring", "steel", "tan", "thistle", "tomato", "turquoise", "violet",
            "wheat", "white", "yellow"]
r_name = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]


def f_1():
    return {'l_shipdate <= ?': np.datetime64('1998-12-01') - random.randint(60, 120 + 1)}
v_1_1 = var_pred_gen('l_shipdate <= ?', table)
t_1 = Template([], [v_1_1], f_1, table)


def f_10():
    year = random.randint(1993, 1995 + 1)
    month = random.randint((2 if year == 1993 else 1), (1 + 1 if year == 1995 else 12 + 1))
    date_str = str(year) + '-' + ('0' if (month < 10) else '') + str(month) + '-01'
    date = np.datetime64(date_str)
    upper_month = (month - 9) if (month > 9) else (month + 3)
    upper_year = (year + 1) if (month > 9) else year
    # print('Months:', month, upper_month)
    # print('Years:', year, upper_year)
    upper_date_str = str(upper_year) + '-' + ('0' if upper_month < 10 else '') + str(upper_month) + '-01'
    upper_date = np.datetime64(upper_date_str)
    # print('Lower:', date_str, date)
    # print('Upper:', upper_date_str, upper_date)
    return {'o_orderdate >= ?': date, 'o_orderdate < ?': upper_date}
s_10_1 = pred_gen("l_returnflag = 'R'", table)
v_10_1 = var_pred_gen('o_orderdate >= ?', table)
v_10_2 = var_pred_gen('o_orderdate < ?', table)
t_10 = Template([s_10_1], [v_10_1, v_10_2], f_10, table)


def f_11(): return {}
s_11_1 = pred_gen("n_name = 'ETHIOPIA'", table)
t_11 = Template([s_11_1], [], f_11, table)


def f_12():
    shipmode = (1,1)
    while shipmode[0] == shipmode[1]:
        shipmode = random.choice(modes, 2)
    shipmode = set([str(m) for m in shipmode])
    year = np.random.randint(1993, 1997 + 1)
    low_year, up_year = [np.datetime64(str(year + i) + '-01-01') for i in range(2)]
    return {
        'l_shipmode IN ?': shipmode,
        'l_receiptdate >= ?': low_year,
        'l_receiptdate < ?': up_year,
    }
v_12_1 = var_pred_gen('l_shipmode IN ?', table)
v_12_2 = var_pred_gen('l_receiptdate >= ?', table)
v_12_3 = var_pred_gen('l_receiptdate < ?', table)
s_12_1 = pred_gen('l_commitdate < l_receiptdate', table)
s_12_2 = pred_gen('l_commitdate > l_shipdate', table)
t_12 = Template([s_12_1, s_12_2], [v_12_1, v_12_2, v_12_3], f_12, table)


def f_3():
    day_offset = random.randint(0,31)
    date = np.datetime64('1995-03-01') + day_offset
    seg = {str(random.choice(segments))}
    return {'c_mktsegment = ?': seg, 'o_orderdate < ?': date, 'l_shipdate > ?': date}
v_3_1 = var_pred_gen('c_mktsegment = ?', table)
v_3_2 = var_pred_gen('o_orderdate < ?', table)
v_3_3 = var_pred_gen('l_shipdate > ?', table)
t_3 = Template([], [v_3_1, v_3_2, v_3_3], f_3, table)


def f_4():
    year = random.randint(1993, 1997 + 1)
    month = random.randint(1, 10 + 1)
    date_str = str(year) + '-' + ('0' if (month < 10) else '') + str(month) + '-01'
    date = np.datetime64(date_str)
    upper_month = (month - 9) if (month > 9) else (month + 3)
    upper_year = (year + 1) if (month > 9) else year
    upper_date_str = str(upper_year) + '-' + ('0' if upper_month < 10 else '') + str(upper_month) + '-01'
    upper_date = np.datetime64(upper_date_str)
    return {'o_orderdate >= ?': date, 'o_orderdate < ?': upper_date}
v_4_1 = var_pred_gen('o_orderdate >= ?', table)
v_4_2 = var_pred_gen('o_orderdate < ?', table)
s_4_1 = pred_gen('l_commitdate < l_receiptdate', table)
t_4 = Template([s_4_1], [v_4_1, v_4_2], f_4, table)


def f_6():
    year = random.randint(1993, 1997 + 1)
    lower_date_str = f"{year}-01-01"
    lower_date = np.datetime64(lower_date_str)
    upper_date_str = f"{year + 1}-01-01"
    upper_date = np.datetime64(upper_date_str)
    discount = 0.01 * random.randint(2, 9 + 1)
    quantity = random.randint(24, 25 + 1)
    return {'l_shipdate >= ?': lower_date,
            'l_shipdate < ?': upper_date,
            'l_discount <= ?': discount + 0.01,
            'l_discount >= ?': discount - 0.01,
            'l_quantity < ?': quantity}
v_6_1 = var_pred_gen('l_shipdate >= ?', table)
v_6_2 = var_pred_gen('l_shipdate < ?', table)
v_6_3 = var_pred_gen('l_discount <= ?', table)
v_6_4 = var_pred_gen('l_discount >= ?', table)
v_6_5 = var_pred_gen('l_quantity < ?', table)
t_6 = Template([], [v_6_1, v_6_2, v_6_3, v_6_4, v_6_5], f_6, table)




workload = Workload(sum([list([t() for _ in range(100)]) for t in [t_1, t_10, t_12, t_3, t_4]], []))
