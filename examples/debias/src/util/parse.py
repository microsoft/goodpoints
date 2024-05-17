import re
import math
import numpy as np


def parse_n_expr(text, n):
    if isinstance(text, int):
        return text
    if 'n' in text:
        text = text.replace('n', '(' + str(n) + ')')
    result = int(math.ceil(eval(text)))
    return result


def parse_n_m_expr(text, n, m):
    if isinstance(text, int):
        return text
    if 'n' in text:
        text = text.replace('n', '(' + str(n) + ')')
    if 'm' in text:
        text = text.replace('m', '(' + str(m) + ')')
    result = int(math.ceil(eval(text)))
    return result


def parse_n_m_dict(d, n, m,
                 keywords=['out_size', 'rank', 'num_iter', 'inflate_size']):
    d = d.copy()
    for k in keywords:
        if k in d:
            d[k] = parse_n_m_expr(d[k], n, m)
    return d
