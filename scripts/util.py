def unique(X):
    S = set()
    for x in X:
        if x in S:
            continue
        S.add(x)
        yield x

def get_si_scale(v):
    SCALES = [('T',1e12),('G',1e9),('M',1e6),('k',1e3)]
    for (prefix, scale) in SCALES:
        if v > scale: return (prefix, scale)
    return ('', 1)
