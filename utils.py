def etl_calc(r, a):
    r.sort()
    tail = r[:int(a * len(r))]
    etl = tail.mean()
    return etl

def etr_calc(r, b):
    r.sort()
    tail = r[int(b * len(r)):]
    etr = tail.mean()
    return etr