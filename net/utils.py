import numpy as np

def make_sets(data, labels, props, shuffle=True):
    
    if not np.isclose(sum(props), 1):
        raise ValueError("proportions must sum to 1")
    
    m = data.shape[0]
    idx = np.random.permutation(m) if shuffle else range(m)
    
    sets = []
    cursor = 0
    
    for p in props:
        dset = data[idx, :][cursor:(cursor+int(p*m)), :]
        sets.append(dset)
        lset = labels[idx][cursor:(cursor+int(p*m))]
        sets.append(lset)
        
        cursor = int(p*m)
    
    return sets