import numpy as np, json
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def simple_precision_recall(gt_visible, det_list, tol=2.5):
    tp=fp=fn=0
    for g,d in zip(gt_visible, det_list):
        if d is None and g is None: continue
        if g is not None and d is not None:
            if np.linalg.norm(np.array(g)-np.array(d)) <= tol: tp+=1
            else: fp+=1
        elif d is None and g is not None: fn+=1
        elif d is not None and g is None: fp+=1
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return {'precision': float(prec), 'recall': float(rec), 'tp': tp, 'fp': fp, 'fn': fn}
def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)
