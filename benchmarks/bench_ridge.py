"""bench different Ridge regression implementations"""

import numpy as np
from datetime import datetime


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from sklearn.linear_model import ridge
    start = datetime.now()
    skl_clf = ridge.RidgeCV(alphas = 2 ** np.linspace(-10, 11, 5),
        fit_intercept = True)
    skl_clf.fit(X, y)
    pred = skl_clf.predict(T)
    delta = datetime.now() - start
    mse = np.linalg.norm(pred - valid, 2)**2
    return mse, delta

if __name__ == '__main__':
    import sys, misc

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'
    if not len(sys.argv) == 2:
        print misc.USAGE % __file__
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print 'Loading data ...'
    data = misc.load_data(dataset)

    print 'Done, %s samples with %s features loaded into ' \
      'memory' % data[0].shape

    score, res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %.2f, std %.2f' % (
        np.mean(res_skl), np.std(res_skl))
    print 'MSE: %s\n' % score
