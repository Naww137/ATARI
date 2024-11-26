import numpy as np

def MP_est(params, LTPs):
    X = params # (N,n)
    N = X.shape[0]

    # Relative Likelihoods:
    MLTP = np.max(LTPs)
    L = 10**(LTPs - MLTP).reshape(-1,1)

    # Finding Mean:
    print(X.T.shape)
    print(L.shape)
    M = (X.T @ L) / np.sum(L)

    # Finding Covariances:
    B = (X.T-np.tile(M,(1,N))) @ np.sqrt(L)
    Cov = (B @ B.T) / np.sum(L)
    return M.reshape(-1,), Cov

def MP_print(M, Cov, Txt):
    print('Results:')
    for txt, m, u in zip(Txt, M, np.sqrt(np.diag(Cov))):
        print(f'{txt} = {m:.5f} \u00B1 {u:.5f}')

if __name__ == '__main__':
    with open('Python_ENCORE/LTP_values_FM.csv','r') as file:
        # contents = file.read().split('\n')[273:396]
        contents = file.read().split('\n')[398:569]
        data = np.array(np.float_([content.split(',') for content in contents]))
    
    Txt = ['A freq', 'B freq', 'A Gnm', 'B Gnm', 'FreqF']
    params = data[:,:-1]
    LTPs   = data[:,-1]
    M, Cov = MP_est(params, LTPs)
    MP_print(M, Cov, Txt)

    # Goals = [0.11127431,0.12038765,44.11355000,33.38697000,0.00234002]
    Goals = [0.11127431,0.12038765,44.11355000,33.38697000,0.0234002]
    for j in range(5):
        print(f'{Txt[j]} = {Goals[j]:.5f}')


