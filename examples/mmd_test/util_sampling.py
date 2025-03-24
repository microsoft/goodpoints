import numpy as np
import time
import os
import pickle
from emnist import extract_training_samples

"""
%%%%%%%%%%% Sampling functions %%%%%%%%%%%
"""
    
def load_and_store_EMNIST():
    """
    Loads and stores the EMNIST dataset (byclass dataset)
    """
    X, y = extract_training_samples('byclass')
    
    X = np.array(X)
    y = np.array(y)
    X = X / 255
    characters = {}
    characters_list = []
    for i in range(62):
        characters[str(i)] = []
        characters_list.append(str(i))
    print(f'X type: {type(X)}. y type: {type(y)}.')
    print(f'X[0] type: {type(X[0])}.')
    print(f'Length of y: {len(y)}. Length of X: {len(X)}.')
    print(f'First values of y: {y[0]}, {y[1]}, {y[2]}, {y[4]}.')
    for k, v in characters.items():
        print(f'Key {k} of type {type(k)}.')
    for i in range(len(y)):
        if not (str(y[i]) in characters_list):
            print(f'{str(y[i])} of type {type(str(y[i]))} not in list. Index i={i}.')
        characters[str(y[i])].append(X[i])
    if not os.path.exists('data'):
        os.makedirs('data')
    data_file = os.path.join('data', 'emnist.data')
    f = open(data_file, 'wb')
    pickle.dump(characters, f)
    f.close()
    
def downsample_EMNIST():
    """
    Downsamples the EMNIST dataset from 28x28 to 7x7
    """
    data_file = os.path.join('data', 'emnist.data')
    with open(data_file, 'rb') as handle:
        X = pickle.load(handle)
    # X is a dictionary
    characters = {}
    for i in range(62):
        characters[str(i)] = []
    for i in range(62):
        current = np.array(X[str(i)])
        n = len(current)
        current = np.reshape(current, (n, 7, 4, 7, 4))
        current = current.mean(axis=(2, 4))
        characters[str(i)] = np.reshape(current, (n, 49))
    compressed_data_file = os.path.join('data', 'emnist_7x7.data')
    f = open(compressed_data_file, 'wb')
    pickle.dump(characters, f)
    f.close()

def generate_samples_EMNIST(args,rng):
    """
    Generates two sets P, Q of samples from downsampled (7x7) EMNIST
    
    Outputs:
      P: 2-D array of size (args.n,49), contains random samples of the 10
        digits with equal frequency
      Q: 2-D array of size (args.n,49), contains random samples of digits
        with different frequencies, several behaviors depending on args.name
    """
    start = time.time()
    
    data_file = os.path.join('data', 'emnist_7x7.data')
    if not os.path.exists(data_file):
        load_and_store_EMNIST()
        downsample_EMNIST()
    with open(data_file, 'rb') as handle:
        X = pickle.load(handle)
        
    even_characters = np.vstack(tuple([X[str(2*i)] for i in range(31)]))
    odd_characters = np.vstack(tuple([X[str(2*i+1)] for i in range(31)]))
    print(f'Length even characters: {len(even_characters)}')
    print(f'Length odd characters: {len(odd_characters)}')
    
    n_even_characters_1 = np.sum(rng.binomial(1, 0.5, size=args.n))
    print(f'args.n: {args.n}. n_even_characters_1: {n_even_characters_1}')
    idx_even_1 = rng.integers(len(even_characters), size=n_even_characters_1)
    idx_odd_1 = rng.integers(len(odd_characters), size=args.n-n_even_characters_1)
    
    n_even_characters_2 = np.sum(rng.binomial(1, args.p_even, size=args.n))
    idx_even_2 = rng.integers(len(even_characters), size=n_even_characters_2)
    idx_odd_2 = rng.integers(len(odd_characters), size=args.n-n_even_characters_2)
    
    print(f'even_characters[idx_even_1,:].shape: {even_characters[idx_even_1,:].shape}')
    print(f'odd_characters[idx_odd_1,:].shape: {odd_characters[idx_odd_1,:].shape}')
    
    X1 = np.concatenate((even_characters[idx_even_1,:],odd_characters[idx_odd_1,:]), axis=0)
    X2 = np.concatenate((even_characters[idx_even_2,:],odd_characters[idx_odd_2,:]), axis=0)
    
    rng.shuffle(X1)
    rng.shuffle(X2)
    
    print(f'X1.shape: {X1.shape}')
    print(f'X2.shape: {X2.shape}')
    
    print(f'avg norm: {np.mean(np.linalg.norm(X1, axis=1))}')

    end = time.time()
    print(f'Time elapsed: {end-start}.')
    
    return X1, X2

def generate_samples_Higgs(args,rng):
    """
    Generates two sets P, Q of samples from the Higgs dataset
    """
    start = time.time()
    
    # Load data
    if not args.mixing:
        if not args.null:
            data1 = pickle.load(open('./data/higgs_14_17_0.pckl', 'rb'))
            data2 = pickle.load(open('./data/higgs_14_17_1.pckl', 'rb'))
            #data1 = pickle.load(open('./data/higgs_14_21_0.pckl', 'rb'))
            #data2 = pickle.load(open('./data/higgs_14_21_1.pckl', 'rb'))
            #data1 = data[0]
            #data2 = data[1]
        else:
            #data1 = data[0]
            #data2 = data[0]
            data1 = pickle.load(open('./data/higgs_14_17_0.pckl', 'rb'))
            #data1 = pickle.load(open('./data/higgs_14_21_0.pckl', 'rb'))
            data2 = data1
            
        idx_1 = rng.integers(data1.shape[0], size=args.n)
        X1 = data1[idx_1,:]    

        idx_2 = rng.integers(data2.shape[0], size=args.n)
        X2 = data2[idx_2,:]
        
        return X1, X2

        #X1 = data1[idx_1]
        #X2 = data2[idx_2]

        #X1 = np.expand_dims(X1, axis=1)
        #X2 = np.expand_dims(X2, axis=1)
    else:
        data = pickle.load(open('./data/HIGGS_TST.pckl', 'rb'))
        data1 = data[0]
        data2 = data[1]    

        idx_1 = rng.integers(data1.shape[0], size=args.n)
        X1 = data1[idx_1,:args.d]

        n_poisoned = np.sum(rng.binomial(1, args.p_poisoning, size=args.n))
        print(f'args.n: {args.n}. n_poisoned: {n_poisoned}')
        idx_2_poisoned = rng.integers(data1.shape[0], size=n_poisoned)
        idx_2_true = rng.integers(data2.shape[0], size=args.n-n_poisoned)

        print(f'data1[idx_2_poisoned,:].shape: {data1[idx_2_poisoned,:args.d].shape}')
        print(f'data2[idx_2_true,:].shape: {data2[idx_2_true,:args.d].shape}')

        X2 = np.concatenate((data1[idx_2_poisoned,:args.d],data2[idx_2_true,:args.d]), axis=0)
        rng.shuffle(X2)

        print(f'X1.shape: {X1.shape}')
        print(f'X2.shape: {X2.shape}')

        end = time.time()
        print(f'Time elapsed: {end-start}.')

        return X1, X2

def generate_samples_blobs(args,cov,rng):
    """
    Generates one set of samples from the Blobs distribution, with separation 10 between centers
    
    args.grid_size is the number of centers in each side of the grid, e.g. if 
      args.grid_size = 3, then the grid is 3x3
    cov: covariance between the two components, must be in [-1,1]
    """
    centers = 10*(rng.integers(args.grid_size, size=(args.n,args.d)) - int(np.floor(args.grid_size/2))) 
    cov_matrix = np.array([[1,cov],[cov,1]])
    X = centers+rng.multivariate_normal(np.zeros(args.d), cov_matrix, args.n)
    return X

def generate_samples_gaussians(args,rng):
    """
    Generates two sets of multivariate Gaussians in dimension args.d, with identity covariances
      and means that are args.mean_diff apart
    """
    mean_1 = np.zeros(args.d)
    mean_2 = np.zeros(args.d)
    mean_1[args.d-1] = args.mean_diff/2
    mean_2[args.d-1] = -args.mean_diff/2
    mean_1 = tuple(mean_1)
    mean_2 = tuple(mean_2)
    cov_1 = np.eye(args.d)
    cov_2 = np.eye(args.d)
    
    X1 = rng.multivariate_normal(mean_1, cov_1, args.n) 
    X2 = rng.multivariate_normal(mean_2, cov_2, args.n)
    return X1, X2

def generate_samples_HDGM(args,rng,alternative=True):
    """
    Generates two sets of multivariate Gaussians in dimension args.d, with identity covariances
      and means that are args.mean_diff apart
    """
    mean_a = np.zeros(args.d)
    mean_b = 0.5*np.ones(args.d)
    mean_a = tuple(mean_1)
    mean_b = tuple(mean_2)
    cov_a = np.eye(args.d)
    cov_b = np.eye(args.d)
    cov_b[0,1] = 0.5
    cov_b[1,0] = 0.5
    cov_c = np.eye(args.d)
    cov_c[0,1] = -0.5
    cov_c[1,0] = -0.5
    
    n_points_a_P = np.sum(rng.binomial(1, 0.5, size=args.n))
    X1a = rng.multivariate_normal(mean_a, cov_a, n_points_a_P) 
    X1b = rng.multivariate_normal(mean_b, cov_a, args.n-n_points_a_P)
    X1 = np.concatenate((X1a,X1b), axis=0)
    rng.shuffle(X1)
    if alternative:
        n_points_a_Q = np.sum(rng.binomial(1, 0.5, size=args.n))
        X2a = rng.multivariate_normal(mean_a, cov_b, n_points_a_Q) 
        X2b = rng.multivariate_normal(mean_b, cov_c, args.n-n_points_a_Q)
        X2 = np.concatenate((X2a,X2b), axis=0)
        rng.shuffle(X2)
    else:
        n_points_a_Q = np.sum(rng.binomial(1, 0.5, size=args.n))
        X2a = rng.multivariate_normal(mean_a, cov_a, n_points_a_Q) 
        X2b = rng.multivariate_normal(mean_b, cov_b, args.n-n_points_a_Q)
        X2 = np.concatenate((X2a,X2b), axis=0)
        rng.shuffle(X2)
    return X1, X2

def generate_samples_sine(args,rng,alternative=True):
    """
    Generates two sets of multivariate Gaussians in dimension args.d, with identity covariances
      and means that are args.mean_diff apart
    """
    mean = np.zeros(args.d)
    mean_t = tuple(mean)
    cov = np.eye(args.d)
    e_1 = np.zeros(args.d)
    e_1[0] = 1
    
    X1 = rng.multivariate_normal(mean_t, cov, args.n)
 
    small_n = int(np.ceil(args.n/5))
    k = 0
    length = 0
    while length < args.n:
        X2_small = rng.multivariate_normal(mean_t, cov, small_n)
        sin_factor = np.sin(args.omega*np.matmul(X2_small,e_1))
        masks = rng.binomial(1,(1+sin_factor)/2)
        X2_subsample = X2_small[masks[:] == 1,:]
        if k == 0:
            X2 = X2_subsample
        else:
            samples_left = args.n-X2.shape[0]
            if X2_subsample.shape[0] < samples_left:
                X2 = np.concatenate((X2,X2_subsample), axis=0)
            else:
                X2 = np.concatenate((X2,X2_subsample[:samples_left,:]), axis=0)
        k += 1
        length = X2.shape[0]
    print(f'{k} batches used, shape of X2: {X2.shape}')
    
    return X1, X2

def generate_samples(args,rng,seed=None,f_theta_seed=None):
    """
    Depending on args.name, returns two sets of samples for the Gaussians, 
      Blobs or MNIST distributions
    """
    if args.name == 'gaussians':
        return generate_samples_gaussians(args,rng)
    elif args.name == 'blobs':
        X1 = generate_samples_blobs(args,0,rng)
        cov = (args.epsilon-1)/(args.epsilon+1)
        X2 = generate_samples_blobs(args,cov,rng)
        return X1, X2
    elif args.name == 'MNIST':
        return generate_samples_MNIST(args,rng)
    elif args.name == 'EMNIST':
        return generate_samples_EMNIST(args,rng)
    elif args.name == 'Higgs':
        return generate_samples_Higgs(args,rng)
    elif args.name == 'sine':
        return generate_samples_sine(args,rng)
