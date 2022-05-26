import numpy as np
import torch
import tqdm
import math

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    nonzero_idxs = X.norm(dim=1, p=0).nonzero().squeeze()
    num_samples = len(nonzero_idxs)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[nonzero_idxs[indices]]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu'),
        num_samples=50000,
        batch_size = 10000,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    full_X = X
    perm = torch.randperm(len(X))
    X = X[perm[:num_samples]]
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm.tqdm(desc='[running kmeans]')
    while True:
        dis = torch.empty((len(X), num_clusters))
        for i in range(math.ceil(len(X)/batch_size)):
            # print(dis[i*batch_size:i*batch_size+batch_size].shape, pairwise_distance_function(X[i*batch_size:i*batch_size+batch_size], initial_state).shape)
            dis[i*batch_size:i*batch_size+batch_size] = pairwise_distance_function(X[i*batch_size:i*batch_size+batch_size], initial_state)
        # dis = 

        choice_cluster = torch.argmin(dis, dim=1)
        # print(dis)
        # print(initial_state[choice_cluster[0]])
        # print(choice_cluster)
        initial_state_pre = initial_state.clone()
        null_count = 0
        for index in range(num_clusters):
            # print(torch.nonzero(choice_cluster == index))
            selected = torch.nonzero(choice_cluster == index).squeeze(1).to(device)
            # print(len(selected.shape))
            # print(len(selected))
            if (len(selected)>0):
                selected = torch.index_select(X, 0, selected)
                initial_state[index] = selected.mean(dim=0)
            else:
                null_count += 1
        print(null_count)
            
        # print(initial_state)
        # print(initial_state_pre)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if iteration%1==0:

            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )            
            # tqdm_meter.update() 
              
        if center_shift ** 2< tol:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )            
            tqdm_meter.close()
            break
        # print(center_shift)
        if torch.isnan(center_shift ** 2):
            tqdm_meter.close()
            return None, None
        tol *= 1.2
    tqdm_meter.close()
    X = full_X
    dis = torch.empty((batch_size, num_clusters), device=device)
    choice_cluster = torch.empty(len(X), device=device)
    for i in tqdm.trange(math.ceil(len(X)/batch_size), mininterval=5):
        dis[:min(batch_size, len(X)-batch_size*i)] = pairwise_distance_function(X[i*batch_size:i*batch_size+batch_size].to(device), initial_state, device)
        # print(dis.shape, choice_cluster[i*batch_size:i*batch_size+batch_size].shape)
        choice_cluster[i*batch_size:i*batch_size+batch_size] = torch.argmin(dis[:min(batch_size, len(X)-batch_size*i)], dim=1)
    for index in range(num_clusters):
        selected = torch.nonzero(choice_cluster == index).squeeze(1).to(device)
        # selected = torch.index_select(X, 0, selected)
        # initial_state[index] = selected.mean(dim=0)    
        if (len(selected)>0):
            selected = torch.index_select(X, 0, selected.to(X.device))
            initial_state[index] = selected.mean(dim=0).to(device)
           
    return choice_cluster.cpu(), initial_state.cpu()




def get_centers(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu'),
        num_samples=50000,
        batch_size = 10000,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    perm = torch.randperm(len(X))
    X = X[perm[:num_samples]]
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters)

    iteration = 0
    tqdm_meter = tqdm.tqdm(desc='[running kmeans]')
    while True:
        dis = torch.empty((len(X), num_clusters))
        for i in range(math.ceil(len(X)/batch_size)):
            # print(dis[i*batch_size:i*batch_size+batch_size].shape, pairwise_distance_function(X[i*batch_size:i*batch_size+batch_size], initial_state).shape)
            dis[i*batch_size:i*batch_size+batch_size] = pairwise_distance_function(X[i*batch_size:i*batch_size+batch_size], initial_state)
        # dis = 

        choice_cluster = torch.argmin(dis, dim=1)
        # print(dis)
        # print(initial_state[choice_cluster[0]])
        # print(choice_cluster)
        initial_state_pre = initial_state.clone()
        null_count = 0
        for index in range(num_clusters):
            # print(torch.nonzero(choice_cluster == index))
            selected = torch.nonzero(choice_cluster == index).squeeze(1).to(device)
            # print(len(selected.shape))
            # print(len(selected))
            if (len(selected)>0):
                selected = torch.index_select(X, 0, selected)
                initial_state[index] = selected.mean(dim=0)
            else:
                null_count += 1
        print(null_count)
            
        # print(initial_state)
        # print(initial_state_pre)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if iteration%1==0:

            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )            
            # tqdm_meter.update() 
              
        if center_shift ** 2< tol:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )            
            tqdm_meter.close()
            break
        # print(center_shift)
        if torch.isnan(center_shift ** 2):
            tqdm_meter.close()
            return None, None
        tol *= 1.2
    tqdm_meter.close()
           
    return initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu'),
        batch_size = 10000
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    # print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # dis = pairwise_distance_function(X, cluster_centers)
    # choice_cluster = torch.argmin(dis, dim=1)
    num_clusters = len(cluster_centers)
    # print(X.shape, cluster_centers.shape)

    dis = torch.empty((batch_size, num_clusters), device=device)
    choice_cluster = torch.empty(len(X), device=device)
    for i in range(math.ceil(len(X)/batch_size)):
        dis[:min(batch_size, len(X)-batch_size*i)] = pairwise_distance_function(X[i*batch_size:i*batch_size+batch_size].to(device), cluster_centers, device)

        choice_cluster[i*batch_size:i*batch_size+batch_size] = torch.argmin(dis[:min(batch_size, len(X)-batch_size*i)], dim=1)
    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    # data1, data2 = data1.to(torch.float16), data2.to(torch.float16)
    # BS*M, len*M
    data1, data2 = data1.to(device), data2.to(device)
    # print(data1.shape, data2.shape)
    # M*N*1
    # cosine = torch.ones((data1.shape[0], data2.shape[0]), dtype=torch.float32, device=device)
    data1 = data1.unsqueeze(dim=1)
    data2 = data2.unsqueeze(dim=0)
    cosine = 1-torch.nn.functional.cosine_similarity(data1, data2, dim=2)


    return cosine

