import torch
import torch.nn as nn
# from knn_cuda import KNN as KNNCUDA


def _resolve_query_chunk_size(query, support, chunk_size=None, max_cdist_bytes=None):
    if chunk_size is not None:
        return max(1, min(query.shape[1], int(chunk_size)))
    if max_cdist_bytes is None:
        return query.shape[1]
    element_size = max(query.element_size(), support.element_size())
    bytes_per_query = max(1, query.shape[0] * support.shape[1] * element_size)
    return max(1, min(query.shape[1], max_cdist_bytes // bytes_per_query))


@torch.no_grad()
def _cdist_topk(query, support, k, largest=False, sorted=True, chunk_size=None, max_cdist_bytes=None):
    # Chunk large cdist calls to avoid allocating a full B x M x N distance matrix.
    query_chunk_size = _resolve_query_chunk_size(query, support, chunk_size, max_cdist_bytes)
    if query_chunk_size >= query.shape[1]:
        dist = torch.cdist(query, support)
        k_dist = dist.topk(k=k, dim=-1, largest=largest, sorted=sorted)
        return k_dist.values, k_dist.indices

    values = []
    indices = []
    for start in range(0, query.shape[1], query_chunk_size):
        end = min(start + query_chunk_size, query.shape[1])
        dist = torch.cdist(query[:, start:end], support)
        k_dist = dist.topk(k=k, dim=-1, largest=largest, sorted=sorted)
        values.append(k_dist.values)
        indices.append(k_dist.indices)
    return torch.cat(values, dim=1), torch.cat(indices, dim=1)


@torch.no_grad()
def knn_point(k, query, support=None, chunk_size=None, max_cdist_mb=128):
    """Get the distances and indices to a fixed number of neighbors
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
    """
    if support is None:
        support = query
    max_cdist_bytes = None if max_cdist_mb is None else int(max_cdist_mb * 1024 * 1024)
    return _cdist_topk(
        query,
        support,
        k,
        largest=False,
        sorted=True,
        chunk_size=chunk_size,
        max_cdist_bytes=max_cdist_bytes,
    )
    

class KNN(nn.Module):
    """Get the distances and indices to a fixed number of neighbors

    Reference: https://gist.github.com/ModarTensai/60fe0d0e3536adc28778448419908f47

    Args:
        neighbors: number of neighbors to consider
        p_norm: distances are computed based on L_p norm
        farthest: whether to get the farthest or the nearest neighbors
        ordered: distance sorted (descending if `farthest`)

    Returns:
        (distances, indices) both of shape [B, N, `num_neighbors`]
    """
    
    def __init__(self, neighbors, 
                 farthest=False, 
                 sorted=True, 
                 chunk_size=None,
                 max_cdist_mb=128,
                 **kwargs):
        super(KNN, self).__init__()
        self.neighbors = neighbors
        self.farthest = farthest
        self.sorted = sorted
        self.chunk_size = chunk_size
        self.max_cdist_bytes = None if max_cdist_mb is None else int(max_cdist_mb * 1024 * 1024)

    @torch.no_grad()
    def forward(self, query, support=None):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]

        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        if support is None:
            support = query
        k_dist_values, k_dist_indices = _cdist_topk(
            query,
            support,
            self.neighbors,
            largest=self.farthest,
            sorted=self.sorted,
            chunk_size=self.chunk_size,
            max_cdist_bytes=self.max_cdist_bytes,
        )
        return k_dist_values, k_dist_indices.int()


# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0,
                 knn_query_chunk_size=None, knn_max_cdist_mb=128):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        # self.knn = KNNCUDA(k * self.dilation, transpose_mode=True)
        self.knn = KNN(
            k * self.dilation,
            transpose_mode=True,
            chunk_size=knn_query_chunk_size,
            max_cdist_mb=knn_max_cdist_mb,
        )

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)
