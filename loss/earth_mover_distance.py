import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def sinkhorn_transport(cost: torch.Tensor, eps: float = 0.01, max_iters: int = 20) -> torch.Tensor:
	"""Entropic-regularized optimal transport plan (Sinkhorn) on the
	Birkhoff polytope (row/col sums of 1, not 1/N), solved in the log domain
	for numerical stability. As eps -> 0 this plan approaches the optimal
	one-to-one assignment between the two point sets, so `sum(plan * cost)`
	is directly comparable to a bipartite-matching EMD (a sum over N matched
	pairs), not an average-cost-per-unit-probability-mass.

	cost: (B, N, M) pairwise cost matrix.
	"""
	B, N, M = cost.shape
	log_mu = cost.new_zeros((B, N))
	log_nu = cost.new_zeros((B, M))
	u = torch.zeros_like(log_mu)
	v = torch.zeros_like(log_nu)

	def log_kernel(u, v):
		return (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

	for _ in range(max_iters):
		u = eps * (log_mu - torch.logsumexp(log_kernel(u, v), dim=-1)) + u
		v = eps * (log_nu - torch.logsumexp(log_kernel(u, v).transpose(-2, -1), dim=-1)) + v

	return torch.exp(log_kernel(u, v))


def emd(template: torch.Tensor, source: torch.Tensor, eps: float = 0.01, max_iters: int = 20):
	"""Differentiable approximate Earth Mover's Distance between two equally
	sized point sets, computed purely in PyTorch (no compiled CUDA extension)
	via entropic optimal transport (Sinkhorn).
	"""
	assert template.size(1) == source.size(1), 'template and source must have the same number of points'
	cost = torch.cdist(template, source, p=2)
	# The transport plan is solved without tracking gradients through the
	# iterations: at Sinkhorn's fixed point, the plan is the argmin of the
	# entropic OT objective, so by the envelope theorem the gradient of the
	# cost w.r.t. the point coordinates only needs to flow through `cost`,
	# treating `plan` as constant. Backprop-ing through all iterations
	# instead is what makes naive unrolled Sinkhorn blow up in memory/time.
	with torch.no_grad():
		plan = sinkhorn_transport(cost, eps=eps, max_iters=max_iters)
	emd_cost = torch.sum(plan * cost, dim=(-2, -1))
	return emd_cost.mean() / template.size(1)


class EMDLosspy(nn.Module):
	def __init__(self, eps: float = 0.01, max_iters: int = 20):
		super(EMDLosspy, self).__init__()
		self.eps = eps
		self.max_iters = max_iters

	def forward(self, template, source):
		return emd(template, source, eps=self.eps, max_iters=self.max_iters)


if __name__ == '__main__':
    loss = EMDLosspy()
    a = torch.randn(4, 5, 3).cuda()
    b = copy.deepcopy(a)
    v = loss(a, b)
    print(v)
