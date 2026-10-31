import torch
import torch.nn as nn
import torch.nn.functional as F
class SinkhornLoss(nn.Module):
    """
    优化版的Sinkhorn损失函数
    支持GPU加速和对数域稳定计算
    """
    def __init__(self, eps=0.01, max_iter=50, reduction='mean', 
                 cost_fn='l2', stop_thresh=1e-3):
        super(SinkhornLoss, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.cost_fn = cost_fn
        self.stop_thresh = stop_thresh
        

    # def compute_cost_matrix(self, x, y):
    # # ... 原有计算得到 C ...
    # # 新增标准化
    # C_mean = C.mean(dim=(-2,-1), keepdim=True)
    # C_std = C.std(dim=(-2,-1), keepdim=True).clamp_min(1e-8)
    # C = (C - C_mean) / C_std   # 变成零均值、单位方差
    # # 可选：再加一个偏移保证非负
    # C = C - C.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    # return C
    def compute_cost_matrix(self, x, y):
        """计算成本矩阵"""
        B, N, D = x.shape
        _, M, _ = y.shape
        if self.cost_fn == 'l2':
            # 使用展开的L2距离计算，更高效
            x_norm = (x ** 2).sum(-1, keepdim=True)  # (B, N, 1)
            y_norm = (y ** 2).sum(-1, keepdim=True)  # (B, M, 1)
            
            x_norm_expanded = x_norm.expand(-1, -1, M)  # (B, N, M)
            y_norm_expanded = y_norm.permute(0, 2, 1).expand(-1, N, -1)  # (B, N, M)
            
            # C_ij = ||x_i||^2 + ||y_j||^2 - 2<x_i, y_j>
            xy = torch.bmm(x, y.permute(0, 2, 1))  # (B, N, M)
            C = x_norm_expanded + y_norm_expanded - 2 * xy
            
            # 确保非负
            C = torch.clamp(C, min=0.0)
            C  = C/C.mean()
            # print('ccccc',C.min(),C.max(),C.mean())
        elif self.cost_fn == 'cosine':
            # 余弦距离
            x_norm = F.normalize(x, p=2, dim=-1)
            y_norm = F.normalize(y, p=2, dim=-1)
            C = 1 - torch.bmm(x_norm, y_norm.permute(0, 2, 1))
            
        else:
            # 通用Lp距离
            x_expanded = x.unsqueeze(2)  # (B, N, 1, D)
            y_expanded = y.unsqueeze(1)  # (B, 1, M, D)
            C = torch.norm(x_expanded - y_expanded, p=2, dim=-1)
        
        return C
    
    def sinkhorn_iteration(self, C, a, b):
        """Sinkhorn迭代（对数域稳定版本）"""
        B, N, M = C.shape
        
        # 核矩阵的对数形式: K = exp(-C/eps)
        K = -C / self.eps
        
        # 初始化对数u, v
        u = torch.zeros(B, N, device=C.device)
        v = torch.zeros(B, M, device=C.device)
        
        # 对数权重
        log_a = torch.log(a)
        log_b = torch.log(b)
        
        prev_error = float('inf')
        iter = 0
        for i in range(self.max_iter):
            # 更新u (log-sum-exp稳定计算)
            v_expanded = v.unsqueeze(1)  # (B, 1, M)
            K_v = K + v_expanded
            u_new = log_a - torch.logsumexp(K_v, dim=2)
            
            # 更新v
            u_expanded = u_new.unsqueeze(2)  # (B, N, 1)
            K_u = K + u_expanded
            v_new = log_b - torch.logsumexp(K_u, dim=1)
            
            # 检查收敛
            u_diff = torch.norm(u_new - u, dim=1).mean()
            v_diff = torch.norm(v_new - v, dim=1).mean()
            iter+=1
            # print(iter,u_diff,v_diff)
            if u_diff < self.stop_thresh and v_diff < self.stop_thresh:
                break
                
            u, v = u_new, v_new
        
        # 计算传输矩阵T的对数
        # print(iter,u_diff,v_diff)
        u_expanded = u.unsqueeze(2)  # (B, N, 1)
        v_expanded = v.unsqueeze(1)  # (B, 1, M)
        log_T = u_expanded + v_expanded + K
        
        return log_T
    
    def forward(self, x, y, a=None, b=None, return_transport=False):
        """
        计算Sinkhorn损失
        
        Args:
            x: (B, N, D) 源分布样本
            y: (B, M, D) 目标分布样本
            a: (B, N) 源分布权重
            b: (B, M) 目标分布权重
            return_transport: 是否返回传输矩阵
            
        Returns:
            loss: Sinkhorn损失
            (可选) T: 传输矩阵
        """
        B, N, D = x.shape
        _, M, _ = y.shape
        
        # 默认均匀分布
        if a is None:
            a = torch.ones(B, N, device=x.device) / N
        if b is None:
            b = torch.ones(B, M, device=y.device) / M
        
        # 归一化权重
        a = a / a.sum(dim=1, keepdim=True)
        b = b / b.sum(dim=1, keepdim=True)
        
        # 计算成本矩阵
        C = self.compute_cost_matrix(x, y)
        
        # Sinkhorn迭代
        log_T = self.sinkhorn_iteration(C, a, b)
        T = torch.exp(log_T)
        
        # 计算Wasserstein距离
        loss = torch.sum(T * C, dim=(-2, -1))
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        if return_transport:
            return loss, T
        return loss