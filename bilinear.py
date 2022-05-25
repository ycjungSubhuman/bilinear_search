import torch

def linear_interpolation(a, b, x):
    return torch.einsum('b,bc->bc', x, a) + torch.einsum('b,bc->bc', (1-x), b)

def find_linear_interpolation(a, b, v):
    """
    Find linear interpolation point that minimizes the error ||f - v||^2 where
    f = x*a + (1-x)*b

    a      (B, C)
    b      (B, C)
    v      (B, C)
    
    Returns)
    x               a floating point value for each batch (x \in [0, 1]), shape=(B,)
    
    
    argmin_x E=0.5||x*a + (1-x)*b - v||^2
    =        0.5||x(a-b)-(v-b)||^2
    =        0.5(x(a-b)-(v-b)).T@(x(a-b)-(v-b))
    =        0.5(x^2(a-b).T@(a-b) - 2x(a-b).T@(v-b) + (v-b).T@(v-b))
    === dE/dx = 0 ==>
    =>       x(a-b).T@(a-b) -(a-b).T@(v-b) = 0
    =>       x = (a-b).T@(v-b) / (a-b).T@(a-b) 
    """
    return torch.clip(((a-b)*(v-b)).sum(dim=1) / ((a-b)*(a-b)).sum(dim=1), 0, 1).nan_to_num(1.0)

def bilinear_interpolation(Q11, Q21, Q12, Q22, alpha, beta):
    Q_l = linear_interpolation(Q11, Q12, beta)
    Q_r = linear_interpolation(Q21, Q22, beta)
    Q = linear_interpolation(Q_l, Q_r, alpha)
    
    return Q


def find_bilinear_interpolation(Q11, Q21, Q12, Q22, v):
    """
    Find bilinear interpolation point that minimizes the error ||f - v||^2 where

    discrete samples at each corner
    Q11      (B, C)
    Q21      (B, C)
    Q12      (B, C)
    Q22      (B, C)

    Returns)
    alpha (B,) in [0, 1]
    beta (B,) in [0, 1]
    """
    # First solve beta
    beta = find_linear_interpolation(0.5*(Q11+Q21), 0.5*(Q12+Q22), v)
    # Then solve alpha
    Q_l = linear_interpolation(Q11, Q12, beta)
    Q_r = linear_interpolation(Q21, Q22, beta)
    alpha = find_linear_interpolation(Q_l, Q_r, v)
    
    return alpha, beta

def find_bilinear_window(image, v):
    """
    Find the minimum position in the image that minimizes distance to v
    image:     (B, C, H, W)
    v:         (B, C)
    
    Returns)
    positions   (B, 2) each position in ([0,W], [0,H])
    """
    B = image.shape[0]
    C = image.shape[1]
    H = image.shape[2]
    W = image.shape[3]
    
    dx = torch.tensor([1, 0], device=image.device, dtype=torch.long)
    dy = torch.tensor([0, 1], device=image.device, dtype=torch.long)

    xinds = torch.arange(W, device=image.device, dtype=torch.long)
    # Avoid finding borders
    xinds[0] = 1
    xinds[-1] = W-2
    yinds = torch.arange(H, device=image.device, dtype=torch.long)
    yinds[0] = 1
    yinds[-1] = H-2
    grid_x, grid_y = torch.meshgrid(xinds, yinds)
    grid = torch.cat([grid_x.T[...,None], grid_y.T[...,None]], dim=2).reshape(-1, 2)

    posinds = ((image - v.reshape(B, C, 1, 1).repeat(1, 1, H, W))**2).sum(dim=1).view(B, -1).argmin(dim=1)
    pos = grid[posinds]
    binds = torch.arange(B, device=image.device)
    
    quads = [
        [pos, pos+dx, pos+dy, pos+dx+dy],
        [pos, pos-dx, pos+dy, pos-dx+dy],
        [pos, pos+dx, pos-dy, pos+dx-dy],
        [pos, pos-dx, pos-dy, pos-dx-dy],
    ]
    
    errors = []
    solutions = []
    for iQ11, iQ21, iQ12, iQ22 in quads:
        Q11 = image[binds, :, iQ11[:,1], iQ11[:,0]]
        Q21 = image[binds, :, iQ21[:,1], iQ21[:,0]]
        Q12 = image[binds, :, iQ12[:,1], iQ12[:,0]]
        Q22 = image[binds, :, iQ22[:,1], iQ22[:,0]]
        alpha, beta = find_bilinear_interpolation(Q11, Q21, Q12, Q22, v)
        Q = bilinear_interpolation(Q11, Q21, Q12, Q22, alpha, beta)
        iQ = bilinear_interpolation(iQ11, iQ21, iQ12, iQ22, alpha, beta)
        
        errors.append(((Q - v)**2).sum(dim=1))
        solutions.append(iQ)
    ind_best = torch.stack(errors).argmin(dim=0)
    print(ind_best)
    solutions = torch.stack(solutions)
    solution_final = solutions[ind_best, binds] 
    
    return solution_final

if __name__ == '__main__':
    import time
    grid_x, grid_y = torch.meshgrid(
    torch.arange(10),
    torch.arange(10))
    grid = torch.cat([grid_x.T[...,None], grid_y.T[...,None]], dim=2) * 1.0
    image1 = grid.permute(2, 0, 1)
    image2 = grid.permute(2, 1, 0)

    im_batch = torch.stack([image1, image2]).cuda().repeat(1000000, 1, 1, 1)

    v1 = torch.tensor([5.3, 7.0])
    v2 = torch.tensor([3.7, 3.9])
    v = torch.stack([v1, v2]).cuda().repeat(1000000, 1)

    torch.cuda.synchronize()
    start = time.time()
    solution = find_bilinear_window(im_batch, v)
    print('GT: ')
    print([
        [5.3, 7.0],
        [3.9, 3.7],
    ])
    print('Solution:')
    print(solution)
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)

    print('If image is constant: ')
    im_batch = 0.0*im_batch
    v1 = torch.tensor([5.7, 5.9])
    v2 = torch.tensor([3.7, 3.9])
    v = torch.stack([v1, v2]).cuda().repeat(1000000, 1)

    torch.cuda.synchronize()
    start = time.time()
    solution = find_bilinear_window(im_batch, v)
    print('GT: all not NaN')
    print([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    print('Solution:')
    print(solution)
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)