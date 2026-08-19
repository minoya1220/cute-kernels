import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(q_ptr, k_ptr, v_ptr, 
               out_ptr, 
               N: int, d_model: int, d_h: int, 
               BLK_SIZE_N: tl.constexpr, BLK_SIZE_H: tl.constexpr):
    
    pid_qN = tl.program_id(0)
    pid_H = tl.program_id(1)
    
    offset_qN = tl.arange(0, BLK_SIZE_N) + pid_qN * BLK_SIZE_N
    offset_H = tl.arange(0, BLK_SIZE_H) + pid_H * d_h

    mask_qN = offset_qN < N
    mask_H = tl.arange(0, BLK_SIZE_H) < d_h 
    
    offset_q = offset_qN[:, None] * d_model + offset_H[None,:]
    mask_q = mask_qN[:, None] & mask_H[None, :]


    tile_q = tl.load(q_ptr + offset_q, mask=mask_q, other=0.0)
    accum = tl.zeros((BLK_SIZE_N, BLK_SIZE_H), tl.float32)
    running_max = tl.full((BLK_SIZE_N,),float('-inf'), tl.float32) 
    running_sum = tl.zeros((BLK_SIZE_N,), tl.float32)

    for kv_i in range(0, N, BLK_SIZE_N):
        offset_kvN = tl.arange(0, BLK_SIZE_N) + kv_i 
        mask_kvN = offset_kvN < N 

        offset_kv = offset_kvN[:,None] * d_model + offset_H[None,:]
        mask_kv = mask_kvN[:, None] & mask_H[None, :]

        tile_k = tl.load(k_ptr + offset_kv, mask=mask_kv, other=0.0)
        tile_v = tl.load(v_ptr + offset_kv, mask=mask_kv, other=0.0)
        
        
        tile_qkt = tl.dot(tile_q, tile_k.T) / tl.sqrt(tl.cast(d_h, tl.float32)) # Q @ K.T / √d_h
        mask_qkt = (tl.arange(0, BLK_SIZE_N)[:,None] < N) & (offset_kvN[None, :] < N) 
        tile_qkt = tl.where(mask_qkt, tile_qkt, float('-inf')) 

        current_max = tl.maximum(tl.max(tile_qkt, 1), running_max)
        tile_qkt_scores = tl.exp(tile_qkt - current_max[:, None])
        rescale = tl.exp(running_max - current_max)
        tile_qkv = tl.dot(tile_qkt_scores, tile_v) 
        
        running_max = current_max
        running_sum = tl.fma(running_sum, rescale, tl.sum(tile_qkt_scores, 1))
        accum = tl.fma(accum, rescale[:, None], tile_qkv)
        
    accum /= running_sum[:, None]

    tl.store(out_ptr + offset_q, accum, mask=mask_q) # output and Q have the same shape
    

# Q, K, V, output are tensors on the GPU
def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, h: int):
    d_model = Q.shape[-1]
    N = Q.shape[-2]

    BLK_SIZE_N = 64
    d_h = d_model // h
    BLK_SIZE_H = max(32, triton.next_power_of_2(d_h))
    
    output = torch.empty_like(Q)

    grid = (triton.cdiv(N,BLK_SIZE_N),h) 
    flash_attn_kernel[grid](Q, K, V, output, N, d_model, d_h, BLK_SIZE_N, BLK_SIZE_H)
    
    return output


if __name__ =="__main__":
    d_model = 512
    N = 2048
    h = 8

    Q = torch.randn(N, d_model, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    
    # verify correctness
    actual = flash_attn(Q, K, V, h=h)
    
    Q_multihead = Q.view(N, h, d_model // h).transpose(0,1)
    K_multihead = K.view(N, h, d_model // h).transpose(0,1)
    V_multihead = V.view(N, h, d_model // h).transpose(0,1)
    expected = F.scaled_dot_product_attention(Q_multihead, K_multihead, V_multihead)
    expected = expected.transpose(0,1).contiguous().view(N, d_model)
    
    torch.testing.assert_close(expected, actual)
    print("passed")

    
