import torch
from torch.autograd import Function
from torch import nn
import awq_inference_engine

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class WQLinearFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, x, qweight, qzeros, scales, bias=None, out_features=0):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, qweight, qzeros, scales, bias)
        ctx.out_features = out_features

        out_shape = x.shape[:-1] + (out_features, )
        x = x.to(torch.float16)  # TODO: qlora wants float16, not float32, how to fix?
        # out = awq_inference_engine.dequantize_weights_cuda(x.reshape(-1, x.shape[-1]), qw, scales[:2,:qout_c*8], qzeros[:qout_c,:2], 1, 0, 0, False)
        
        out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, 8)
        out = out + bias if bias is not None else out

        return out.reshape(out_shape)

    @staticmethod
    def backward(ctx, grad_output):
        print(ctx.needs_input_grad)
        input, qweight, qzeros, scales, bias = ctx.saved_tensors
        out_features = ctx.out_features
        grad_input = grad_weight = grad_zeros = grad_scales = grad_bias = grad_out_features = None
        weight = awq_inference_engine.dequantize_weights_cuda(qweight, scales, qzeros, 1, 0, 0, False)

        if ctx.needs_input_grad[0]:
            print("here!", grad_output[0].shape, weight.t().shape)
            pass
            grad_input = grad_output[0].mm(weight.t()).unsqueeze(0)
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_zeros, grad_scales, grad_bias, grad_out_features


class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer('qweight', torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
        
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=zeros.device)
        
        for col in range(zeros.shape[1] // pack_num):     
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros

        in_c = 3
        qout_c = 3

        # print(qweight[:in_c, :qout_c])
        # print(qzeros[:qout_c, :1])
        # print(scales[:qout_c*8, :1].t())
        
        return awq_linear

    @classmethod
    def from_qweight(cls, qweight, scales, qzeros, w_bit, group_size):
        if w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")

        pack_num = 32 // w_bit
        zeros = torch.zeros((qzeros.shape[0], qzeros.shape[1] * 32 // w_bit), dtype=torch.float16, device=qzeros.device)

        for col in range(zeros.shape[1] // pack_num):
            for i in range(pack_num):
                qzero_col = qzeros[:, col] << ((pack_num-1-i) * w_bit) >> ((pack_num-1) * w_bit)
                zeros[:, col * pack_num + order_map[i]] = qzero_col

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight


    def forward(self, x):
        return WQLinearFunction.apply(x, self.qweight, self.qzeros, self.scales, self.bias, self.out_features)

    # # @torch.no_grad()
    # def forward(self, x):
    #     out_shape = x.shape[:-1] + (self.out_features, )
    #     x = x.to(torch.float16)  # TODO: qlora wants float16, not float32, how to fix?
    #     out = awq_inference_engine.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
    #     out = out + self.bias if self.bias is not None else out
    #     return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )

if __name__ == "__main__":

    in_c = 3
    qout_c = 3

    l = nn.Linear(4096, 512, bias=None)
    #l.weight.data = torch.Tensor(list(range(4096*512))).reshape((512, 4096)) % 128
    l.weight.data = torch.zeros((512, 4096))
    l.weight.data[:, :] = 9
    l.weight.data[0, 0] = 1
    # l.weight.data[0, 1] = 2
    # l.weight.data[0, 2] = 3
    # l.weight.data[0, 3] = 4
    # l.weight.data[0, 4] = 5
    # l.weight.data[0, 5] = 6
    # l.weight.data[0, 6] = 7
    # l.weight.data[0, 7] = 8
    l.weight.data[1, 0] = 2
    l.weight.data[2, 0] = 3
    l.weight.data[3, 0] = 4
    l.weight.data[4, 0] = 5
    l.weight.data[5, 0] = 6
    l.weight.data[6, 0] = 7
    l.weight.data[7, 0] = 8
    l.weight.data = torch.randint_like(l.weight.data, 15)
    # l.weight.data[0, 2] = 3
    # l.weight.data[0, 3] = 3
    # l.weight.data[1, 0] = 3
    # l.weight.data[2, 0] = 3
    # l.weight.data[3, 0] = 3
    # l.weight.data[:,:] = 3
    # l.weight.data[:, 0] = 6
    # l.weight.data[0, 0] = 6
    # l.weight.data[0, 1] = 6
    # l.weight.data[0, 2] = 6
    # l.weight.data[1, 0] = 2
    # l.weight.data[0, 1] = 1
    # l.weight.data[:8, 0] = 3
    # l.weight.data[:8, 1] = 3
    # l.weight.data[0, :] = 3
    # l.weight.data[1, :] = 3
    # l.weight.data[1, 0] = 2
    # l.weight.data[1, 1] = 2
    # print(l.weight.data[:qout_c*8,:in_c].squeeze(0).t())
    # print(l.weight.data[:in_c,:qout_c*8].squeeze(0))
    # l.weight.data[-1, 0] = 2
    # l.weight.data[0, -1] = 2
    # l.weight.data[-1, -1] = 2
    wql = WQLinear.from_linear(l, 4, 128, scales=torch.ones((32, 512))*1, zeros=torch.zeros((32, 512))).cuda(0)
    # w = WQLinear.from_qweight(wql.qweight, wql.scales, wql.qzeros, 4, 128)
    inp = torch.randn((1,257,4096)).cuda(0) #(1,257,4096)
    inp.requires_grad = True
    o = wql(inp)
    loss = nn.functional.mse_loss(o, torch.zeros_like(o))
    loss.backward()
    # print(l.weight.data.t().shape)    
    # l.weight.data.t() - 
