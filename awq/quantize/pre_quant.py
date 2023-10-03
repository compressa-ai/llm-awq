import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip

__all__ = ["run_awq"]

_FREEZE_FRACTION = 1.0


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers
    
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))

@torch.no_grad()
def run_awq(
    model, enc,
    w_bit, q_config,
    n_samples=512, seqlen=512,
    auto_scale=True, mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name


    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    
    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()
        
        if mse_range:
            clip_list = auto_clip_block(layer,
                            w_bit=w_bit, q_config=q_config,
                            input_feat=input_feat,)
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        
    return awq_results


def apply_awq(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])



from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from ..utils.module import get_op_by_name, get_op_name, set_op_by_name


@torch.no_grad()
def freeze_fc_fc(fc1, fc2, fc1_orig, fc2_orig, scales, freeze_frac):
    assert fc1.__class__.__name__.endswith('Linear')
    assert fc2.__class__.__name__.endswith('Linear')
    # assert isinstance(fc1, nn.Linear), type(fc1)
    # assert isinstance(fc2, nn.Linear), type(fc2)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    """
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))
    """
    p = freeze_frac

    inds = torch.argsort(scales)
    num_inds_to_restore = int(len(scales) * p)
    num_inds_to_skip = len(scales) - num_inds_to_restore
    inds_to_restore = inds[num_inds_to_skip:]
    inds_to_skip = inds[:num_inds_to_skip]

    assert len(inds_to_skip) + len(inds_to_restore) == len(scales)
    assert torch.argmax(scales) in inds_to_restore

    print(f'!!! Num inds: {len(inds)}. Num inds to restore: {num_inds_to_restore}.')
    # print(f'!!! {torch.sum(scales)} {scales[:20]}')
    scales[inds_to_restore] = 1.0
    # print(f'!!! {torch.sum(scales)} {scales[:20]}')
    scales[inds_to_skip] = 0.0
    # print(f'!!! {torch.sum(scales)} {scales[:20]}')
    # print()

    assert torch.sum(scales) == num_inds_to_restore, f'{sum(scales)} {num_inds_to_restore} {len(inds_to_restore)} {len(scales)}'


    fc1.weight[-scales.size(0):] = (
        fc1.weight[-scales.size(0):] * (1 - scales.view(-1, 1))
        + fc1_orig.weight[-scales.size(0):] * scales.view(-1, 1)
    )
    if fc1.bias is not None:
        fc1.bias = (
            fc1.bias * (1 - scales.view(-1))
            + fc1_orig.bias * scales.view(-1)
        )

    fc2.weight.data = (
        fc2.weight.data * (1 - scales.view(1, -1))
        + fc2_orig.weight.data * scales.view(1, -1)
    )


    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def freeze_ln_fcs(ln, fcs, ln_orig, fcs_orig, scales, freeze_frac):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # debugging start even scales = 1 does not work?
    """
    scales = scales * 0
    scales = scales + 1
    """
    # debugging end

    """
    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    """
    p = freeze_frac

    inds = torch.argsort(scales)
    num_inds_to_restore = int(len(scales) * p)
    num_inds_to_skip = len(scales) - num_inds_to_restore
    inds_to_restore = inds[num_inds_to_skip:]
    inds_to_skip = inds[:num_inds_to_skip]

    assert len(inds_to_skip) + len(inds_to_restore) == len(scales)
    assert torch.argmax(scales) in inds_to_restore

    # print(f'!!! {torch.sum(scales)} {scales[:20]}')
    scales[inds_to_restore] = 1.0
    # print(f'!!! {torch.sum(scales)} {scales[:20]}')
    scales[inds_to_skip] = 0.0
    # print(f'!!! {torch.sum(scales)} {scales[:20]}')
    # print()

    assert torch.sum(scales) == num_inds_to_restore, f'{sum(scales)} {num_inds_to_restore} {len(inds_to_restore)} {len(scales)}'


    ln.weight.data = (
        ln.weight.data * (1 - scales)
        + ln_orig.weight.data * scales
    )

    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias = (
            ln.bias * (1 - scales)
            + ln_orig.bias.data * scales
        )

    for fc, fc_orig in zip(fcs, fcs_orig):
        fc.weight.data = (
            fc.weight.data * (1 - scales)
            + fc_orig.weight.data * scales
        )


    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    # for n, p in ln.named_parameters():
    #     assert p.is_cuda, f'{n}, {p}, {p.device}'
    #
    # for fc in fcs:
    #     for n, p in fc.named_parameters():
    #         assert p.is_cuda, f'{n}, {p}, {p.device}'


def freeze_awq(module, orig_module, awq_results, freeze_frac):
    scales_list = awq_results['scale']

    i = 0

    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        prev_op_orig = get_op_by_name(orig_module, prev_op_name)

        layers = [get_op_by_name(module, name) for name in layer_names]
        layers_orig = [get_op_by_name(orig_module, name) for name in layer_names]

        # prev_op.cuda()
        # for layer in layers:
        #     layer.cuda()
        scales.cuda()

        if prev_op.__class__.__name__.endswith('Linear'):  #isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            freeze_fc_fc(prev_op, layers[0], prev_op_orig, layers_orig[0], scales, freeze_frac)
        elif prev_op.__class__.__name__.endswith('LayerNorm') or prev_op.__class__.__name__.endswith('LlamaRMSNorm'):  #isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)):
            freeze_ln_fcs(prev_op, layers, prev_op_orig, layers_orig, scales, freeze_frac)
        else:
            raise NotImplementedError(
                f"prev_op {type(prev_op)} not supported yet!")

        # prev_op.cpu()
        # for layer in layers:
        #     layer.cpu()
        scales.cpu()

        # for n, p in module.named_parameters():
        #     assert p.is_cuda, f'{n}, {p}, {p.device}, iter {i}'

        i += 1

