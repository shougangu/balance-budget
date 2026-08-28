# ABOUTME: Compares a full fine-tune checkpoint against the Llama-3.1-8B base, per module.
# ABOUTME: Reports the fraction of bit-identical elements and the relative weight delta.
from safetensors import safe_open
import glob,json,os,torch,collections,sys
base=[d for d in glob.glob('/home/shougan/.cache/huggingface/hub/models--unsloth--Meta-Llama-3.1-8B/snapshots/*/') if os.path.exists(d+'model.safetensors.index.json')][0]
ft=sys.argv[1]
bi=json.load(open(base+'model.safetensors.index.json'))['weight_map']; fi=json.load(open(ft+'model.safetensors.index.json'))['weight_map']
agg=collections.defaultdict(lambda:[0,0,0.0,0.0]); CH=4096
for k in bi:
    typ='embed' if 'embed' in k else 'lm_head' if 'lm_head' in k else 'norm' if 'norm' in k else k.split('.')[-2]
    a=agg[typ]
    with safe_open(base+bi[k],'pt') as fb, safe_open(ft+fi[k],'pt') as ff:
        sb,sf=fb.get_slice(k),ff.get_slice(k); n=sb.get_shape()[0]
        for i in range(0,n,CH):
            b=sb[i:i+CH]; x=sf[i:i+CH]
            a[0]+=b.numel(); a[1]+=(b==x).sum().item()
            a[2]+=(x.float()-b.float()).pow(2).sum().item(); a[3]+=b.float().pow(2).sum().item()
    print(k, flush=True, file=sys.stderr)
tot=[0,0,0.0,0.0]
print(f"{'module':12s} {'params':>14s} {'unchanged':>10s} {'rel-delta':>10s}")
for t,a in sorted(agg.items(),key=lambda x:-x[1][0]):
    for i in range(4): tot[i]+=a[i]
    print(f"{t:12s} {a[0]:14,d} {a[1]/a[0]:10.1%} {(a[2]/a[3])**0.5:10.4f}")
print(f"{'ALL':12s} {tot[0]:14,d} {tot[1]/tot[0]:10.1%} {(tot[2]/tot[3])**0.5:10.4f}")
