import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from util.configuration import Config
from util.exp_preprocess import standard_preprocessing
from util.pygidloader import PyGIDDataset
from util.postprocessing import onnx_to_xyxy, filter_boxes
from util.matchers import get_matcher

CKPT='/mnt/lustre/work/schreiber/szb389/train_output/ringseg_2class_20260603-142434/checkpoint.pth'
DSET='/mnt/lustre/work/schreiber/szb389/datasets/organic_labeled.h5'
SCORE_THRESH=0.3   # operating point for the FN/FP breakdown

args=get_args_parser().parse_args([])
for k,v in SLConfig.fromfile('config/DINO/DINO_4scale_swin.py')._cfg_dict.to_dict().items(): setattr(args,k,v)
args.device='cuda'; args.export=False
model,_,_=build_model_main(args); model=model.cuda().eval()
model.load_state_dict(torch.load(CKPT,map_location='cpu')['model'])

cfg=Config(); cfg.PREPROCESSING_POLAR_SHAPE=[512,1024]; cfg.POSTPROCESSING_SCORE=0.1
cfg.POSTPROCESSING_CLASSAWARE_NMS=True; cfg.INPUT_DATASET=DSET
ds=PyGIDDataset(cfg,path=DSET,preprocess_func=standard_preprocessing,buffer_size=5,load_labels=True)
matcher=get_matcher('q',min_iou=0.1)

# accumulators
gt_vis=[]; gt_isr=[]; gt_matched=[]; gt_q=[]          # per GT box
fp_score=[]; fp_q=[]; fp_label=[]                       # per false positive (score>thresh, unmatched)
n_img=0
with torch.no_grad():
    for ic in ds.iter_images():
        n_img+=1
        img=torch.tensor(ic.converted_polar_image[:,0,:,:]).unsqueeze(0).cuda().repeat(1,args.num_channels,1,1)
        out=model(img)
        raw=[out['pred_logits'].detach().cpu().numpy(), out['pred_boxes'].detach().cpu().numpy()]
        ic=filter_boxes(cfg, onnx_to_xyxy(cfg, ic, raw))
        pred=ic.boxes; sc=ic.scores; lab=getattr(ic,'pred_labels',None)
        keep=sc>SCORE_THRESH
        pred=pred[keep]; sc=sc[keep]; lab=(lab[keep] if lab is not None else None)
        L=ic.polar_labels
        gt=torch.tensor(np.array(L.boxes),dtype=torch.float32) if len(L.boxes) else torch.zeros((0,4))
        vis=np.array(L.visibility) if len(L.visibility) else np.zeros(len(gt))
        isr=np.array([bool(x) for x in L.is_ring]) if len(L.is_ring) else np.zeros(len(gt),bool)
        row=np.array([],int); col=np.array([],int)
        if len(gt) and len(pred):
            try: _,row,col=matcher(gt,pred)
            except IndexError: pass
        mset=set(row.tolist())
        for i in range(len(gt)):
            gt_vis.append(int(vis[i])); gt_isr.append(bool(isr[i])); gt_matched.append(i in mset)
            gt_q.append(float((gt[i,0]+gt[i,2])/2))
        cset=set(col.tolist())
        for j in range(len(pred)):
            if j not in cset:
                fp_score.append(float(sc[j])); fp_q.append(float((pred[j,0]+pred[j,2])/2))
                fp_label.append(int(lab[j]) if lab is not None else -1)
ds.close() if hasattr(ds,'close') else None

gt_vis=np.array(gt_vis); gt_isr=np.array(gt_isr); gt_matched=np.array(gt_matched); gt_q=np.array(gt_q)
fp_score=np.array(fp_score); fp_q=np.array(fp_q); fp_label=np.array(fp_label)
N=len(gt_vis); M=gt_matched.sum()
print(f"=== DIAGNOSTIC C (organic, {n_img} imgs, score>{SCORE_THRESH}) ===")
print(f"GT peaks={N}  matched(TP)={M}  missed(FN)={N-M}  recall={M/N:.3f}   FP={len(fp_score)}  precision={M/(M+len(fp_score)):.3f}")
print("\n-- recall by visibility (3=high/2=med/1=low) --")
for v in [3,2,1]:
    m=gt_vis==v; print(f"  vis={v}: n={m.sum():4d}  recall={gt_matched[m].mean() if m.sum() else float('nan'):.3f}")
print("\n-- recall by type --")
for name,m in [('ring',gt_isr),('segment',~gt_isr)]:
    print(f"  {name:8s}: n={m.sum():4d}  recall={gt_matched[m].mean() if m.sum() else float('nan'):.3f}")
print("\n-- recall by q-position (thirds of 1024) --")
for lo,hi in [(0,341),(341,682),(682,1024)]:
    m=(gt_q>=lo)&(gt_q<hi); print(f"  q[{lo:4d}-{hi:4d}): n={m.sum():4d}  recall={gt_matched[m].mean() if m.sum() else float('nan'):.3f}")
print("\n-- false positives --")
print(f"  total FP={len(fp_score)}  per image={len(fp_score)/n_img:.1f}")
if len(fp_score):
    print(f"  FP score: mean={fp_score.mean():.2f} p50={np.percentile(fp_score,50):.2f} p90={np.percentile(fp_score,90):.2f}  high-conf(>0.5)={np.mean(fp_score>0.5):.2f}")
    print(f"  FP predicted class: ring={np.mean(fp_label==1):.2f} segment={np.mean(fp_label==0):.2f}")

# figures
fig,ax=plt.subplots(1,3,figsize=(16,4.2))
ax[0].bar(['high(3)','med(2)','low(1)'],[gt_matched[gt_vis==v].mean() if (gt_vis==v).sum() else 0 for v in [3,2,1]],color=['tab:green','tab:orange','tab:red'])
ax[0].set_title('recall by visibility'); ax[0].set_ylim(0,1); ax[0].set_ylabel('recall')
ax[1].hist(gt_q[~gt_matched],bins=30,range=(0,1024),alpha=0.6,label='missed (FN)',color='tab:red')
ax[1].hist(gt_q[gt_matched],bins=30,range=(0,1024),alpha=0.6,label='detected (TP)',color='tab:green')
ax[1].set_title('GT peaks by q-position'); ax[1].set_xlabel('q (px)'); ax[1].legend()
if len(fp_score): ax[2].hist(fp_score,bins=30,range=(SCORE_THRESH,1),color='tab:purple'); ax[2].set_title('false-positive score distribution'); ax[2].set_xlabel('score')
fig.tight_layout(); out=__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'diagnose_C.png'); fig.savefig(out,dpi=100,bbox_inches='tight'); print('\nsaved',out)
