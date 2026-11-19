"""Validate official metrics; create curves and honest, matched visual examples."""
from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['pca_angle','angular_fps','coverage']; COUNTS=[12,24,48,96]
SCENE='flowers'
LABELS={'pca_angle':'PCA angle','angular_fps':'3D angular FPS','coverage':'Coverage adaptation'}
METRICS=['PSNR','SSIM','LPIPS']

def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(path,rows):
    if not rows:return
    with path.open('w') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def compare_panel(path,title,columns):
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,17) if Path(font_path).exists() else ImageFont.load_default()
    big=ImageFont.truetype(font_path,22) if Path(font_path).exists() else font
    w=650; images=[]
    for image,label in columns:
        im=Image.open(image).convert('RGB');im.thumbnail((w,420));images.append((im,label))
    h=max(im.height for im,_ in images)
    canvas=Image.new('RGB',(w*len(images),h+125),'white');d=ImageDraw.Draw(canvas)
    d.text((12,8),title,fill='#173346',font=big)
    for i,(im,label) in enumerate(images):
        d.multiline_text((w*i+10,45),label,fill='black',font=font,spacing=4)
        canvas.paste(im,(w*i+(w-im.width)//2,120))
    canvas.save(path)

def prepare_plots(root):
    manifests=root/'control/manifests'; out=root/'plots';out.mkdir(exist_ok=True)
    geo=np.load(manifests/'geometry.npz'); C=geo['C'];names=geo['names'].tolist()
    test=json.loads((manifests/f'{SCENE}_fullview_split_v1.json').read_text())['test_images']
    for method in METHODS:
        for count in COUNTS:
            m=json.loads((manifests/f'{SCENE}_{method}_{count}_v1.json').read_text())
            ix=[names.index(n) for n in m['train_images']]
            fig=plt.figure(figsize=(10,7));ax=fig.add_subplot(111,projection='3d')
            ax.scatter(*C.T,color='#b0b0b0',s=15,alpha=.4,label=f'{len(names)} available training cameras')
            ax.scatter(*C[ix].T,c=np.arange(count),cmap='viridis',s=45,label=f'{count} selected cameras')
            ax.quiver(*C[ix].T,*geo['F'][ix].T,length=.5,color='#326e82',alpha=.6)
            R=np.array(m['selection'].get('camera_center_mean',geo['R']))
            if method!='coverage':ax.scatter(*R,marker='*',color='black',s=130,label='Selection reference')
            ax.set(xlabel='Reconstruction X',ylabel='Reconstruction Y',zlabel='Reconstruction Z')
            ax.set_box_aspect(np.ptp(C,axis=0));ax.view_init(23,-65);ax.legend(loc='upper left',fontsize=8)
            ax.set_title(f'{LABELS[method]}: {count} training photos')
            fig.text(.05,.02,'Positions and lens arrows; coordinates are relative reconstruction units, not measured ground heights.',fontsize=9)
            fig.savefig(out/f'{method}_{count}_cameras.png',dpi=140,bbox_inches='tight');plt.close(fig)
    h=json.loads((manifests/'coverage_selection_history.json').read_text())
    for count in COUNTS:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5));steps=np.arange(1,count+1)
        axes[0].bar(steps,[v['new_points'] for v in h[:count]],color='#188a8a')
        axes[0].set(xlabel='Photo selection order',ylabel='New distinct COLMAP points',title='New points added by each photo')
        axes[1].plot(steps,[v['point_coverage_percent'] for v in h[:count]],label='Distinct point coverage')
        axes[1].plot(steps,[v['directional_score_percent'] for v in h[:count]],label='Directional representation score')
        axes[1].set(xlabel='Selected photos',ylabel='Percent of eligible point total',ylim=(0,100));axes[1].legend(fontsize=8)
        fig.suptitle(f'Coverage adaptation: first {count} selections')
        fig.text(.05,.01,'New-point bars do not include repeated points seen from useful new directions. These also affect selection.',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/f'coverage_{count}_progress.png',dpi=140);plt.close(fig)

def process_run(root,method,count):
    run=root/'runs'/f'{method}_{count}';model=run/'model';out=run/'report';out.mkdir(exist_ok=True)
    manifest=json.loads((root/f'control/manifests/{SCENE}_{method}_{count}_v1.json').read_text())
    result=json.loads((model/'results_test.json').read_text())['ours_30000']
    per=json.loads((model/'per_view_test.json').read_text())['ours_30000']
    folder=model/'test/ours_30000';order=json.loads((folder/'view_order.json').read_text())
    assert order==sorted(manifest['test_images']) and len(order)==len(manifest['test_images'])
    rows=[]
    for i,name in enumerate(order):
        key=f'{i:05d}.png';v={metric:float(per[metric][key]) for metric in METRICS}
        assert all(math.isfinite(x) for x in v.values())
        assert (folder/'gt'/key).exists() and (folder/'renders'/key).exists()
        rows.append({'test_image':name,'render_file':key,**v})
    for metric in METRICS:assert abs(np.mean([x[metric] for x in rows])-result[metric])<1e-6
    # Frozen GT is also compared across every method/budget. Never assume PNG numbers identify images.
    cache=root/'ground_truth';cache.mkdir(exist_ok=True)
    for row in rows:
        src=folder/'gt'/row['render_file'];dst=cache/(row['test_image']+'.png')
        if dst.exists():assert sha(src)==sha(dst),f'Ground truth mismatch: {row["test_image"]}'
        else:dst.write_bytes(src.read_bytes())
    csvout(out/'per_test_image.csv',rows)
    ranked=sorted(rows,key=lambda x:(x['PSNR'],x['test_image']))
    picks={label:ranked[i] for label,i in [('worst',0),('median',len(rows)//2),('best',len(rows)-1)]}
    dump(out/'representatives.json',{'ranking':f'ascending final test PSNR; middle rank {len(rows)//2+1} of {len(rows)}; filename tie break',
        'used_for_training_or_model_selection':False,'examples':picks})
    for label,row in picks.items():
        subtitle=f'PSNR {row["PSNR"]:.2f} dB | SSIM {row["SSIM"]:.3f} | LPIPS {row["LPIPS"]:.3f}'
        compare_panel(out/f'{label}.png',f'{LABELS[method]} / {count} photos / {label}: {row["test_image"]}',[
            (folder/'gt'/row['render_file'],'Ground truth'),
            (folder/'renders'/row['render_file'],f'{LABELS[method]}\n{subtitle}')])
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,METRICS):
        ax.plot(np.arange(1,len(rows)+1),[r[metric] for r in rows],'o-');ax.set(xlabel='Held-out photo (filename order)',ylabel=metric)
        ax.grid(alpha=.2)
    fig.suptitle(f'{LABELS[method]}, {count} views: all {len(rows)} test images');fig.tight_layout()
    fig.savefig(out/'test_metrics.png',dpi=150);plt.close(fig)
    # Export every recorded scalar. Curves retain the official checkpoint evaluation frequency.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea=EventAccumulator(str(model),size_guidance={'scalars':0});ea.Reload();scalars=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4)); plotted=[False]*3
    for tag in ea.Tags()['scalars']:
        events=ea.Scalars(tag)
        scalars.extend({'tag':tag,'iteration':e.step,'wall_time':e.wall_time,'value':e.value} for e in events)
        if 'total_loss' in tag:
            values=np.array([e.value for e in events]);steps=np.array([e.step for e in events])
            width=min(200,len(values));smooth=np.convolve(values,np.ones(width)/width,mode='valid')
            axes[0].plot(steps[width-1:],smooth,label='Training loss (200-step mean)');plotted[0]=True
        if 'psnr' in tag:
            axes[1].plot([e.step for e in events],[e.value for e in events],'o-',label=tag);plotted[1]=True
        if tag=='total_points':axes[2].plot([e.step for e in events],[e.value for e in events]);plotted[2]=True
    for i,ax in enumerate(axes):
        ax.set_xlabel('Iteration');ax.grid(alpha=.2)
        if plotted[i] and i!=2:ax.legend(fontsize=7)
    axes[0].set_ylabel('Loss');axes[1].set_ylabel('Checkpoint PSNR (train uses sampled views)');axes[2].set_ylabel('Gaussians')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
    csvout(out/'training_scalars.csv',scalars)
    dump(out/'validation.json',{'state':'passed','test_count':len(rows),'mean_metrics_verified':True,'all_test_gt_matches_other_runs':True})
    return result

def summarize(root):
    out=root/'summary';out.mkdir(exist_ok=True);rows=[]
    for method in METHODS:
        for count in COUNTS:
            run=root/f'runs/{method}_{count}';p=run/'report/validation.json'
            if not p.exists():continue
            r=json.loads((run/'model/results_test.json').read_text())['ours_30000']
            t=json.loads((run/'timing.json').read_text())
            rows.append({'method':method,'training_views':count,**r,**t})
    csvout(out/'results.csv',rows);dump(out/'results.json',rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for method in METHODS:
        vals=[r for r in rows if r['method']==method]
        for ax,metric in zip(axes,METRICS):
            if vals:ax.plot([r['training_views'] for r in vals],[r[metric] for r in vals],'o-',label=LABELS[method])
            ax.set(xlabel='Training photos',ylabel=metric,xticks=COUNTS);ax.grid(alpha=.2)
    if rows:axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'quality_vs_view_count.png',dpi=160);plt.close(fig)
    for count in COUNTS:
        paths=[root/f'runs/{m}_{count}' for m in METHODS]
        if not all((p/'report/validation.json').exists() for p in paths):continue
        for anchor,anchor_path in zip(METHODS,paths):
            picks=json.loads((anchor_path/'report/representatives.json').read_text())['examples']
            for label,ref in picks.items():
                cols=[(root/'ground_truth'/(ref['test_image']+'.png'),'Ground truth')]
                for method,p in zip(METHODS,paths):
                    with (p/'report/per_test_image.csv').open() as f: candidates=list(csv.DictReader(f))
                    row=next(x for x in candidates if x['test_image']==ref['test_image'])
                    cols.append((p/'model/test/ours_30000/renders'/row['render_file'],
                      f'{LABELS[method]}\nPSNR {float(row["PSNR"]):.2f} | SSIM {float(row["SSIM"]):.3f} | LPIPS {float(row["LPIPS"]):.3f}'))
                compare_panel(out/f'{count}_{anchor}_{label}_matched.png',
                    f'{count} training photos / {LABELS[anchor]}-{label}: {ref["test_image"]}',cols)
            dump(out/f'{count}_{anchor}_matched_examples.json',
                {'selection_rule':f'{LABELS[anchor]} test PSNR worst/median/best; identical photo across all methods',
                 'examples':picks,'presentation_only':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prepare-plots',action='store_true');ap.add_argument('--method',choices=METHODS)
    ap.add_argument('--count',type=int);a=ap.parse_args()
    base_files=list((a.root/'control/manifests').glob('*_fullview_split_v1.json'))
    assert len(base_files)==1
    SCENE=json.loads(base_files[0].read_text())['scene']
    if a.prepare_plots:prepare_plots(a.root)
    if a.method:process_run(a.root,a.method,a.count)
    summarize(a.root)
