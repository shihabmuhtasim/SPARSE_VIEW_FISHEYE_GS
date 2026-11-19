"""Persistent supervisor for the approved Flowers queue; training code unchanged.

Finished runs are verified and preserved. A killed partial training attempt is
recorded and restarted at seed0, because legacy checkpoints omit RNG/CVO/EMA
state. A fully saved final EMA model can continue rendering without retraining.
systemd restarts this supervisor after errors; its persistent user unit starts
on user-manager startup after login/boot. Host power/admin policy still applies.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,shutil,socket,subprocess,sys,time,traceback

METHODS=['pca_angle','angular_fps','coverage'];COUNTS=[12,24,48,96]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.recovery.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)

def completed_archive(shared,name):
    dest=shared/'runs'/name
    if not (dest/'done.json').exists():return False
    inventory=json.loads((dest/'archive_sha256.json').read_text())
    for n,h in inventory.items():
        assert sha(dest/n)==h,f'Completed archive mismatch for{name}: {n}'
    return True

def active_training(root):
    """Match argument tokens, not shell text or stale PIDs after reboot."""
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if str(root/'source/train_eval.py') in args:found.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found

def other_worker(root):
    p=(root/'runner.lock').open('a')
    try:
        fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(p,fcntl.LOCK_UN);return False
    except BlockingIOError:return True
    finally:p.close()

def metrics_complete(model,tests):
    try:
        folder=model/'test/ours_30000'
        order=json.loads((folder/'view_order.json').read_text())
        assert order==sorted(tests)
        for split in ['train','test']:
            values=json.loads((model/f'results_{split}.json').read_text())['ours_30000']
            assert set(values)=={'PSNR','SSIM','LPIPS'}
            assert all(__import__('math').isfinite(v) for v in values.values())
        per=json.loads((model/'per_view_test.json').read_text())['ours_30000']
        for metric in ['PSNR','SSIM','LPIPS']:
            assert len(per[metric])==len(tests)
            assert all(__import__('math').isfinite(v) for v in per[metric].values())
        assert all((folder/s/f'{i:05d}.png').exists() for i in range(len(tests)) for s in ['gt','renders'])
        return True
    except (OSError,KeyError,ValueError,AssertionError):return False

def prepare_recovery(root,shared):
    """Called only with no worker/renderer for this queue still active."""
    done=[];restarted=[];evaluation=[]
    for method in METHODS:
      for count in COUNTS:
        name=f'{method}_{count}';run=root/'runs'/name
        if completed_archive(shared,name):
            run.mkdir(parents=True,exist_ok=True)
            shutil.copy2(shared/'runs'/name/'done.json',run/'done.json')
            done.append(name);continue
        if (run/'training_succeeded.json').exists():continue
        model=run/'model'
        if not model.exists():continue
        log=run/'train_eval.log';text=log.read_text(errors='replace') if log.exists() else ''
        final=model/'point_cloud/iteration_30000/point_cloud.ply'
        manifest=root/f'control/manifests/flowers_{method}_{count}_v1.json'
        tests=json.loads(manifest.read_text())['test_images']
        if 'Training complete.' in text and final.is_file():
            # Training complete is printed only after the final EMA PLY save returns.
            from plyfile import PlyData
            assert len(PlyData.read(final,mmap=True)['vertex'])>0
            if metrics_complete(model,tests):
                dump(run/'training_succeeded.json',{'state':'complete','iteration':30000,
                     'recovery':'Already completed training and evaluation verified; postprocessing only','time':time.time()})
                if not (run/'timing.json').exists():
                    dump(run/'timing.json',{'train_render_evaluate_seconds':None,'timing_note':'Interrupted before timing marker; not measured'})
            else:evaluation.append((method,count))
            continue
        # Preserve all old evidence. Do not mix partial optimizer/EMA state into a new run.
        iteration=max([int(x) for x in re.findall(r'(\d+)/30000',text)] or [0])
        stamp=time.strftime('%Y%m%dT%H%M%S')+f'_{time.time_ns()%1000000:06d}'
        history=root/'interrupted_attempts'/name/stamp;history.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(run),history)
        run.mkdir(parents=True)
        record={'run':name,'last_observed_iteration':iteration,'action':'restart unfinished run from seed0',
                'reason':'Legacy optimizer checkpoints do not preserve full RNG,CVO and EMA state',
                'preserved_attempt':str(history),'time':time.time()}
        dump(history/'recovery.json',record)
        dest=shared/'interrupted_attempts'/name/stamp;dest.mkdir(parents=True,exist_ok=True)
        for f in ['train_eval.log','command.json','timing.json','recovery.json']:
            if (history/f).is_file():shutil.copy2(history/f,dest/f)
        restarted.append(record)
    return done,restarted,evaluation

def bootstrap(root,shared):
    root.mkdir(parents=True,exist_ok=True)
    # /local may be cleaned by the university. Restore validated persistent inputs.
    for sub in ['control','source','scene_flowers']:
        if not (root/sub).exists():shutil.copytree(shared/sub,root/sub)
    base=root/'control/manifests/flowers_fullview_split_v1.json'
    if not base.exists():shutil.copytree(shared/'control',root/'control',dirs_exist_ok=True)
    if not (root/'source/train_eval.py').exists():shutil.copytree(shared/'source',root/'source',dirs_exist_ok=True)
    hashes=json.loads((root/'control/image_sha256.json').read_text())
    for name,h in hashes.items():
        p=root/'scene_flowers/images'/name
        if not p.exists() or sha(p)!=h:
            src=shared/'scene_flowers/images'/name
            assert sha(src)==h,f'Persistent input mismatch: {name}'
            p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
    for name,h in json.loads(base.read_text())['reconstruction']['model_files_sha256'].items():
        p=root/'scene_flowers/sparse/0'/name
        if not p.exists() or sha(p)!=h:
            src=shared/'scene_flowers/sparse/0'/name
            assert sha(src)==h;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
    if (shared/'recovery_torch_cache').exists():
        for src in (shared/'recovery_torch_cache').rglob('*'):
            if not src.is_file():continue
            dst=root/'cache/torch'/src.relative_to(shared/'recovery_torch_cache')
            if not dst.exists() or dst.stat().st_size!=src.stat().st_size:
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def resume_evaluation(root,method,count):
    """Load the already saved final EMA model; never perform more optimization."""
    env=dict(os.environ,PYTHONPATH=str(root/'source'),CUDA_VISIBLE_DEVICES='0',
        TORCH_HOME=str(root/'cache/torch'),XDG_CACHE_HOME=str(root/'cache'),
        TMPDIR=str(root/'cache/tmp'),MPLCONFIGDIR=str(root/'cache/mpl'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        DFGS_FIXED_SPLIT=str(root/f'control/manifests/flowers_{method}_{count}_v1.json'))
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        time.sleep(20)
    run=root/f'runs/{method}_{count}';code=r'''
from argparse import ArgumentParser
from arguments import ModelParams,PipelineParams
from render import render_sets
from utils.general_utils import safe_state
p=ArgumentParser();m=ModelParams(p);q=PipelineParams(p)
args=p.parse_args();safe_state(False)
render_sets(m.extract(args),30000,q.extract(args),False,False,False,compute_metrics=True)
'''
    with (run/'evaluation_recovery.log').open('a') as log:
        subprocess.run([sys.executable,'-c',code,'-s',str(root/'scene_flowers'),'-m',str(run/'model'),'--eval'],
            cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    m=json.loads(Path(env['DFGS_FIXED_SPLIT']).read_text())
    assert metrics_complete(run/'model',m['test_images'])
    dump(run/'training_succeeded.json',{'state':'complete','iteration':30000,'recovery':'Final EMA model reevaluated; no further optimization','time':time.time()})
    if not (run/'timing.json').exists():dump(run/'timing.json',{'train_render_evaluate_seconds':None,'timing_note':'Original training was interrupted after final save'})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--shared',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();shared=args.shared.resolve()
    assert socket.gethostname().split('.')[0]=='eomer'
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Observe an existing worker instead of killing it or launching a duplicate.
    while other_worker(root) or active_training(root):
        dump(root/'supervisor_status.json',{'state':'existing_queue_still_running','time':time.time()})
        time.sleep(20)
    bootstrap(root,shared)
    done,restarted,evaluation=prepare_recovery(root,shared)
    record={'state':'recovering','completed_preserved':done,'restarted_attempts':restarted,
            'evaluation_only':[f'{m}_{k}' for m,k in evaluation],'time':time.time()}
    dump(root/'supervisor_status.json',record);dump(shared/'supervisor_status.json',record)
    print(json.dumps(record),flush=True)
    if len(done)==12:
        dump(root/'status.json',{'state':'complete','completed_runs':done,'total_runs':12,
             'message':'ALL 12 EXPERIMENTS HAVE BEEN COMPLETED'})
        print('ALL 12 EXPERIMENTS HAVE BEEN COMPLETED',flush=True);return
    for method,count in evaluation:resume_evaluation(root,method,count)
    worker=root/'control/run_study.py'
    with (root/'runner.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(worker),'--root',str(root),'--shared',str(shared)],
             cwd=root,stdout=log,stderr=subprocess.STDOUT)
        result=child.wait()
    if result!=0:raise RuntimeError(f'Queue worker exited{result}; systemd will retry after60 seconds')
    assert json.loads((root/'status.json').read_text())['state']=='complete'

if __name__=='__main__':
    try:main()
    except Exception:
        print(traceback.format_exc(),flush=True);raise
