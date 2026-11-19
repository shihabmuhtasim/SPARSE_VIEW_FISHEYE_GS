"""Sequential, logout-safe twelve-run queue. No silent partial checkpoint resume.

Completed, checksum-verified runs are skipped on a deliberate relaunch. If a
training process fails, preserve its files and stop with an actionable failure.
If training succeeded but reporting failed, a relaunch retries reporting only.
"""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, re, shutil, socket, subprocess, sys, time, traceback

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
ap.add_argument('--shared',type=Path,required=True);a=ap.parse_args()
ROOT=a.root.resolve();SHARED=a.shared.resolve();CONTROL=ROOT/'control';SOURCE=ROOT/'source'
PY=sys.executable;GiB=1024**3;SOFT=45*GiB;WORK=SHARED.parents[1]
METHODS=['pca_angle','angular_fps','coverage'];COUNTS=[12,24,48,96]
ROOT.mkdir(parents=True,exist_ok=True);SHARED.mkdir(parents=True,exist_ok=True)
lock=(ROOT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.umask(0o077)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',
    PYTHONPATH=str(SOURCE),TMPDIR=str(ROOT/'cache/tmp'),TORCH_HOME=str(ROOT/'cache/torch'),
    XDG_CACHE_HOME=str(ROOT/'cache'),MPLCONFIGDIR=str(ROOT/'cache/mpl'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for key in list(env):
    if key.startswith('DFGS_'):env.pop(key)
for name in ['tmp','torch','mpl']:(ROOT/'cache'/name).mkdir(parents=True,exist_ok=True)
# Keep saved progress visible during preflight/archive checks after a restart.
# The supervisor and the per-run loop still verify archive checksums before reuse.
completed=[f'{m}_{k}' for m in METHODS for k in COUNTS
           if (SHARED/f'runs/{m}_{k}/done.json').exists()]
child=None

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def status(state,**kw):
    d=dict(state=state,host=socket.gethostname(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           completed_runs=completed,total_runs=12,**kw)
    for p in [ROOT/'status.json',SHARED/'status.json']:dump(p,d)
    print(json.dumps(d),flush=True)
def space(reserve=3*GiB):
    used=int(subprocess.check_output(['du','-s','-B1',str(WORK)],text=True).split()[0])
    assert used+reserve<SOFT,f'Work storage guard: {used/GiB:.2f} GiB used; reserve {reserve/GiB:.2f}; soft limit45'
    assert shutil.disk_usage(ROOT).free>20*GiB,'Local storage guard: require20 GiB free'
    return used
def gpu_wait():
    clear=0
    while clear<3:
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        clear=clear+1 if not jobs else 0
        status('waiting_for_gpu',other_compute_pids=jobs,consecutive_clear_checks=clear)
        if clear<3:time.sleep(5)
def run_checked(cmd,log):
    with log.open('w') as f:subprocess.run(cmd,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def archive_tree(src,dst,ignore=None):
    shutil.copytree(src,dst,dirs_exist_ok=True,ignore=ignore)
    inventory={str(p.relative_to(src)):sha(p) for p in src.rglob('*') if p.is_file()
               and not (ignore and any(part=='train' or part=='__pycache__' or part.startswith('chkpnt') for part in p.relative_to(src).parts))}
    for name,h in inventory.items():assert sha(dst/name)==h,f'Archive mismatch: {name}'
    return inventory
def postprocess(method=None,count=None,prepare=False):
    cmd=[PY,str(CONTROL/'postprocess.py'),'--root',str(ROOT)]
    if prepare:cmd+=['--prepare-plots']
    if method:cmd+=['--method',method,'--count',str(count)]
    log=ROOT/(f'postprocess_{method}_{count}.log' if method else 'prepare_plots.log')
    run_checked(cmd,log)
def archive_run(run):
    space()
    dest=SHARED/'runs'/run.name
    inventory=archive_tree(run,dest,shutil.ignore_patterns('chkpnt*.pth','train','__pycache__'))
    dump(dest/'archive_sha256.json',inventory)
    assert all((dest/'model/test/ours_30000/renders'/f'{i:05d}.png').is_file() for i in range(19))
    # Only remove temporary optimizer states and train PNGs after metrics + final model backup pass.
    deleted=[]
    for p in (run/'model').glob('chkpnt*.pth'):
        deleted.append({'file':str(p),'bytes':p.stat().st_size,'reason':'Final EMA model and all test results verified in persistent archive'});p.unlink()
    p=run/'model/train'
    if p.exists():shutil.rmtree(p)
    dump(run/'temporary_cleanup.json',deleted)
    shutil.copy2(run/'temporary_cleanup.json',dest/'temporary_cleanup.json')
    dump(run/'done.json',{'state':'complete','archive_sha256_count':len(inventory),'time':time.time()})
    shutil.copy2(run/'done.json',dest/'done.json')

try:
    status('preflight')
    assert socket.gethostname().split('.')[0]=='eomer','This queue was prepared for Eomer'
    linger=subprocess.check_output(['loginctl','show-user',os.environ['USER'],'-p','Linger'],text=True).strip()
    if linger!='Linger=yes':
        print('Background permission was reset by the host; running in the current user session. Persistent service will start on next login.',flush=True)
    assert json.loads((CONTROL/'manifests/validation.json').read_text())['state']=='passed'
    expected=json.loads((CONTROL/'image_sha256.json').read_text())
    while True:
        missing=[n for n in expected if not (ROOT/'scene_flowers/images'/n).is_file()]
        if not missing:break
        status('waiting_for_data_transfer',received_images=len(expected)-len(missing),
               total_images=len(expected),missing_files=len(missing))
        time.sleep(20)
    for n,h in expected.items():assert sha(ROOT/'scene_flowers/images'/n)==h,f'Image mismatch: {n}'
    src_hash=json.loads((CONTROL/'training_source_sha256.json').read_text())
    for n,h in src_hash.items():assert sha(SOURCE/n)==h,f'Training source mismatch: {n}'
    already_done=sum((SHARED/f'runs/{m}_{k}/done.json').exists() for m in METHODS for k in COUNTS)
    remaining=12-already_done
    used=space(max(3,min(12,remaining))*GiB)
    dump(ROOT/'storage_budget.json',{'work_used_bytes':used,'work_soft_limit_bytes':SOFT,
        'planned_additional_persistent_allowance_bytes':max(3,min(12,remaining))*GiB,'local_free_bytes':shutil.disk_usage(ROOT).free,
        'retained':'Original input once; final EMA models,19 test GT/renders each, metrics, logs, TensorBoard, graphs',
        'temporary':'Optimizer checkpoints and training-render PNGs deleted only after validated persistent archive'})
    gpu_wait()
    audit=r'''
import json,os,torch,numpy as np
from pathlib import Path
from scene.dataset_readers import readColmapSceneInfo
r=Path(os.environ['STUDY_ROOT']); rows=[]
from simple_knn._C import distCUDA2
assert torch.isfinite(distCUDA2(torch.randn(16,3,device='cuda'))).all()
from lpipsPyTorch.modules.lpips import LPIPS
net=LPIPS('vgg','0.1');x=torch.zeros(1,3,32,32);assert abs(net(x,x).item())<1e-8
for method in ['pca_angle','angular_fps','coverage']:
 for k in [12,24,48,96]:
  manifest=r/f'control/manifests/flowers_{method}_{k}_v1.json'
  m=json.loads(manifest.read_text());os.environ['DFGS_FIXED_SPLIT']=str(manifest)
  s=readColmapSceneInfo(str(r/'scene_flowers'),'images',True)
  assert [c.image_name for c in s.train_cameras]==m['train_images']
  assert sorted(c.image_name for c in s.test_cameras)==sorted(m['test_images'])
  assert len(s.train_cameras)==k and len(s.test_cameras)==19
  assert len(s.point_cloud.points)==54865
  assert abs(float(s.nerf_normalization['radius'])-7.153377532958984)<1e-5
  assert all(c.camera_model==1 and c.image is not None for c in s.train_cameras+s.test_cameras)
  rows.append({'method':method,'count':k,'tests':19,'fisheye':True,'points':54865,'radius':float(s.nerf_normalization['radius'])})
(r/'split_audits.json').write_text(json.dumps(rows,indent=2))
(r/'environment.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_kernel_test':True,'lpips_preloaded':True,'source':'Frozen prior DirectFisheye bundle with explicit seed/manifest compatibility and filename-sorted final rendering; see patch'},indent=2))
print('PASS: CUDA/simple-knn, offline LPIPS-VGG, all12 splits,19 frozen tests, native fisheye,54865 points, original130-camera normalization')
'''
    env['STUDY_ROOT']=str(ROOT)
    run_checked([PY,'-c',audit],ROOT/'preflight.log')
    run_checked([PY,'-m','pip','freeze'],ROOT/'environment_packages.txt')
    status('archiving_inputs')
    for name in ['storage_budget.json','split_audits.json','environment.json','environment_packages.txt','preflight.log']:
        shutil.copy2(ROOT/name,SHARED/name)
    archive_tree(CONTROL,SHARED/'control',shutil.ignore_patterns('__pycache__'))
    archive_tree(SOURCE,SHARED/'source',shutil.ignore_patterns('__pycache__'))
    archive_tree(ROOT/'scene_flowers',SHARED/'scene_flowers')
    postprocess(prepare=True)
    archive_tree(ROOT/'plots',SHARED/'plots')
    for method in METHODS:
      for count in COUNTS:
        name=f'{method}_{count}';run=ROOT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
        if (run/'done.json').exists():
            stored=json.loads((SHARED/'runs'/name/'archive_sha256.json').read_text())
            for n,h in stored.items():assert sha(SHARED/'runs'/name/n)==h
            if name not in completed:completed.append(name)
            continue
        model=run/'model';manifest=CONTROL/f'manifests/flowers_{method}_{count}_v1.json'
        if not (run/'training_succeeded.json').exists():
            assert not model.exists(),f'Partial previous output for{name}; no silent restart. Preserve and inspect it.'
            space();gpu_wait()
            cmd=[PY,'-u',str(SOURCE/'train_eval.py'),'-s',str(ROOT/'scene_flowers'),'-m',str(model),
                 '--eval','--split_manifest',str(manifest),'--bs','2','--cvo_preset','recip',
                 '--iterations','30000','--seed','0','--disable_viewer',
                 '--test_iterations','7000','30000','--save_iterations','30000',
                 '--checkpoint_iterations','10000','20000','25000']
            dump(run/'command.json',{'argv':cmd,'manifest_sha256':sha(manifest),
               'environment':{k:v for k,v in env.items() if k in ['CUDA_VISIBLE_DEVICES','TORCH_HOME','PYTHONPATH','OPENBLAS_NUM_THREADS']},
               'source_hashes':src_hash})
            started=time.time();last_progress=started;last_iteration=-1;last_backup=0
            with (run/'train_eval.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    with (run/'train_eval.log').open('rb') as f:
                        f.seek(max(0,f.seek(0,2)-16000));tail=f.read().decode(errors='replace')
                    its=re.findall(r'(\d+)/30000',tail);iteration=int(its[-1]) if its else max(last_iteration,0)
                    if iteration!=last_iteration:last_progress=time.time();last_iteration=iteration
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
                    status('rendering_and_evaluating' if iteration>=30000 else 'training',run=name,
                       method=method,training_views=count,pid=child.pid,iteration=iteration,
                       progress_percent=round(iteration/300,2),gpu_utilization_and_memory=gpu,
                       elapsed_seconds=round(time.time()-started,1))
                    if time.time()-last_backup>120:
                        dest=SHARED/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(run/'train_eval.log',dest/'train_eval.log');last_backup=time.time()
                    if time.time()-last_progress>1200:
                        child.terminate()
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        raise RuntimeError(f'{name}: no iteration progress for20 minutes. Preserved logs/checkpoints; no automatic inaccurate resume.')
                    time.sleep(20)
            if child.returncode!=0:raise RuntimeError(f'{name}: training exit{child.returncode}; inspect {run}/train_eval.log')
            assert (model/'point_cloud/iteration_30000/point_cloud.ply').exists()
            dump(run/'timing.json',{'train_render_evaluate_seconds':time.time()-started})
            dump(run/'training_succeeded.json',{'state':'complete','iteration':30000,'time':time.time()})
        status('postprocessing',run=name)
        postprocess(method,count)
        status('archiving_results',run=name)
        archive_run(run);completed.append(name)
        postprocess()
        archive_tree(ROOT/'summary',SHARED/'summary')
    assert len(completed)==12
    postprocess();archive_tree(ROOT/'summary',SHARED/'summary')
    status('complete',message='ALL 12 EXPERIMENTS HAVE BEEN COMPLETED',results=str(SHARED/'summary/results.csv'))
    print('\nALL 12 EXPERIMENTS HAVE BEEN COMPLETED\n',flush=True)
except Exception:
    error=traceback.format_exc()
    # An archive/status/network error must not leave an untracked CUDA child alive.
    if child is not None and child.poll() is None:
        child.terminate()
        try:child.wait(timeout=45)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    try:status('failed',error=error)
    except OSError:print(error,flush=True)
    raise
