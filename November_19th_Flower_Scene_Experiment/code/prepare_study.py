"""Portable deterministic PCA / angular FPS / directional coverage manifests.

Usage: python prepare_study.py --scene-dir DATA --baseline-manifest BASE --output OUT
PCA invokes Sofia's unchanged, commit-pinned script. No test RGB or test tracks
enter the selectors. The shared full-scene COLMAP geometry remains test-assisted.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, struct, subprocess, sys
import numpy as np

HERE=Path(__file__).resolve().parent
BUDGETS=[12,24,48,96]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def poses(path):
    result={}
    with open(path,'rb') as f:
        for _ in range(struct.unpack('<Q',f.read(8))[0]):
            row=struct.unpack('<idddddddi',f.read(64)); name=bytearray()
            while (c:=f.read(1))!=b'\0':
                if not c: raise ValueError('Truncated COLMAP filename')
                name.extend(c)
            f.seek(24*struct.unpack('<Q',f.read(8))[0],1)
            q0,q1,q2,q3=row[1:5]
            R=np.array([[1-2*(q2*q2+q3*q3),2*(q1*q2-q0*q3),2*(q1*q3+q0*q2)],
                        [2*(q1*q2+q0*q3),1-2*(q1*q1+q3*q3),2*(q2*q3-q0*q1)],
                        [2*(q1*q3-q0*q2),2*(q2*q3+q0*q1),1-2*(q1*q1+q2*q2)]])
            result[name.decode()]={'id':row[0],'C':-R.T@np.array(row[5:8]),'forward':R.T[:,2]}
    return result

def points(path,ids):
    xyz=[]; tracks=[]; all_xyz=[]; point_ids=[]
    with open(path,'rb') as f:
        total=struct.unpack('<Q',f.read(8))[0]
        for _ in range(total):
            row=struct.unpack('<QdddBBBd',f.read(43)); n=struct.unpack('<Q',f.read(8))[0]
            tr=np.frombuffer(f.read(8*n),dtype='<i4').reshape(-1,2)
            eligible=sorted({ids[int(i)] for i in tr[:,0] if int(i) in ids})
            all_xyz.append(row[1:4])
            if len(eligible)>=2: xyz.append(row[1:4]); tracks.append(eligible); point_ids.append(row[0])
    return np.array(all_xyz),np.array(xyz),tracks,np.array(point_ids)

def fps(C,R,count):
    U=C-R; length=np.linalg.norm(U,axis=1)
    assert np.all(length>1e-12),'Camera at reference: undefined direction'
    U/=length[:,None]
    D=np.degrees(np.arccos(np.clip(U@U.T,-1,1))); np.fill_diagonal(D,0)
    pair=np.unravel_index(np.argmax(np.where(np.triu(np.ones_like(D,dtype=bool),1),D,-np.inf)),D.shape)
    selected=[]; history=[]
    for step in range(count):
        score=D[:,selected].min(1) if selected else None
        if step<2: i=int(pair[step])
        else:
            score[selected]=-np.inf; i=int(np.argmax(score))
        gap=None if not selected else float(D[i,selected].min())
        selected.append(i)
        history.append({'step':step+1,'index':i,'nearest_selected_angle_at_addition':gap,
                        'max_remaining_nearest_angle':float(D[:,selected].min(1).max())})
    return selected,history,U,D

def coverage(C,P,tracks,count):
    lengths=np.array([len(t) for t in tracks]); offsets=np.r_[0,np.cumsum(lengths)]
    weights=np.repeat(1/lengths,lengths).astype(np.float32)
    Q=np.zeros((len(C),int(offsets[-1])),np.float32)
    visible=np.zeros((len(C),len(P)),bool)
    for j,tr in enumerate(tracks):
        ix=np.array(tr);v=C[ix]-P[j];v/=np.linalg.norm(v,axis=1)[:,None]
        angle=np.degrees(np.arccos(np.clip(v@v.T,-1,1)))
        Q[np.ix_(ix,np.arange(offsets[j],offsets[j+1]))]=np.exp(-.5*(angle/30.)**2)
        visible[ix,j]=True
    selected=[];best=np.zeros(Q.shape[1],np.float32);seen=np.zeros(len(P),np.int32);hist=[]
    for step in range(count):
        score=np.maximum(Q-best,0)@weights;score[selected]=-np.inf;i=int(np.argmax(score))
        gain=float(score[i]);new=int(np.count_nonzero(visible[i]&(seen==0)))
        selected.append(i);best=np.maximum(best,Q[i]);seen+=visible[i]
        hist.append({'step':step+1,'index':i,'new_points':new,'covered_points':int(np.count_nonzero(seen)),
          'point_coverage_percent':float(100*np.count_nonzero(seen)/len(P)),
          'directional_score_percent':float(100*(best@weights)/len(P)),
          'directional_gain':gain,'two_view_supported_points':int(np.count_nonzero(seen>=2))})
    return selected,hist,visible,int(offsets[-1])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene-dir',type=Path,required=True)
    ap.add_argument('--baseline-manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    base=json.loads(a.baseline_manifest.read_text());names=list(base['train_images']);tests=base['test_images']
    assert names==sorted(names),'Use the same lexicographic baseline ordering as Sofia'
    assert len(names)>=96 and not set(names)&set(tests)
    geom=a.scene_dir/'sparse/0'; cams=poses(geom/'images.bin')
    assert set(cams)==set(names+tests)
    hashes={n:sha(geom/n) for n in ['cameras.bin','images.bin','points3D.bin']}
    assert hashes==base['reconstruction']['model_files_sha256']
    C=np.array([cams[n]['C'] for n in names]); F=np.array([cams[n]['forward'] for n in names])
    ids={cams[n]['id']:i for i,n in enumerate(names)}
    allP,P,tracks,point_ids=points(geom/'points3D.bin',ids);R=np.median(allP,axis=0)
    selected_fps,history_fps,U,D=fps(C,R,96)
    selected_cov,history_cov,visible,obs=coverage(C,P,tracks,96)
    if base['scene']=='flowers':
        assert len(allP)==54865 and len(P)==51695 and len(names)==130 and len(tests)==19
        assert [names[i] for i in selected_fps[:10]]==['110.jpg','91.jpg','102.jpg','55.jpg','123.jpg','98.jpg','60.jpg','117.jpg','107.jpg','126.jpg']
        assert [names[i] for i in selected_cov[:10]]==['89.jpg','41.jpg','125.jpg','87.jpg','114.jpg','31.jpg','83.jpg','11.jpg','38.jpg','6.jpg']
        assert history_cov[9]['covered_points']==22788
    for h in history_cov+history_fps:h['image']=names[h['index']]
    pca_script=HERE/'partner_reference/scripts/create_uniform_pose_manifest.py'
    provenance=json.loads((HERE/'partner_provenance.json').read_text())
    for method in ['pca_angle','angular_fps','coverage']:
        for count in BUDGETS:
            out=a.output/f'{base["scene"]}_{method}_{count}_v1.json'
            if method=='pca_angle':
                subprocess.run([sys.executable,str(pca_script),'--baseline-manifest',str(a.baseline_manifest),
                    '--images-bin',str(geom/'images.bin'),'--output',str(out),'--count',str(count)],check=True,stdout=subprocess.DEVNULL)
                m=json.loads(out.read_text()); selected=[names.index(n) for n in m['train_images']]
                params=m['selection']; params['source_repository']=provenance
                params['pca_script_sha256']=sha(pca_script)
                # Repeat the exact partner script to verify deterministic filenames, signs and ties.
                again=out.with_suffix('.repeat.json')
                subprocess.run([sys.executable,str(pca_script),'--baseline-manifest',str(a.baseline_manifest),
                    '--images-bin',str(geom/'images.bin'),'--output',str(again),'--count',str(count)],check=True,stdout=subprocess.DEVNULL)
                assert json.loads(again.read_text())['train_images']==m['train_images'];again.unlink()
            elif method=='angular_fps':
                selected=selected_fps[:count]
                params={'reference_rule':'coordinate-wise median of ALL frozen COLMAP scene points','reference_xyz':R.tolist(),
                    'direction':'normalize(camera_center-reference), using XYZ',
                    'distance':'degrees(arccos(clip(dot(unit_direction_i,unit_direction_j),-1,1)))',
                    'initialization':'pair with largest angle; first lexicographic pair for exact numerical ties',
                    'iteration':'unused camera with greatest minimum angular distance to selected cameras',
                    'tie_breaking':'first maximum in float64, lexicographic original filenames',
                    'nested':True,'selection_order':[names[i] for i in selected],
                    'reference':'PointNet++ FPS inspiration; normalized camera directions, not its neural network'}
            else:
                selected=selected_cov[:count]
                params={'point_filter':'at least2 distinct ORIGINAL candidate-training observations; exclude all test observation tracks',
                    'eligible_point_count':len(P),'eligible_observation_count':obs,'sigma_degrees':30.,
                    'similarity':'exp(-0.5*(point-to-camera 3D angle/30 degrees)^2); zero for unobserved point',
                    'score':'sum over points of mean over that points candidate observation directions of max selected similarity',
                    'weights':'equal total weight per point; each observation weight=1/track_length',
                    'initialization':'empty set; choose maximum total score',
                    'iteration':'choose largest marginal score improvement',
                    'tie_breaking':'first float32 maximum in lexicographic filename order',
                    'reference_used':False,'nested':True,'selection_order':[names[i] for i in selected],
                    'reference':'custom static COLMAP adaptation inspired by COVER, not published COVER'}
            counts=visible[selected].sum(0)
            m={'manifest_version':2,'scene':base['scene'],'selector':method,'selected_view_count':count,
                'eligible_train_images':names,'train_images':[names[i] for i in selected],'test_images':tests,
                'train_names':[names[i] for i in selected],'test_names':tests,
                'base_manifest_sha256':sha(a.baseline_manifest),'full_colmap_sha256':hashes,
                'selector_script_sha256':sha(__file__),'selection':params,
                'summary':{'eligible_points':len(P),'covered_points':int(np.count_nonzero(counts)),
                   'point_coverage_percent':float(100*np.count_nonzero(counts)/len(P)),
                   'two_view_supported_points':int(np.count_nonzero(counts>=2))},
                'geometry_limitation':base.get('geometry_limitation','full-scene COLMAP geometry')}
            assert len(set(m['train_images']))==count and not set(m['train_images'])&set(tests)
            out.write_text(json.dumps(m,indent=2)+'\n')
            print(method,count,m['summary'],flush=True)
    for method,hist in [('coverage',history_cov),('angular_fps',history_fps)]:
        (a.output/f'{method}_selection_history.json').write_text(json.dumps(hist,indent=2))
        with (a.output/f'{method}_selection_history.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    np.savez(a.output/'geometry.npz',C=C,F=F,R=R,U=U,names=np.array(names),
             selected_fps=np.array(selected_fps),selected_cov=np.array(selected_cov))
    (a.output/'validation.json').write_text(json.dumps({'state':'passed','manifest_count':12,
        'tests':len(tests),'candidates':len(names),'points':len(allP),'eligible_points':len(P),
        'historical_fps10_and_coverage10_sequences_reproduced':base['scene']=='flowers',
        'pca_exact_partner_script_used_and_repeated':True},indent=2))

if __name__=='__main__':main()
