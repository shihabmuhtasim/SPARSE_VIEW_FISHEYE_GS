"""Focused recovery tests. No training, CUDA work, or real results are modified."""
from pathlib import Path
import tempfile,json,fcntl
from recover_queue import prepare_recovery,sha,other_worker

def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))

with tempfile.TemporaryDirectory(prefix='flowers-recovery-test-') as directory:
    base=Path(directory);root=base/'local';shared=base/'persistent'
    root.mkdir();shared.mkdir()
    finished=shared/'runs/pca_angle_12';finished.mkdir(parents=True)
    artifact=finished/'proof.txt';artifact.write_text('Finished result must stay unchanged')
    write(finished/'done.json',{'state':'complete'})
    write(finished/'archive_sha256.json',{'proof.txt':sha(artifact)})
    partial=root/'runs/pca_angle_24';(partial/'model').mkdir(parents=True)
    (partial/'model/partial.txt').write_text('Preserve interrupted evidence')
    (partial/'train_eval.log').write_text('Training progress: 32% 9650/30000')
    write(root/'control/manifests/flowers_pca_angle_24_v1.json',{'test_images':['fixed-test.jpg']})
    done,restarted,evaluation=prepare_recovery(root,shared)
    assert done==['pca_angle_12'] and len(restarted)==1 and not evaluation
    assert restarted[0]['last_observed_iteration']==9650
    assert (root/'runs/pca_angle_12/done.json').exists()
    assert not (partial/'model').exists()
    history=Path(restarted[0]['preserved_attempt'])
    assert (history/'model/partial.txt').read_text()=='Preserve interrupted evidence'
    assert (shared/'interrupted_attempts/pca_angle_24'/history.name/'train_eval.log').exists()
    again=prepare_recovery(root,shared)
    assert again[0]==done and not again[1] and not again[2]
    assert not other_worker(root)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert other_worker(root)
    artifact.write_text('corrupted')
    try:prepare_recovery(root,shared)
    except AssertionError:pass
    else:raise AssertionError('Corrupted completed archive was incorrectly accepted')
print('PASS: completed-result preservation; interrupted-attempt isolation; idempotent recovery; duplicate-worker lock; archive corruption detection')
