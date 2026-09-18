from pathlib import Path
import os, sys, json, random, pickle, hashlib, time
# Source files stay in PROJECT; all generated artifacts stay in ROOT (result).
PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent / 'result'
DATA = PROJECT.parent / 'Lightweight_FE_KFormer_CR_virtual_data' / 'Lightweight_FE_KFormer_CR_virtual_data' / '07_Features'
for directory in [ROOT, ROOT / 'tmp', ROOT / 'logs', ROOT / 'cache' / 'matplotlib']:
    directory.mkdir(parents=True, exist_ok=True)
# OpenFE writes relative temporary files, so its working directory must be result.
os.chdir(ROOT)
for key in ['TEMP','TMP','TMPDIR']:
    os.environ[key] = str(ROOT / 'tmp')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache' / 'matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / 'cache' / 'torch')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'cache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch
from light_fe_kformer_cr.openfe_stage import FEATURE_GROUPS
from light_fe_kformer_cr.data import compute_cr
from light_fe_kformer_cr.tokens import SemanticTokenPipeline
from light_fe_kformer_cr.model import LightFEKFormerCR
from light_fe_kformer_cr.losses import CoolingResilienceLoss
from light_fe_kformer_cr.train import make_loader, fit_model, evaluate

def main():
    use_openfe = '--openfe' in sys.argv
    if use_openfe:
        # Restricted Windows session cannot create multiprocessing named pipes.
        # Preserve worker isolation by copying callables and arguments at submit.
        import importlib, copy
        from concurrent.futures import ThreadPoolExecutor
        class IsolatedThreadExecutor(ThreadPoolExecutor):
            def submit(self, fn, /, *args, **kwargs):
                fn, args, kwargs = copy.deepcopy((fn, args, kwargs))
                return super().submit(fn, *args, **kwargs)
        for module in ['openfe.openfe', 'openfe.utils']:
            importlib.import_module(module).ProcessPoolExecutor = IsolatedThreadExecutor
    out = ROOT / ('full_openfe' if use_openfe else 'baseline_no_openfe')
    out.mkdir(exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    # Missing NDBI and within-park green ratio are omitted, never fabricated.
    FEATURE_GROUPS['park'][:] = ['area','shape_index','ndvi_park']
    for distance in [100,300,500]:
        FEATURE_GROUPS[f'buf{distance}'][:] = [f'ndvi_{distance}',f'building_{distance}',f'green_{distance}']
    mapping = {'area':'area_ha','shape_index':'shape_index','ndvi_park':'NDVI_green','temperature':'Tmax_C','wind_speed':'Wind_mean_ms','solar_radiation':'Solar_MJ_m2_day'}
    for distance in [100,300,500]:
        mapping.update({f'ndvi_{distance}':f'NDVI_{distance}',f'building_{distance}':f'built_frac_{distance}',f'green_{distance}':f'green_frac_{distance}'})
    raw = pd.read_csv(DATA / 'model_input.csv')
    normal = raw[raw.state == 'Normal'].groupby('green_id').mean(numeric_only=True)
    heat = raw[raw.state != 'Normal'].copy()
    rows=[]
    for _, h in heat.iterrows():
        n=normal.loc[h.green_id]
        row={'park_id':h.green_id,'pair_id':f'{h.green_id}_{h.date}','heat_date':h.date,'ci_normal':n.CI_C,'ci_heat':h.CI_C}
        for target, source in mapping.items():
            row[target+'_normal']=n[source]; row[target+'_heat']=h[source]
        rows.append(row)
    paired=compute_cr(pd.DataFrame(rows))
    assert len(paired)==144 and paired.park_id.nunique()==48
    assert np.isfinite(paired.select_dtypes('number')).all().all()
    paired.to_csv(out/'paired_input.csv',index=False)
    splits={}
    for name in ['train','val','test']:
        ids=set(pd.read_csv(DATA/'splits'/f'{name}.csv').green_id)
        splits[name]=paired[paired.park_id.isin(ids)].reset_index(drop=True)
        assert len(splits[name])>0
        splits[name].to_csv(out/f'{name}.csv',index=False)
    ids=[set(frame.park_id) for frame in splits.values()]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    assert sum(map(len,splits.values()))==len(paired)
    config={'seed':42,'device':'cpu','epochs':200,'ae_epochs':150,'use_openfe':use_openfe,'openfe_executor':'isolated single thread' if use_openfe else None,'features':FEATURE_GROUPS,'mapping':mapping,'pairing':'Per-park mean of 3 Normal observations paired with each Heatwave observation','omitted_features':['green_ratio','ndbi_100','ndbi_300','ndbi_500'],'split_counts':{k:{'pairs':len(v),'parks':v.park_id.nunique()} for k,v in splits.items()},'source_sha256':hashlib.sha256((DATA/'model_input.csv').read_bytes()).hexdigest()}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('CONFIG',json.dumps(config),flush=True)
    start=time.time()
    pipeline=SemanticTokenPipeline(use_openfe=use_openfe,ae_epochs=150,openfe_n_jobs=1,device='cpu').fit(splits['train'],splits['val'])
    print('TOKEN PIPELINE FIT COMPLETE',flush=True)
    loaders={}; tokens={}
    for name,frame in splits.items():
        n,h=pipeline.transform(frame); tokens[name]=(n,h)
        assert n.shape==(len(frame),5,8) and np.isfinite(n).all() and np.isfinite(h).all()
        np.savez_compressed(out/f'{name}_tokens.npz',normal=n,heat=h)
        loaders[name]=make_loader(n,h,frame.ci_normal,frame.ci_heat,frame.cr,shuffle=name=='train')
    loss=CoolingResilienceLoss(lambda_cr=1.0,lambda_consistency=0.2)
    model,history=fit_model(LightFEKFormerCR(token_dim=8),loaders['train'],loaders['val'],epochs=200,device='cpu',loss_fn=loss)
    metrics=evaluate(model,loaders['test'],loss,device='cpu')
    assert all(np.isfinite(v) for v in metrics.values())
    torch.save(model.state_dict(),out/'light_fe_kformer_cr.pt')
    with open(out/'token_pipeline.pkl','wb') as f: pickle.dump(pipeline,f)
    model.eval()
    with torch.no_grad():
        predictions=model(*[torch.from_numpy(x) for x in tokens['test']])
    pred=splits['test'][['park_id','pair_id','ci_normal','ci_heat','cr']].copy()
    for key,value in predictions.items():
        if key in ['ci_normal','ci_heat','cr']: pred['pred_'+key]=value.numpy().ravel()
    pred.to_csv(out/'test_predictions.csv',index=False)
    for name,obj in [('metrics',metrics),('history',history)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    pd.DataFrame(history).to_csv(out/'history.csv',index_label='epoch_zero_based')
    baseline={key:float(np.mean(np.abs(splits['test'][key]-splits['train'][key].mean()))) for key in ['ci_normal','ci_heat','cr']}
    summary={'metrics':metrics,'train_mean_baseline_mae':baseline,'epochs_completed':len(history['train_loss']),'elapsed_seconds':time.time()-start,'status':'success'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__': main()


