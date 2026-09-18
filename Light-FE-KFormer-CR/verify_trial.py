from pathlib import Path
import os,sys,json,pickle,copy,importlib
from concurrent.futures import ThreadPoolExecutor
PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT.parent/'result'
os.chdir(ROOT)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache'/'matplotlib')
os.environ['TEMP']=os.environ['TMP']=str(ROOT/'tmp')
sys.dont_write_bytecode=True
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from light_fe_kformer_cr.openfe_stage import FEATURE_GROUPS
from light_fe_kformer_cr.model import LightFEKFormerCR
class IsolatedThreadExecutor(ThreadPoolExecutor):
    def submit(self,fn,/,*args,**kwargs):
        fn,args,kwargs=copy.deepcopy((fn,args,kwargs))
        return super().submit(fn,*args,**kwargs)
for module in ['openfe.openfe','openfe.utils']:
    importlib.import_module(module).ProcessPoolExecutor=IsolatedThreadExecutor
torch.set_num_threads(4)
results={}
for variant in ['baseline_no_openfe','full_openfe']:
    out=ROOT/variant
    config=json.loads((out/'config.json').read_text())
    FEATURE_GROUPS.clear();FEATURE_GROUPS.update(config['features'])
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='success'
    with open(out/'token_pipeline.pkl','rb') as f: pipeline=pickle.load(f)
    test=pd.read_csv(out/'test.csv')
    n,h=pipeline.transform(test)
    cached=np.load(out/'test_tokens.npz')
    np.testing.assert_allclose(n,cached['normal'],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(h,cached['heat'],rtol=1e-5,atol=1e-5)
    model=LightFEKFormerCR(token_dim=8)
    model.load_state_dict(torch.load(out/'light_fe_kformer_cr.pt',map_location='cpu',weights_only=True));model.eval()
    with torch.no_grad(): p=model(torch.from_numpy(n),torch.from_numpy(h))
    saved=pd.read_csv(out/'test_predictions.csv')
    for key in ['ci_normal','ci_heat','cr']:
        np.testing.assert_allclose(p[key].numpy().ravel(),saved['pred_'+key],rtol=1e-5,atol=1e-5)
    formulas={k:e.feature_formulas_ if e is not None else [] for k,e in pipeline.enhancers_.items()}
    (out/'selected_features.json').write_text(json.dumps(formulas,indent=2),encoding='utf-8')
    results[variant]={'reload_and_predictions':'passed','token_shape':list(n.shape),'selected_feature_counts':{k:len(v) for k,v in formulas.items()},'summary':summary}
(ROOT/'verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))

