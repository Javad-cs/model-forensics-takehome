#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import pandas as pd

T = 20_200_000
SHUFFLE_SEED = 20260830
EXPECTED_TARGETS = ["B008","B011","B012","B036","B039","B047"]
EXPECTED_PAIRS = {"B008":9,"B011":6,"B012":7,"B036":4,"B039":10,"B047":5}

def load_jsonl(p):
    rows=[]
    with Path(p).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: rows.append(json.loads(line))
                except Exception as e: raise RuntimeError(f"Invalid JSON at {p}:{i}") from e
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a=ap.parse_args(); root=a.root.resolve()
    raw=root/'results/raw'; proc=root/'results/processed'; labels=root/'labels'
    raw_paths=[raw/'placebo_worker01.jsonl', raw/'placebo_worker23.jsonl']
    locked=labels/'placebo_outcome_extraction_blind_locked.csv'
    causal=proc/'causal_pair_effects.csv'
    for p in [*raw_paths,locked,causal]:
        if not p.exists(): raise SystemExit(f"Missing required input: {p}")
    rec=[]
    for p in raw_paths: rec.extend(load_jsonl(p))
    if len(rec)!=82: raise SystemExit(f"Expected 82 records, found {len(rec)}")
    keys=[(r['blind_id'],int(r['pair_index']),r['arm']) for r in rec]
    if len(set(keys))!=82: raise SystemExit('Duplicate placebo records')
    if not all(bool(r.get('hit_eos')) for r in rec): raise SystemExit('At least one record did not reach EOS')
    if any(bool(r.get('hit_max_tokens')) for r in rec): raise SystemExit('At least one record hit max tokens')

    order=list(range(82)); random.Random(SHUFFLE_SEED).shuffle(order)
    mapping=pd.DataFrame({'placebo_blind_id':[f'P{i:03d}' for i in range(1,83)],'record_index':order})
    lab=pd.read_csv(locked)
    if len(lab)!=82 or set(lab['status'])!={'CLEAR'}: raise SystemExit('Unexpected locked extraction')
    j=lab.merge(mapping,on='placebo_blind_id',validate='one_to_one')
    rows=[]
    for _,x in j.iterrows():
        r=rec[int(x['record_index'])]
        rows.append({'placebo_blind_id':x['placebo_blind_id'],'Y_final':int(x['Y_final']),'status':x['status'],
                     'target':r['blind_id'],'direction':r['direction'],'pair_index':int(r['pair_index']),'arm':r['arm'],
                     'continuation_seed':int(r['continuation_seed']),'source_regeneration_seed':int(r['source_regeneration_seed']),
                     'left_source_regeneration_seed':int(r['left_source_regeneration_seed']),
                     'right_source_regeneration_seed':int(r['right_source_regeneration_seed'])})
    un=pd.DataFrame(rows).sort_values(['target','pair_index','arm'])
    if sorted(un.target.unique()) != EXPECTED_TARGETS: raise SystemExit('Target mismatch')

    pairs=[]
    for (t,pi),g in un.groupby(['target','pair_index'],sort=True):
        if len(g)!=2 or set(g.arm)!={'left','right'}: raise SystemExit(f'Bad pair {t}/{pi}')
        L=int(g.loc[g.arm=='left','Y_final'].iloc[0]); R=int(g.loc[g.arm=='right','Y_final'].iloc[0]); dname=g.direction.iloc[0]
        d=1 if dname=='above_good' else -1
        pairs.append({'target':t,'direction':dname,'pair_index':int(pi),'Y_left':L,'Y_right':R,
                      'placebo_signed':d*(R-L)/T,'placebo_abs':abs(R-L)/T,'abs_diff_raw':abs(R-L)})
    pf=pd.DataFrame(pairs).sort_values(['target','pair_index'])
    if len(pf)!=41: raise SystemExit(f'Expected 41 pairs, found {len(pf)}')
    ps=pf.groupby(['target','direction']).agg(n_pairs=('pair_index','count'),mean_placebo_abs=('placebo_abs','mean'),
        median_placebo_abs=('placebo_abs','median'),max_placebo_abs=('placebo_abs','max'),mean_abs_raw=('abs_diff_raw','mean')).reset_index()

    cf=pd.read_csv(causal)
    cs=cf.groupby(['target','direction']).agg(causal_mean_signed=('pair_B_final','mean'),
        causal_mean_abs=('pair_B_final',lambda s:s.abs().mean()), causal_median_abs=('pair_B_final',lambda s:s.abs().median()),
        causal_positive=('pair_B_final',lambda s:int((s>0).sum())), causal_negative=('pair_B_final',lambda s:int((s<0).sum())),
        causal_zero=('pair_B_final',lambda s:int((s==0).sum()))).reset_index()
    comp=cs.merge(ps,on=['target','direction'],validate='one_to_one')
    comp['abs_signed_mean_over_placebo_mean_abs']=comp.causal_mean_signed.abs()/comp.mean_placebo_abs
    comp['mean_abs_causal_over_placebo_mean_abs']=comp.causal_mean_abs/comp.mean_placebo_abs

    proc.mkdir(parents=True,exist_ok=True)
    outs=[(un,proc/'placebo_outcomes_unblinded_reproduced.csv'),(pf,proc/'placebo_pair_effects_reproduced.csv'),
          (ps,proc/'placebo_target_summary_reproduced.csv'),(comp,proc/'placebo_vs_causal_summary_reproduced.csv')]
    for df,p in outs: df.to_csv(p,index=False)
    print('PLACEBO ANALYSIS REPRODUCED')
    print('Records:',len(un)); print('Pairs:',len(pf)); print('Targets:',len(ps)); print()
    for _,r in comp.iterrows():
        print(f"{r.target} | signed={100*r.causal_mean_signed:+.3f}% | placebo |drift|={100*r.mean_placebo_abs:.3f}% | |signed|/placebo={r.abs_signed_mean_over_placebo_mean_abs:.3f}")
    print('\nWROTE:'); [print(p) for _,p in outs]
if __name__=='__main__': main()
