from pathlib import Path
import sys,json,itertools,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control_blocks'))
from atlas import Atlas,Reduced,reference,SEED
from feeder_tube import certify_response
from coupled_certificate import certify_main,classify,slack_loss_interval
import numpy as np

def main():
    allrows=[]
    for degree in (6,10):
        a=Atlas(degree);t=time.perf_counter()
        tubes={key:certify_response(r) for key,r in a.responses.items()}
        training=a.seconds+time.perf_counter()-t
        assert all(c['certified'] for c in tubes.values())
        print('trained',degree,training,flush=True)
        for fam,load in [('mixed9',.9),('mixed9',1.25),('mixed30',.6),('mixed30',.8)]:
          for T,X,F in itertools.product((0.,1.),(0.,1.),(0,1)):
            p=dict(T=T,X=X,F=F,S=0,R=1,C=1)
            for c in list(reference.controls(p))[::7]:
                s=reference.build(fam,load,p,c,shape_seed=SEED);model=Reduced(s,a)
                r=model.solve();cs=[tubes[k,p['X'],p['F'] if k==0 else 0,c['taps'][k],c['caps'][k]] for k in (0,1)]
                start=time.perf_counter();cert=certify_main(model,r,cs)
                result=classify(model,r,cert,cs);interval=slack_loss_interval(model,r,cert)
                overhead=time.perf_counter()-start
                exact=s.solve();m=s.metrics(exact['x']);violation=max((.9-m['vmin'])/.1,(m['vmax']-1.1)/.1,m['thermal']-1,m['genviol'])
                mainerr=float(max(abs(exact['voltage'][:model.main.nb]-r['voltage'][:model.main.nb])))
                record=dict(degree=degree,family=fam,load=load,T=T,X=X,F=F,**c,certificate=cert,
                            classification=result,loss_interval=list(interval),exact_loss=exact['loss'],exact_violation=violation,
                            observed_main_voltage_error=mainerr,certificate_seconds=overhead)
                if cert['certified']:
                    assert mainerr<=np.sqrt(2)*cert['error']+1e-9,record
                    assert interval[0]-1e-9<=exact['loss']<=interval[1]+1e-9,record
                    assert result['violation_lower']-1e-8<=violation<=result['violation_upper']+1e-8,record
                allrows.append(record)
        rows=[r for r in allrows if r['degree']==degree]
        print(json.dumps(dict(degree=degree,states=len(rows),certified=sum(r['certificate']['certified'] for r in rows),
                              decisions={k:sum(r['classification']['status']==k for r in rows) for k in ['feasible','violating','unresolved']},
                              median_cert_seconds=np.median([r['certificate_seconds'] for r in rows]),
                              max_main_bound=max(r['certificate'].get('error',0) for r in rows),
                              max_loss_width=max(r['loss_interval'][1]-r['loss_interval'][0] for r in rows))),flush=True)
    Path(__file__).with_name('certificate_pilot.json').write_text(json.dumps(allrows,indent=2))

if __name__=='__main__':main()
