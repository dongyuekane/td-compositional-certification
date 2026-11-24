"""Residual-centered output enclosure using an existing state certificate.

REAL-ARITHMETIC PROOF
--------------------
Let F(x*)=0, e=x*-x, r=-C F(x), and A=I-C J(x).  Assume the
existing single-state certificate proves ||e||_inf<=E<=R and provides
a>=||A||_inf and h bounding the equationwise entrywise-l1 Hessian norm
of -C F throughout the radius-R ball. Taylor's integral identity gives

    e = r + A e + integral_0^1 (1-t) H_{-CF}(x+t e)[e,e] dt.

Therefore ||x*-(x+r)||_inf <= a E + h E^2/2. This encloses the SAME
root already certified in the original ball; it requires neither a new
existence test at x+r nor a new power-flow solve. The corrected point need
not itself be feasible, and feasibility must not be inferred from it alone.
For a quadratic output q, apply its exact Taylor bound at x+r with the
new radius. The old E must not simply be reused as an estimate of the new
radius, and h must remain the valid original regional bound.

The implementation additionally propagates the residual-evaluation error,
matrix-vector multiplication error, and rounded construction of x+r. Its
numerical interpretation is the same guarded binary64 interpretation as
current_certificates.py, rather than formal interval verification.

This is a standard residual/defect correction baseline, not a claimed new
power-flow algorithm. The preconditioner C and original certificate are
reused. With cached r and its enclosure, the added work consists only of
scalar bounds, a vector addition, and a quadratic-output evaluation.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from certificates import dot_enclosure, gamma, norm_inf, quantity_interval, up


def current_residual_enclosure(model, x):
    """Same residual guard as CurrentModel.enclosures, without forming J."""
    v,z=model.parts(x)
    f=model.residual(x)
    current_scale=np.abs(model.Y)@np.abs(v)+np.abs(model.y0)+np.abs(z)/np.abs(v)
    magnitude_scale=(abs(v[model.pvid])**2
                     +model.system.vm[model.system.ids][model.pvid]**2)
    scale=np.r_[current_scale,current_scale,magnitude_scale]
    radius=gamma(6*model.m+32)*scale
    return f,up(radius+1e-300)


def corrected_state_enclosure(model,x,C,certificate,correction=None,
                              correction_radius=None):
    """Return the corrected center and an error radius for the existing root.

    Optional correction/correction_radius are the cached r,rr returned by
    current_preconditioned(model,x,C). If supplied, both must be supplied.
    """
    if not certificate.get('certified'):
        return dict(certified=False,reason='no_original_state_certificate')
    if (correction is None)!=(correction_radius is None):
        raise ValueError('Supply both correction and correction_radius, or neither.')
    x=np.asarray(x,dtype=float)
    E=float(certificate['error']);R=float(certificate['radius'])
    a=float(certificate['linear_bound']);h=float(certificate['hessian_bound'])
    if not (0<=E<=R and 0<=a and 0<=h):
        raise ValueError('Invalid original certificate bounds.')
    if correction is None:
        f,fr=current_residual_enclosure(model,x)
        r,rr=dot_enclosure(-np.asarray(C),f,fr)
    else:
        r=np.asarray(correction,dtype=float)
        rr=np.asarray(correction_radius,dtype=float)
    center=x+r
    addition_radius=up(gamma(3)*(np.abs(x)+np.abs(r)))
    nonlinear=float(up((a*E+.5*h*E*E)*(1+gamma(12))))
    arithmetic=norm_inf(rr+addition_radius)
    radius=float(up((nonlinear+arithmetic)*(1+gamma(6))))
    return dict(certified=True,center=center,error=radius,
                original_error=E,analytic_remainder=nonlinear,
                correction_roundoff=arithmetic,correction_norm=norm_inf(r),
                semantics='Output enclosure for the root certified at the original center.')


def corrected_loss_interval(model,x,C,certificate,quantity=None,
                            correction=None,correction_radius=None):
    """Quadratic loss interval around x-CF(x), with a valid remainder radius."""
    state=corrected_state_enclosure(model,x,C,certificate,correction,correction_radius)
    if not state['certified']:
        return dict(certified=False,lower=-np.inf,upper=np.inf,error=np.inf,
                    reason=state['reason'])
    q=model.loss_quantity() if quantity is None else quantity
    result=quantity_interval(q,state['center'],state['error'])
    result.update(certified=True,corrected_state_error=state['error'],
                  original_state_error=state['original_error'],
                  analytic_state_remainder=state['analytic_remainder'],
                  correction_roundoff=state['correction_roundoff'],
                  correction_norm=state['correction_norm'])
    return result


def test_two_networks():
    """Compare the fairer single-state baseline; do not presume pairing wins."""
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'simulation'))
    from ac import build_network
    from current_certificates import (CurrentModel,current_single_certificate,
        current_paired_certificate)
    from certificates import difference_interval
    records=[]
    for family in ['mixed9','mixed30']:
        pi=dict(T=0.,X=.5,F=1,S=1,R=1,C=1)
        pj=dict(pi,X=1.)
        si=build_network(family,load=1.15,plan=pi)
        sj=build_network(family,load=1.15,plan=pj)
        ai=si.solve(trace=True);aj=sj.solve(trace=True)
        assert ai['converged'] and aj['converged']
        mi=CurrentModel(si);mj=CurrentModel(sj)
        qi=mi.loss_quantity();qj=mj.loss_quantity()
        truth=ai['loss']-aj['loss']
        star_i=mi.augment(ai['x']);star_j=mj.augment(aj['x'])
        for k in [2,3,4]:
            xi=mi.augment(ai['trace'][min(k,len(ai['trace'])-1)])
            xj=mj.augment(aj['trace'][min(k,len(aj['trace'])-1)])
            Ci=np.linalg.inv(mi.jacobian(xi));Cj=np.linalg.inv(mj.jacobian(xj))
            ci=current_single_certificate(mi,xi,Ci)
            cj=current_single_certificate(mj,xj,Cj)
            record=dict(family=family,checkpoint=k,true_loss_difference=truth,
                        single_certified=ci['certified'] and cj['certified'])
            if not record['single_certified']:
                records.append(record);continue
            old_i=quantity_interval(qi,xi,ci['error'])
            old_j=quantity_interval(qj,xj,cj['error'])
            start=time.perf_counter()
            ni=corrected_loss_interval(mi,xi,Ci,ci,qi)
            nj=corrected_loss_interval(mj,xj,Cj,cj,qj)
            overhead=time.perf_counter()-start
            corrected_lower=float(np.nextafter(ni['lower']-nj['upper'],-np.inf))
            corrected_upper=float(up(ni['upper']-nj['lower']))
            corrected_width=corrected_upper-corrected_lower
            old_width=2*(old_i['error']+old_j['error'])
            assert ni['lower']<=ai['loss']<=ni['upper']
            assert nj['lower']<=aj['loss']<=nj['upper']
            assert corrected_lower<=truth<=corrected_upper
            ni_state=corrected_state_enclosure(mi,xi,Ci,ci)
            nj_state=corrected_state_enclosure(mj,xj,Cj,cj)
            assert max(abs(star_i-ni_state['center']))<=ni_state['error']+1e-10
            assert max(abs(star_j-nj_state['center']))<=nj_state['error']+1e-10
            start=time.perf_counter()
            pc=current_paired_certificate(mi,xi,mj,xj,Ci,Cj)
            raw_pair=difference_interval(qi,xi,qj,xj,pc)
            pair_time=time.perf_counter()-start
            record.update(original_independent_width=old_width,
                          corrected_independent_lower=corrected_lower,
                          corrected_independent_upper=corrected_upper,
                          corrected_independent_width=corrected_width,
                          corrected_over_original_width=corrected_width/old_width,
                          corrected_center=(corrected_lower+corrected_upper)/2,
                          corrected_i=ni,corrected_j=nj,
                          corrected_extra_seconds=overhead,
                          paired_extra_seconds=pair_time,
                          paired_certified=pc['certified'])
            if pc['certified']:
                pair_width=raw_pair['upper']-raw_pair['lower']
                record.update(raw_paired_width=pair_width,
                              corrected_over_raw_pair_width=corrected_width/pair_width)
            records.append(record)
            print(json.dumps({key:record[key] for key in [
                'family','checkpoint','original_independent_width',
                'raw_paired_width','corrected_independent_width',
                'corrected_over_raw_pair_width','corrected_extra_seconds',
                'paired_extra_seconds'] if key in record}),flush=True)
    path=Path(__file__).with_name('corrected_output_results.json')
    path.write_text(json.dumps(records,indent=2))
    return records


if __name__=='__main__':
    test_two_networks()
