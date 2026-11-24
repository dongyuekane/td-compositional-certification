"""Independent numerical diagnostics of already-constructed analytic bounds."""
from feeder_tube import *
from atlas import Response,feeder_at

def verify():
    tests=0;failures=[];ratios=[]
    for feeder in (0,1):
     for degree in (6,10):
      for X,F,tap,cap in [(0.,0,1.04,0.),(.5,1,1.075,1.2),(1.,1,1.1,1.2)]:
        response=Response(feeder_at(feeder,X,F,tap,cap),degree)
        cert=certify_response(response);assert cert['certified']
        s=response.system;original=s.spec.copy();ids=s.ids
        demand=-original[ids];dc=np.conj(demand);Y=s.Y[np.ix_(ids,ids)]
        for z in np.linspace(.4,1.8,13):
            s.spec=original*z;ans=s.solve(tol=1e-11,maxiter=30)
            assert ans['converged'];w=ans['voltage'][ids];wb=np.conj(w)
            c=-z*dc/wb**2
            J=np.block([[Y.real+np.diag(c.real),-Y.imag+np.diag(c.imag)],
                        [Y.imag+np.diag(c.imag),Y.real-np.diag(c.real)]])
            rhs=-dc/wb;pr=solve(J,np.r_[rhs.real,rhs.imag],check_finite=False)
            p=pr[:s.m]+1j*pr[s.m:]
            rhs2=2*dc*np.conj(p)/wb**2-2*z*dc*np.conj(p)**2/wb**3
            p2r=solve(J,np.r_[rhs2.real,rhs2.imag],check_finite=False)
            p2=p2r[:s.m]+1j*p2r[s.m:]
            value,derivative=response.evaluate(z)
            wh=value[2:2+s.m]+1j*value[2+s.m:]
            ph=derivative[2:2+s.m]+1j*derivative[2+s.m:]
            true_sd=np.conj(s.Y[s.slack,ids]@p)
            sdhat=complex(derivative[0],derivative[1])
            true_s2=np.conj(s.Y[s.slack,ids]@p2)
            ident=(feeder,degree,X,F,tap,cap,float(z))
            def check(name,actual,bound,tolerance=1e-10):
                if np.any(np.asarray(actual)>np.asarray(bound)+tolerance):failures.append(dict(case=ident,quantity=name,max_excess=float(np.max(np.asarray(actual)-bound))))
            check('voltage_derivative_error',abs(p-ph),cert['voltage_derivative_error'])
            check('power_derivative_error',abs(true_sd-sdhat),cert['power_derivative_error'])
            check('power_second_derivative',abs(true_s2),cert['power_second_derivative_bound'])
            check('voltage_derivative_bound',abs(p),cert['voltage_derivative_bound'])
            for matrix,key in [(s.Yf,'current_from'),(s.Yt,'current_to')]:
                exact=matrix@ans['voltage'];pred=matrix@np.r_[1+0j,wh]
                exact_d=matrix[:,ids]@p;exact_d2=matrix[:,ids]@p2
                check(key+'_error',abs(exact-pred),cert[key]['error'],2e-9)
                check(key+'_derivative',abs(exact_d),cert[key]['derivative_bound'])
                check(key+'_second_derivative',abs(exact_d2),cert[key]['second_derivative_bound'])
            ratios.append(abs(true_sd-sdhat)/cert['power_derivative_error']);tests+=1
        s.spec=original
    result=dict(tested_points=tests,failures=failures,max_observed_derivative_error_fraction=float(max(ratios)),
                statement='Diagnostics only. Bounds were formed analytically before these test points.')
    Path(__file__).with_name('feeder_derivative_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert not failures

if __name__=='__main__':verify()
