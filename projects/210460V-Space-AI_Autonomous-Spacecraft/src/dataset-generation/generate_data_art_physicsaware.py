import os, sys, random as rnd, numpy as np, matplotlib.pyplot as plt

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from tqdm import tqdm
from dynamics.orbit_dynamics import dynamics_roe_optimization, roe_to_rtn_horizon
from optimization.rpod_scenario import oe_0_ref, t_0, E_koz, n_time_rpod
from optimization.ocp import ocp_cvx

# --- sampling & CVX config ---

# size / attempts
TARGET_SAMPLES = 10000       # Sample size
MAX_TRIALS     = 15000       # Max attempts to reach target (Better keep target size / max attempts ratio =  2/3)
RNG_SEED       = 2030        # Random seed for reproducibility

# OCP horizon (in orbits)
HORIZON_GRID   = np.linspace(1.0, 3.0, 16)

# Relative semi-major axis [m]  & mean longitude [m]
DA_GRID        = np.linspace(-5.0, 5.0, 41)
DLAM_GRID      = np.linspace(-100.0, 100.0, 41)

# Eccentricity / inclination magnitude [m] relative to KOZ radii
rr_koz = 60    # typical radial KOZ size (m)
rt_koz = 94
rn_koz = 123

DE_MAG_GRID    = rr_koz + np.linspace(5, 30, 20)   # [m]
DI_MAG_GRID    = rn_koz + np.linspace(5, 30, 20)   # [m]

# Orientation phases (90 ± 5 degrees)
PHI_GRID       = (90 + np.linspace(-5, 5, 21)) * np.pi/180


# --- helpers ---
def set_seeds(s): rnd.seed(s); np.random.seed(s)
def sample_initial_roe():
    da = rnd.choice(DA_GRID)
    dl = rnd.choice(DLAM_GRID)
    de = rnd.choice(DE_MAG_GRID)
    di = rnd.choice(DI_MAG_GRID)
    ph_de = rnd.choice(PHI_GRID)
    ph_di = rnd.choice(PHI_GRID)
    return np.array([
        da,
        dl,
        de * np.cos(ph_de),
        de * np.sin(ph_de),
        di * np.cos(ph_di),
        di * np.sin(ph_di)
    ], float)

def safe_time_slice(stm,cim,psi,oe,time):
    T=min(stm.shape[-1],cim.shape[-1],psi.shape[-1],oe.shape[1],time.shape[0])
    return stm[...,:T],cim[...,:T],psi[...,:T],oe[:,:T],time[:T],T

def try_cvx(stm,cim,psi,s0,T):
    try:
        ret=ocp_cvx(stm,cim,psi,s0,T)
        if isinstance(ret,(tuple,list))and len(ret)>=3:
            s,a,st=ret[0],ret[1],ret[2]
        elif isinstance(ret,dict):
            s,a,st=ret.get('states_roe')or ret.get('x'),ret.get('actions')or ret.get('u'),ret.get('status')or ret.get('feasible')
        else: return None,None,'bad'
        return (s,a,st) if str(st).lower()=='optimal' else (None,None,st)
    except: return None,None,'fail'

def compute_rtg(a): step=np.linalg.norm(a,1,axis=1); return -np.cumsum(step[::-1])[::-1]

def compute_ctg(s_rtn,E):
    pos=s_rtn[:,:3]; vals=np.einsum('ti,ij,tj->t',pos,E,pos)
    viol=(vals<1).astype(int); return np.cumsum(viol[::-1])[::-1].astype(float)

# --- main ---
def main():
    set_seeds(RNG_SEED)
    T_nom=int(n_time_rpod); nS,nA=6,3
    data=dict(states_roe=[],states_rtn=[],actions=[],time=[],oe=[],
              dtime=[],horizon=[],rtg=[],ctg=[])
    kept=0; trials=0
    pbar=tqdm(total=TARGET_SAMPLES,desc="[cvx-gen]")
    while kept<TARGET_SAMPLES and trials<MAX_TRIALS:
        trials+=1
        hrz=rnd.choice(HORIZON_GRID); s0=sample_initial_roe()
        stm,cim,psi,oe,time_vec,dt=dynamics_roe_optimization(oe_0_ref,t_0,hrz,n_time_rpod)
        stm,cim,psi,oe,time_vec,T=safe_time_slice(stm,cim,psi,oe,time_vec)
        if T<3 or not np.isfinite(dt) or dt<=0: continue
        s,a,st=try_cvx(stm,cim,psi,s0,T)
        if s is None: continue
        try: s_rtn=roe_to_rtn_horizon(s,oe,T)
        except: continue
        s_roe_T,s_rtn_T,a_T=np.transpose(s),np.transpose(s_rtn),np.transpose(a)
        # pad/crop
        if T<T_nom:
            pad=T_nom-T
            s_roe_T=np.vstack([s_roe_T,np.repeat(s_roe_T[-1:],pad,axis=0)])
            s_rtn_T=np.vstack([s_rtn_T,np.repeat(s_rtn_T[-1:],pad,axis=0)])
            a_T=np.vstack([a_T,np.zeros((pad,3))])
            time_T=np.hstack([time_vec,np.repeat(time_vec[-1],pad)])
            oe_T=np.vstack([np.transpose(oe),np.repeat(np.transpose(oe)[-1:],pad,axis=0)])
        else:
            time_T=time_vec[:T_nom]; oe_T=np.transpose(oe)[:T_nom]
        rtg_T=compute_rtg(a_T); ctg_T=compute_ctg(s_rtn_T,E_koz)
        for k,v in zip(['states_roe','states_rtn','actions','time','oe','dtime','horizon','rtg','ctg'],
                       [s_roe_T,s_rtn_T,a_T,time_T,oe_T,dt,hrz,rtg_T,ctg_T]):
            data[k].append(v)
        kept+=1; pbar.update(1)
    pbar.close()

    # stack + save
    for k in data: data[k]=np.array(data[k])

    out_dir=os.path.join(root_folder,'dataset-seed-'+ str(RNG_SEED))
    os.makedirs(out_dir,exist_ok=True)

    out_name = os.path.join(out_dir, f'dataset-rpod-cvx')

    np.savez_compressed(f'{out_name}.npz',
                        states_roe_cvx=data['states_roe'],
                        states_rtn_cvx=data['states_rtn'],
                        actions_cvx=data['actions'])
    np.savez_compressed(f'{out_name}-param.npz',
                        time=data['time'],oe=data['oe'],
                        dtime=data['dtime'],horizons=data['horizon'],
                        rtg=data['rtg'],ctg=data['ctg'])
    print(f"\nFeasible {len(data['horizon'])}/{trials}")

    # --- plotting preview ---
    print("Plotting 5 random trajectories ...")
    fig,ax=plt.subplots(1,1,figsize=(6,6))
    theta=np.linspace(0,2*np.pi,200)
    # draw KOZ ellipse for reference
    if E_koz.shape==(3,3):
        # parametric circle in body frame, scale to ellipse via sqrtm
        try:
            from scipy.linalg import sqrtm
            M=sqrtm(np.linalg.inv(E_koz))
            ell=M@np.vstack([np.cos(theta),np.sin(theta),np.zeros_like(theta)])
            ax.plot(ell[0,:],ell[1,:],'k--',lw=1,label='KOZ boundary')
        except Exception:
            ax.add_patch(plt.Circle((0,0),1.0,fill=False,color='k',ls='--'))
    sel=np.random.choice(len(data['horizon']),size=min(5,len(data['horizon'])),replace=False)
    for i in sel:
        s=data['states_rtn'][i]; rtg=data['rtg'][i]
        ax.scatter(s[:,1],s[:,0],c=rtg,cmap='coolwarm',s=8,label=f'Traj {i}')
    ax.set_xlabel('Along-track [m]'); ax.set_ylabel('Radial [m]')
    ax.set_title('Sample RTN trajectories (color = RTG)')
    ax.legend(); ax.axis('equal'); plt.show()

if __name__=="__main__":
    main()
