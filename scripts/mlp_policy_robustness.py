"""Policy-class robustness: rerun the component matrix's key arms with an MLP policy
(4 hidden tanh units over [sigma, is_L]) instead of the 2-param softmax -- does the
failure class depend on the tiny policy class? (panel ds: 'deterministic small MDP
may overamplify')"""
import math, random, statistics as st, sys
sys.path.insert(0, '.')
import scripts.grpo_train_coupled as C

H_UNITS = 4
def mlp_logit(w, f):  # f=[sigma, is_L]; w: H*(2+1) + H+1 params
    z = 0.0
    for h in range(H_UNITS):
        a = math.tanh(w[h*3]*f[0] + w[h*3+1]*f[1] + w[h*3+2])
        z += w[H_UNITS*3 + h] * a
    return z + w[H_UNITS*3 + H_UNITS]

def grad_logit(w, f):
    g = [0.0]*len(w)
    for h in range(H_UNITS):
        pre = w[h*3]*f[0] + w[h*3+1]*f[1] + w[h*3+2]
        a = math.tanh(pre); da = (1-a*a) * w[H_UNITS*3+h]
        g[h*3] = da*f[0]; g[h*3+1] = da*f[1]; g[h*3+2] = da
        g[H_UNITS*3+h] = a
    g[H_UNITS*3+H_UNITS] = 1.0
    return g

def train_mlp(arm, seed, couple=1.5, floor=3.0, growth=1.25, nL=3, nS=2, H=4,
              iters=150, G=32, lr=0.05, step_cost=0.02):
    rng = random.Random(seed)
    w = [rng.gauss(0, 0.1) for _ in range(H_UNITS*3 + H_UNITS + 1)]
    for _ in range(iters):
        eps = []
        for _ in range(G):
            sig = {("L", i): rng.uniform(6.0, 9.0) for i in range(nL)}
            sig.update({("S", i): rng.uniform(0.5, 1.5) for i in range(nS)})
            alive = {k: True for k in sig}; floored = False; traj = []
            for _ in range(H):
                acts = [k for k in sig if alive[k]]
                if not acts: break
                fts = [[sig[k], 1.0 if k[0]=="L" else 0.0] for k in acts]
                logits = [mlp_logit(w, f) for f in fts]
                probs = C.softmax(logits)
                r = rng.random(); c=0.0; ci=len(acts)-1
                for j,p in enumerate(probs):
                    c+=p
                    if r<=c: ci=j; break
                choose = acts[ci]
                pre = {k: sig[k] for k in acts}; post = dict(pre); del post[choose]
                if choose[0]=="L":
                    for k in list(post):
                        if k[0]=="S": post[k]+=couple
                rew = C.reward(arm, pre, post) - step_cost
                traj.append((fts, ci, rew))
                alive[choose]=False
                if choose[0]=="L":
                    for k in sig:
                        if alive[k] and k[0]=="S": sig[k]+=couple
                for k in sig:
                    if alive[k] and k[0]=="L": sig[k]*=growth
                for k in sig:
                    if alive[k] and k[0]=="S" and sig[k]>=floor: alive[k]=False; floored=True
            rec=(not floored) and not any(alive[k] for k in sig if k[0]=="S")
            eps.append((rec,traj))
        rets=[sum(s[2] for s in tj) for _,tj in eps]
        mu=st.mean(rets); sd=st.pstdev(rets) or 1e-6
        adv=[(R-mu)/sd for R in rets]
        gacc=[0.0]*len(w)
        for (rec,tj),A in zip(eps,adv):
            for fts,ci,_ in tj:
                logits=[mlp_logit(w,f) for f in fts]; probs=C.softmax(logits)
                gci=grad_logit(w,fts[ci])
                gbar=[0.0]*len(w)
                for p,f in zip(probs,fts):
                    gf=grad_logit(w,f)
                    for d in range(len(w)): gbar[d]+=p*gf[d]
                for d in range(len(w)): gacc[d]+=A*(gci[d]-gbar[d])
        for d in range(len(w)): w[d]+=lr*gacc[d]/G
    # eval: greedy under trained MLP... use stochastic policy eval like others
    wins=0
    for _ in range(400):
        sig={("L",i):rng.uniform(6.0,9.0) for i in range(nL)}
        sig.update({("S",i):rng.uniform(0.5,1.5) for i in range(nS)})
        alive={k:True for k in sig}; floored=False
        for _ in range(H):
            acts=[k for k in sig if alive[k]]
            if not acts: break
            fts=[[sig[k],1.0 if k[0]=="L" else 0.0] for k in acts]
            probs=C.softmax([mlp_logit(w,f) for f in fts])
            r=rng.random(); c=0.0; ci=len(acts)-1
            for j,p in enumerate(probs):
                c+=p
                if r<=c: ci=j; break
            choose=acts[ci]; alive[choose]=False
            if choose[0]=="L":
                for k in sig:
                    if alive[k] and k[0]=="S": sig[k]+=couple
            for k in sig:
                if alive[k] and k[0]=="L": sig[k]*=growth
            for k in sig:
                if alive[k] and k[0]=="S" and sig[k]>=floor: alive[k]=False; floored=True
        if (not floored) and not any(alive[k] for k in sig if k[0]=="S"): wins+=1
    return wins/400

print("=== MLP-policy robustness (coupled regime, 12 seeds) ===")
for arm in ("D","E","E2"):
    fr=[train_mlp(arm, 9000+s) for s in range(12)]
    print(f"  {arm:3s}: {st.mean(fr):.3f}  (2-param softmax reference: D 1.00 / E 0.12 / E2 0.04)")
