import sys, statistics as st, json
sys.path.insert(0,'.')
import scripts.grpo_train_coupled as C
grid={}
for cp in (1.0,1.5,2.0,2.5):
    for fl in (2.5,3.0,3.5,4.0):
        d=st.mean([C.train("D",cp,fl,1.25,3,2,4,seed=3000+s)[0] for s in range(10)])
        e=st.mean([C.train("E",cp,fl,1.25,3,2,4,seed=3000+s)[0] for s in range(10)])
        grid[f"{cp},{fl}"]=(round(d,3),round(e,3),round(d-e,3))
        print(f"couple={cp} floor={fl}: D {d:.2f} E {e:.2f} gap {d-e:+.2f}")
json.dump(grid,open("figures/coupled_grid.json","w"),indent=1)
print("saved figures/coupled_grid.json")
