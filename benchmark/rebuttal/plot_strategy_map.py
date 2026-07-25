#!/usr/bin/env python3
import argparse, csv
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--map",default="benchmark/results/rebuttal/strategy_map/heuristic.csv")
p.add_argument("--output",default="benchmark/results/rebuttal/strategy_map/strategy_map.pdf")
a=p.parse_args()
with open(a.map,newline="") as f: rows=list(csv.DictReader(f))
counts=sorted({int(r["cone_count"]) for r in rows})
dims=sorted({int(r["cone_dimension"]) for r in rows})
labels=["gridWise","blockWise","warpWise","threadWise"]
z=np.full((len(dims),len(counts)),np.nan)
for r in rows:z[dims.index(int(r["cone_dimension"])),counts.index(int(r["cone_count"]))]=labels.index(r["selected_strategy"])
fig,ax=plt.subplots(figsize=(8,5))
im=ax.imshow(z,origin="lower",aspect="auto",vmin=-.5,vmax=3.5,cmap="viridis")
ax.set_xticks(range(len(counts)),counts,rotation=45);ax.set_yticks(range(len(dims)),dims)
ax.set_xlabel("Cone count");ax.set_ylabel("Full cone dimension")
c=fig.colorbar(im,ax=ax,ticks=range(4));c.ax.set_yticklabels(labels)
fig.tight_layout();fig.savefig(a.output)
