import matplotlib.pyplot as plt
import numpy as np
import os

modellist = [
    #"/data/duolin/splm_gvp/results/config_comparewithgit/config_1/2024-03-07__23-36-23/",
    "/data/duolin/splm_gvp/results/config_comparewithgit/config_nomixedprecision/2024-03-11__11-47-37/",
    #"/data/duolin/splm_gvpgit/splm/results/config_1/2024-03-08__14-30-37/", 
    "/data/duolin/splm_gvpgit/splm/results/config_nomixedprecision/2024-03-11__11-50-11/", #same data.py and add simclr.train()
    "/data/duolin/splm_gvpgit/splm/results/config_1_bf16/2024-03-11__14-39-18/",
    "/data/duolin/splm_gvpgit/splm/results/config_2_fp16/2024-03-11__14-43-48/"
    
]


train_losslist = {}
val_losslist = {}
steps = {}
losskey = "residue_loss:"
#losskey = "lr:"
# losskey="loss:"
# losskey="Loss:"
# losskey="ST_S_loss:"
# losskey ="S_ESMS_loss:"
# losskey = "Train_loss_avg:"
for model in modellist:
    train_losslist[model] = []
    val_losslist[model] = []
    steps[model] = []
    if os.path.exists(model + "/" + "training.log"):
        file = model + "/" + "training.log"
    else:
        file = model + "/" + "logs.txt"
    
    input = open(file)
    for line in input:
        if "step" not in line:
            continue
        
        if "MLM" in model:
            MLMkey = "Train_simclr_loss_avg:"
            # MLMkey="Train_MLM_loss_avg"
            if MLMkey in line:
                valloss = line.split(MLMkey)[1].strip().split()[0].split(",")[0]
                val_losslist[model].append(float(valloss))
                step = int(line.split("step:")[1].split()[0])
                steps[model].append(int(step))
        else:
            if losskey in line:
                valloss = line.split(losskey)[1].strip().split()[0].split(",")[0]
                val_losslist[model].append(float(valloss))
                step = int(line.split("step:")[1].split()[0])
                
                steps[model].append(int(step))
        
        if "graph_loss" in line:
            trainloss = line.split("graph_loss:")[1].strip().split()[0].split(",")[0]
            train_losslist[model].append(float(trainloss))

# plots:
# get max step:
plt.cla()
grid = plt.GridSpec(2, 1, wspace=0.1, hspace=0.4)
fig = plt.figure(figsize=(10, 8))

maxstep = 0
for model in train_losslist:
    stepnum = len(train_losslist[model])
    if stepnum > maxstep:
        maxstep = stepnum

ct = ["blue", "red", "black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink", "cyan",
      "peru", "darkgray", "slategray", "gold", "darkseagreen", "purple", "thistle", "crimson"]

index = 0
ax = fig.add_subplot(grid[0])
for model in train_losslist:
    ax.plot(steps[model], train_losslist[model], '--', c=ct[index], label="/".join(model.split("/")[-3:-1]))
    ax.plot(steps[model], val_losslist[model], c=ct[index], label="/".join(model.split("/")[-3:-1]))
    index += 1
    # plt.legend()

ymax=10
plt.ylim(0,ymax)
#plt.xlim(0,1000)
#plt.legend()
#plt.show()
plt.legend(loc='lower center',bbox_to_anchor=(0.3,-1.5), fancybox=True, shadow=True, ncol=2,prop={'size': 6})
plt.tight_layout()
plt.savefig("loss"+str(ymax)+".png")
#plt.savefig("loss.png")