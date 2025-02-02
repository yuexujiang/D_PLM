import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
modellist=[
        #"results/config_16adapterH_plddtweight/2024-02-27__09-08-20",#cath wrong
        #"results/config_16adapterH_plddtweight/2024-02-28__19-52-52",#cath wrong
        "results/config_16adapterH_plddtweight/2024-03-04__08-52-19",
        "results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run1/2024-02-26__11-30-44/", #cath wrong
        "results/config_16adapterH_100random_500cyclcstep/2024-02-27__13-53-30/", #cath wrong
        #"results/config_12adapterH_100random_500cyclcstep/run0_batch20/",
        "results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2", ##cath wrong
        "results/config_12adapterH_2000random/2024-02-28__23-36-08", ##cath wrong
        "results/config_12adapterH_2000random_run2test/2024-03-04__08-52-44",
        "results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run2/2024-03-01__15-19-50",
        "results/config_12adapterH_100random_500cyclcstep/2024-03-02__22-15-43/",
        "results/config_12adapterH_100random_500cyclcstep/2024-03-02__22-19-15/",
        "results/config_12adapterH_2000random_noseq/2024-03-04__12-46-41/",
        "results/config_12adapterH_plddtweight_noseq/2024-03-04__13-35-37",
        "results/config_12adapterH_new_plddtweight/2024-03-04__11-06-55",
]

modellist=[
        "results/config_12adapterH_2000random/2024-02-28__23-36-08", ##cath wrong
        "results/config_12adapterH_2000random_run2test/2024-03-04__08-52-44",
        "results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run2/2024-03-01__15-19-50",
        "results/config_12adapterH_100random_500cyclcstep/2024-03-02__22-15-43/",
        "results/config_12adapterH_2000random_noseq/2024-03-04__12-46-41/",
        "results/config_12adapterH_plddtweight_noseq/2024-03-04__13-35-37",
        "results/config_12adapterH_new_plddtweight/2024-03-04__11-06-55",
        "results/config_12adapterH_plddtposweight_noseq/2024-03-05__12-28-26",
        "results/config_12adapterH_2000random/seqdim64/2024-03-05__23-11-44",
        "results/config_12adapterH_2000random_pcaseq/2024-03-06__16-34-04/",#batch32
        "results/config_12adapterH_2000random_pcaseq/2024-03-06__16-29-43/" #batch64

]

cathwronglist=["results/config_12adapterH_2000random/2024-02-28__23-36-08",
"results/config_16adapterH_plddtweight/2024-02-27__09-08-20",
"results/config_16adapterH_plddtweight/2024-02-28__19-52-52",
"results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run1/2024-02-26__11-30-44/",
"results/config_16adapterH_100random_500cyclcstep/2024-02-27__13-53-30/",
"results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2"

]

pool_mode=2
kinase_score={}
deaminase_score={}
cath_score1={}
cath_score2={}
cath_score3={}

swin_cath_score1={}
swin_cath_score2={}
swin_cath_score3={}

kinase_score_2d={}
deaminase_score_2d={}
cath_score1_2d={}
cath_score2_2d={}
cath_score3_2d={}

steps={}

for model in modellist:
    kinase_score[model]=[]
    deaminase_score[model]=[]
    cath_score1[model]=[]
    cath_score2[model]=[]
    cath_score3[model]=[]
    swin_cath_score1[model]=[]
    swin_cath_score2[model]=[]
    swin_cath_score3[model]=[]
    
    steps[model]=[]
    file = model+"/"+"evaluate_clustering.txt"
    if not os.path.exists(file):
       file = model+"/"+"logs.txt"
    
    input = open(file)
    for line in input:
        if "step" not in line:
             continue
        
        if "kinase_score:" in line:
            kinase=line.split("ARI:")[1].strip()
            kinase_score[model].append(float(kinase))
            step=line.split("step:")[1].split()[0]
            steps[model].append(int(step))
        
        if "deaminase_score:" in line:
            deaminase = line.split("ARI:")[1].strip()
            deaminase_score[model].append(float(deaminase))
        
        if "digit_num_1_ARI:" in line and "gvp" not in line:
            digit_num_1 = line.split("digit_num_1_ARI:")[1].strip().split()[0]
            digit_num_2 = line.split("digit_num_2_ARI:")[1].strip().split()[0]
            digit_num_3 = line.split("digit_num_3_ARI:")[1].strip().split()[0]
            
            cath_score1[model].append(float(digit_num_1))
            cath_score2[model].append(float(digit_num_2))
            cath_score3[model].append(float(digit_num_3))
        
        if "gvp_digit_num_1_ARI:" in line:
            swin_digit_num_1 = line.split("gvp_digit_num_1_ARI:")[1].strip().split()[0]
            swin_digit_num_2 = line.split("gvp_digit_num_2_ARI:")[1].strip().split()[0]
            swin_digit_num_3 = line.split("gvp_digit_num_3_ARI:")[1].strip().split()[0]
            
            swin_cath_score1[model].append(float(swin_digit_num_1))
            swin_cath_score2[model].append(float(swin_digit_num_2))
            swin_cath_score3[model].append(float(swin_digit_num_3))


def sliding_window_average(values, window_size):
    half_window = window_size // 2
    smoothed_values = []
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        window = values[start:end]
        window_average = sum(window) / len(window)
        smoothed_values.append(window_average)
    return smoothed_values



ct=["blue","red","black","orange","green","#9400D3","gray","magenta",'#808000',"hotpink","yellow","peru","darkgray","slategray","gold","darkseagreen","purple","thistle","crimson","cyan","pink"]


ct = [
     '#FFA500', '#FF0000','#FFFF00', '#00FF00', '#00FFFF',
    '#0000FF', '#8A2BE2', '#FF00FF', 'gray', '#008000',
    'black','#4B0082', '#FF4500', '#FFD700', '#87CEEB',
    '#000080', '#800000', '#FF69B4', '#808000', '#9400D3','#32CD32',
]

titles=["cath_digit_1_ARI","cath_digit_2_ARI","cath_digit_3_ARI","struct_cath_digit_1_ARI","struct_cath_digit_2_ARI","struct_cath_digit_3_ARI","deaminase_ARI","kinase_ARI","AVG_ARI"]

grid = plt.GridSpec(6, 3, wspace=0.1, hspace=0.4)
fig = plt.figure(figsize=(9, 10))
drawscores=[cath_score1,cath_score2,cath_score3,swin_cath_score1,swin_cath_score2,swin_cath_score3,deaminase_score,kinase_score]
ESM2scores=[0.005697,0.012342,0.0368,0.005697,0.012342,0.0368,0.6257,0.27798]
swin_bestscores=[0.11779079162009275,0.18588589455713994,0.303310961567756,0.11779079162009275,0.18588589455713994,0.303310961567756,0.6257,0.27798]
gvp_bestscores=[0.48321427337808287,0.2476870837029018,0.23990534005799072,0.48321427337808287,0.2476870837029018,0.23990534005799072,0.6257,0.27798]
average_scores = {}
for drawscore in drawscores:
  index=0
  taskindex=drawscores.index(drawscore)
  ax = fig.add_subplot(grid[taskindex])
  
  ax.set_title(titles[taskindex],fontsize=7)
  ax.tick_params(axis='both', which='major', labelsize=7)  # Set font size to 12
  for model in drawscore:
    if model in cathwronglist and taskindex in [3,4,5]:
         index+=1
         continue
    
    if model not in average_scores:
          average_scores[model]=drawscore[model]
    else:
        if taskindex not in [3,4,5]:
          average_scores[model] = [x + y for x, y in zip(average_scores[model], drawscore[model])]
    
    drawscore[model]=sliding_window_average(drawscore[model],5)
    minsteps=np.min([len(drawscore[model]),len(steps[model])])
    ax.plot(steps[model][:minsteps],drawscore[model][:minsteps],c=ct[index],label=model.split("/")[1],lw=1)
    index+=1
  
  plt.axhline(y=ESM2scores[drawscores.index(drawscore)], color="red", linestyle='--', label='ESM2 650') 
  plt.axhline(y=gvp_bestscores[drawscores.index(drawscore)], color="green", linestyle='--', label='gvp_random') 
  plt.axhline(y=swin_bestscores[drawscores.index(drawscore)], color="darkgreen", linestyle='--', label='swin_contactmap') 
  #plt.ylim(0,6)

for model in modellist:
    average_scores[model] = [x/len(drawscores) for x in average_scores[model]] 


taskindex+=1
ax = fig.add_subplot(grid[taskindex])

ax.set_title(titles[taskindex],fontsize=7)
ax.tick_params(axis='both', which='major', labelsize=7)  # Set font size to 12
index=0
for model in average_scores:
  average_scores[model]=sliding_window_average(average_scores[model],5)
  minsteps=np.min([len(average_scores[model]),len(steps[model])])
  ax.plot(steps[model][:minsteps],average_scores[model][:minsteps],c=ct[index],label=model.split("/")[1],lw=1)
  index+=1

plt.tight_layout()


plt.legend(loc='lower center',bbox_to_anchor=(-1,-1.5), fancybox=True, shadow=True, ncol=2,prop={'size': 6})
#plt.show()
plt.savefig("clutering_ARI.png")
