import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
#matplotlib.use('Agg')

modellist = [
    "/data/duolin/splm_gvp/results/config_comparewithgit/config_1/2024-03-07__23-36-23/",
    "/data/duolin/splm_gvp/results/config_comparewithgit/config_nomixedprecision/2024-03-08__13-05-43/",
    "/data/duolin/splm_gvpgit/splm/results/config_1/2024-03-08__14-30-37/", 
    "/data/duolin/splm_gvpgit/splm/results/config_nomixedprecision/2024-03-08__14-33-55/",
    "/data/duolin/splm_gvpgit/splm/results/config_nomixedprecision/2024-03-08__16-20-55/", #same data.py and add simclr.train()

]


modellist = [
    #"/data/duolin/splm_gvp/results/config_comparewithgit/config_1/2024-03-07__23-36-23/",
    "/data/duolin/splm_gvp/results/config_comparewithgit/config_nomixedprecision/2024-03-11__11-47-37/",
    #"/data/duolin/splm_gvpgit/splm/results/config_1/2024-03-08__14-30-37/", 
    "/data/duolin/splm_gvpgit/splm/results/config_nomixedprecision/2024-03-11__11-50-11/", #same data.py and add simclr.train()
    "/data/duolin/splm_gvpgit/splm/results/config_1_bf16/2024-03-11__14-39-18/",
    "/data/duolin/splm_gvpgit/splm/results/config_2_fp16/2024-03-11__14-43-48/"
    
]

cathwronglist=["results/config_12adapterH_2000random/2024-02-28__23-36-08",
"results/config_16adapterH_plddtweight/2024-02-27__09-08-20",
"results/config_16adapterH_plddtweight/2024-02-28__19-52-52",
"results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2_run1/2024-02-26__11-30-44/",
"results/config_16adapterH_100random_500cyclcstep/2024-02-27__13-53-30/",
"results/config_16fixesmtable_sgdlr001mixlenpad_esmpool2"

]
#"""
kinase_score={}
deaminase_score={}
cath_score1={}
cath_score2={}
cath_score3={}

gvp_cath_score1={}
gvp_cath_score2={}
gvp_cath_score3={}


kinase_score_2d={}
deaminase_score_2d={}
cath_score1_2d={}
cath_score2_2d={}
cath_score3_2d={}
gvp_cath_score1_2d={}
gvp_cath_score2_2d={}
gvp_cath_score3_2d={}

steps={}

for model in modellist:
    kinase_score[model]=[]
    deaminase_score[model]=[]
    cath_score1[model]=[]
    cath_score2[model]=[]
    cath_score3[model]=[]
    
    gvp_cath_score1[model]=[]
    gvp_cath_score2[model]=[]
    gvp_cath_score3[model]=[]
    
    kinase_score_2d[model]=[]
    deaminase_score_2d[model]=[]
    cath_score1_2d[model]=[]
    cath_score2_2d[model]=[]
    cath_score3_2d[model]=[]
    
    gvp_cath_score1_2d[model]=[]
    gvp_cath_score2_2d[model]=[]
    gvp_cath_score3_2d[model]=[]
    
    steps[model]=[]
    file = model+"/"+"evaluate_clustering.txt"
    if not os.path.exists(file):
          file = model+"/"+"logs.txt"
    
    input = open(file)
    for line in input:
        if "step" not in line or "step:1 " in line:
             continue
        
        if "kinase_score:" in line:
            kinase=line.split("kinase_score:")[1].strip().split("(")[0]
            kinase_2d = line.split("kinase_score:")[1].strip().split("(")[1].split(")")[0]
            kinase_score[model].append(float(kinase))
            kinase_score_2d[model].append(float(kinase_2d))
            step=line.split("step:")[1].split()[0]
            if "adapterH_16esm_tuneall_gitcosine100lr001_pool2_run2" in model:
                 step=int(step)+4000
            
            steps[model].append(int(step))
        
        if "deaminase_score" in line:
            deaminase = line.split("deaminase_score:")[1].strip().split("(")[0]
            deaminase_2d = line.split("deaminase_score:")[1].strip().split("(")[1].split(")")[0]
            deaminase_score[model].append(float(deaminase))
            deaminase_score_2d[model].append(float(deaminase_2d))
        
        if "digit_num_1:" in line and "gvp" not in line:
            digit_num_1 = line.split("digit_num_1:")[1].strip().split("(")[0]
            digit_num_2 = line.split("digit_num_2:")[1].strip().split("(")[0]
            digit_num_3 = line.split("digit_num_3:")[1].strip().split("(")[0]
            digit_num_1_2d = line.split("digit_num_1:")[1].strip().split("(")[1].split(")")[0]
            digit_num_2_2d = line.split("digit_num_2:")[1].strip().split("(")[1].split(")")[0]
            digit_num_3_2d = line.split("digit_num_3:")[1].strip().split("(")[1].split(")")[0]
            
            cath_score1[model].append(float(digit_num_1))
            cath_score2[model].append(float(digit_num_2))
            cath_score3[model].append(float(digit_num_3))
            cath_score1_2d[model].append(float(digit_num_1_2d))
            cath_score2_2d[model].append(float(digit_num_2_2d))
            cath_score3_2d[model].append(float(digit_num_3_2d))
        
        if "gvp_digit_num_1:" in line:
            digit_num_1 = line.split("gvp_digit_num_1:")[1].strip().split("(")[0]
            digit_num_2 = line.split("gvp_digit_num_2:")[1].strip().split("(")[0]
            digit_num_3 = line.split("gvp_digit_num_3:")[1].strip().split("(")[0]
            digit_num_1_2d = line.split("gvp_digit_num_1:")[1].strip().split("(")[1].split(")")[0]
            digit_num_2_2d = line.split("gvp_digit_num_2:")[1].strip().split("(")[1].split(")")[0]
            digit_num_3_2d = line.split("gvp_digit_num_3:")[1].strip().split("(")[1].split(")")[0]
            
            gvp_cath_score1[model].append(float(digit_num_1))
            gvp_cath_score2[model].append(float(digit_num_2))
            gvp_cath_score3[model].append(float(digit_num_3))
            gvp_cath_score1_2d[model].append(float(digit_num_1_2d))
            gvp_cath_score2_2d[model].append(float(digit_num_2_2d))
            gvp_cath_score3_2d[model].append(float(digit_num_3_2d))



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
titles=["cath_digit_1","cath_digit_2","cath_digit_3","deaminase","kinase","gvp_cath_digit_1","gvp_cath_digit_2","gvp_cath_digit_3"]

grid = plt.GridSpec(7, 3, wspace=0.1, hspace=0.4)
fig = plt.figure(figsize=(10, 14))
plotby2ds=[False,True]
for plotby2d in plotby2ds:
    if plotby2d:
       #drawscores=[cath_score1_2d,cath_score2_2d,deaminase_score_2d,kinase_score_2d]
       #ESM2scores=[2.50,1.52,70.85,16.81]
       drawscores=[cath_score1_2d,cath_score2_2d,cath_score3_2d,deaminase_score_2d,kinase_score_2d]
       ESM2scores=[6.66,6.51,3.72,70.85,16.81]
       swin_bestscores=[482.3181011612754,190.64295557670664,90.00002220514895
                       ,70.85,16.81]
       
       gvp_bestscores=[1602.8780,444.9441,159.5350,70.85,16.81]
       
    else:
       #drawscores=[cath_score1,cath_score2,deaminase_score,kinase_score]
       #ESM2scores=[3.12,1.68,14.85,10.02]
       drawscores=[cath_score1,cath_score2,cath_score3,deaminase_score,kinase_score]
       ESM2scores=[13.42,7.41,4.31,14.85,10.02]
       swin_bestscores=[143.1679895065888,63.538589232929986,34.026830287329105,
                        14.85,10.02]
       
       gvp_bestscores=[480.7318,135.5101,49.0624,
                       14.85,10.02]
    
    
    
    for drawscore in drawscores:
      index=0
      taskindex=drawscores.index(drawscore)
      #print(taskindex)
      #print(drawscore)
      if plotby2ds.index(plotby2d)==0:
         ax = fig.add_subplot(grid[taskindex])
      else:
         ax = fig.add_subplot(grid[taskindex+5*plotby2ds.index(plotby2d)+1])
      
      ax.set_title(titles[taskindex],fontsize=7)
      ax.tick_params(axis='both', which='major', labelsize=7)  # Set font size to 12
      for model in drawscore:
        drawscore[model]=sliding_window_average(drawscore[model],10)
        minsteps=np.min([len(drawscore[model]),len(steps[model])])
        ax.plot(steps[model][:minsteps],drawscore[model][:minsteps],c=ct[index],label=model.split("/")[1],lw=1)
        index+=1
      
      plt.axhline(y=swin_bestscores[drawscores.index(drawscore)], color="darkgreen", linestyle='--', label='Swin_contactmap') 
      plt.axhline(y=gvp_bestscores[drawscores.index(drawscore)], color="green", linestyle='--', label='gvp_random_initial') 
      plt.axhline(y=ESM2scores[drawscores.index(drawscore)], color="red", linestyle='--', label='ESM2 650') 
      #plt.ylim(0,6)

plt.legend(loc='lower center',bbox_to_anchor=(0.3,-1.5), fancybox=True, shadow=True, ncol=2,prop={'size': 6})

#"""
drawscores=[gvp_cath_score1,gvp_cath_score2,gvp_cath_score3]
ESM2scores=[13.42,7.41,4.31,14.85,10.02]
gvp_bestscores=[480.7318,135.5101,49.0624]

taskindex=12
for drawscore in drawscores:
      index=0
      ax = fig.add_subplot(grid[taskindex])
      ax.set_title(titles[taskindex-7],fontsize=7)
      ax.tick_params(axis='both', which='major', labelsize=7)  # Set font size to 12
      for model in drawscore:
        if model in cathwronglist:
              index+=1
              continue
        
        drawscore[model]=sliding_window_average(drawscore[model],10)
        minsteps=np.min([len(drawscore[model]),len(steps[model])])
        ax.plot(steps[model][:minsteps],drawscore[model][:minsteps],c=ct[index],label=model)#.split("/")[1],lw=1)
        index+=1
      
      plt.axhline(y=gvp_bestscores[drawscores.index(drawscore)], color="green", linestyle='--', label='gvp_random_initial') 
      plt.axhline(y=ESM2scores[drawscores.index(drawscore)], color="red", linestyle='--', label='ESM2 650') 
      #plt.ylim(0,6)
      taskindex+=1

#"""
plt.legend(loc='lower center',bbox_to_anchor=(0.3,-1.5), fancybox=True, shadow=True, ncol=2,prop={'size': 6})
plt.tight_layout()
#plt.show()
plt.savefig("clutering_results.png")
