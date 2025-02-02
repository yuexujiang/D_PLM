import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
# import pylab
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
# import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
# from collections import OrderedDict
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
from utils.utils import load_checkpoints, load_configs,get_logging
from data.data import custom_collate, ProteinGraphDataset
import tqdm
import gvp.models
from accelerate import Accelerator, DistributedDataParallelKwargs

# import math
# from gvp import GVP, GVPConvLayer, LayerNorm, tuple_index
# from model import MoBYMLP
from torch_geometric.nn import radius, global_mean_pool, global_max_pool
from model import prepare_models

def scatter_labeled_z(z_batch, colors, filename="test_plot"):
    fig = plt.gcf()
    plt.switch_backend('Agg')
    fig.set_size_inches(3.5, 3.5)
    plt.clf()
    for n in range(z_batch.shape[0]):
        result = plt.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[n], s=50, marker="o", edgecolors='none')
    
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.savefig(filename)
    # pylab.show()


def evaluate_with_cath_more_struct(out_figure_path, steps, accelerator, batch_size, 
                                   model, cathpath, configs
                                   #seq_mode="embedding",use_rotary_embeddings=False,
                                   ):
    Path(out_figure_path).mkdir(parents=True, exist_ok=True)
    dataset = ProteinGraphDataset(cathpath,max_length=configs.model.esm_encoder.max_length, 
                                        seq_mode = configs.model.struct_encoder.use_seq.seq_embed_mode,
                                        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
                                        rotary_mode=configs.model.struct_encoder.rotary_mode,
                                        use_foldseek = configs.model.struct_encoder.use_foldseek,
                                        use_foldseek_vector = configs.model.struct_encoder.use_foldseek_vector,
                                        top_k = configs.model.struct_encoder.top_k,
                                        num_rbf = configs.model.struct_encoder.num_rbf,
                                        num_positional_embeddings = configs.model.struct_encoder.num_positional_embeddings)
    
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=custom_collate)
    # val_loader = accelerator.prepare(val_loader)
    seq_embeddings = []
    labels = []
    # existingmodel.eval()
    for batch in val_loader:
        with torch.inference_mode():
            graph = batch["graph"].to(accelerator.device)
            _,_,features_struct, _ = model(graph=graph, mode='structure',return_embedding=True)
            seq_embeddings.extend(features_struct.cpu().detach().numpy())
            labels.extend([id.split("_")[1] for id in batch['pid']])
    
    seq_embeddings = np.asarray(seq_embeddings)
    # print("shape of seq_embeddings=" + str(seq_embeddings.shape))
    mdel = TSNE(n_components=2, random_state=0, init='random',method='exact')
    # print("Projecting to 2D by TSNE\n")
    z_tsne_seq = mdel.fit_transform(seq_embeddings)
    scores = []
    tol_class_seq={"1":0,"2":1,"3":2}
    tol_archi_seq={"3.30":3,"3.40":4,"1.10":0,"3.10":2,"2.60":1}
    tol_fold_seq={"1.10.10":0,"3.30.70":3,"2.60.40":2,"2.60.120":1,"3.40.50":4}
    for digit_num in [1, 2, 3]:  # first number of digits
        color = []
        keys = {}
        colorid = []
        colorindex = 0
        if digit_num == 1:
            ct = ["blue", "red", "black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink",
                  "cyan", "peru", "darkgray", "slategray", "gold"]
        else:
            ct = ["black", "yellow", "orange", "green", "olive", "gray", "magenta", "hotpink", "pink", "cyan", "peru",
                  "darkgray", "slategray", "gold"]
        
        select_index = []
        color_dict = {}
        index=0
        for label in labels:
            key = ".".join([x for x in label.split(".")[0:digit_num]])
            if digit_num==1:
               keys = tol_class_seq
            if digit_num==2:
              keys = tol_archi_seq
              if key not in tol_archi_seq:
                index+=1
                continue
            if digit_num==3:
              keys = tol_fold_seq
              if key not in tol_fold_seq:
                index+=1
                continue
            
            color.append(ct[(keys[key]) % len(ct)])
            colorid.append(keys[key])
            select_index.append(index)
            color_dict[keys[key]] = ct[keys[key]]
            index+=1
        
        scores.append(calinski_harabasz_score(seq_embeddings[select_index], color))
        scores.append(calinski_harabasz_score(z_tsne_seq[select_index], color))
        scatter_labeled_z(z_tsne_seq[select_index], color,
                          filename=os.path.join(out_figure_path, f"step_{steps}_CATHgvp_{digit_num}.png"))
        # add kmeans
        kmeans = KMeans(n_clusters=len(color_dict), random_state=42)
        predicted_labels = kmeans.fit_predict(z_tsne_seq[select_index])
        # predicted_labels = kmeans.fit_predict(seq_embeddings[select_index])
        predicted_colors = [color_dict[label] for label in predicted_labels]
        scatter_labeled_z(z_tsne_seq[select_index], predicted_colors,
                          filename=os.path.join(out_figure_path, f"step_{steps}_CATHgvp_{digit_num}_kmpred.png"))
        ari = adjusted_rand_score(colorid, predicted_labels)
        scores.append(ari)
    
    return scores  # [digit_num1_full,digit_num_2d,digit_num2_full,digit_num2_2d]



def test_evaluate_allcases():
    import yaml
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(0))
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    
    config_path = "/data/duolin/splm_gvpgit/splm/configs_local/config_noseq.yaml"
    with open(config_path) as file:
        config_file = yaml.full_load(file)
        configs = load_configs(config_file, args=None)
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.gradient_accumulation,
        dispatch_batches=False,
        kwargs_handlers=[ddp_kwargs]
    )
    
    output_img_path = "gvp_original_ARI_noprojector_5gvplayer"
    Path(output_img_path).mkdir(parents=True, exist_ok=True)
    checkpoint_path = None
    cathpath = "/data/duolin/CATH/CATH_4_3_0_non-rep_gvp/"
    
    logging = get_logging(output_img_path)
    model = prepare_models(logging, configs, accelerator)
    model = accelerator.prepare(model)
    n_iter = 32
    model.eval()
    scores_cath = evaluate_with_cath_more_struct(output_img_path,
                                                 steps=n_iter,
                                                 accelerator=accelerator,
                                                 batch_size=5,
                                                 cathpath=cathpath,
                                                 model=model,
                                                 configs=configs
                                                 #seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
                                                 #use_rotary_embeddings = configs.model.struct_encoder.use_rotary_embeddings
                                                 )
    if accelerator.is_main_process:
       print(
          f"step:{n_iter}\tgvp_digit_num_1:{scores_cath[0]:.4f}({scores_cath[1]:.4f})\tgvp_digit_num_2:{scores_cath[3]:.4f}({scores_cath[4]:.4f})\tgvp_digit_num_3:{scores_cath[6]:.4f}({scores_cath[7]:.4f})\n")
       print(
          f"step:{n_iter}\tgvp_digit_num_1_ARI:{scores_cath[2]}\tgvp_digit_num_2_ARI:{scores_cath[5]}\tgvp_digit_num_3_ARI:{scores_cath[8]}\n")


# test_evaluate_allcases()

"""
device = torch.device('cpu')
checkpoint_path=None with random projection layer
step:1  gvp_digit_num_1:254.5661(1156.9925)     gvp_digit_num_2:108.3430(398.2088)      gvp_digit_num_3:1577219142891.7288(157981159901721.9375)
step:1  gvp_digit_num_1_ARI:0.4882823118773334  gvp_digit_num_2_ARI:0.3676668198164138  gvp_digit_num_3_ARI:1.0

gvp_num_layers = 3
checkpoint_path=None without projectors
step:1  gvp_digit_num_1:352.3880(1235.3478)     gvp_digit_num_2:126.9079(422.1699)      gvp_digit_num_3:2440682022379.8457(599999936853985.0000)
step:1  gvp_digit_num_1_ARI:0.4792891802818779  gvp_digit_num_2_ARI:0.4232081868453065  gvp_digit_num_3_ARI:1.0

gvp_num_layers = 3 node_h_dim=(100, 16) n_iter=1?
step:1  gvp_digit_num_1:496.2665(1733.7281)     gvp_digit_num_2:168.3572(590.9794)      gvp_digit_num_3:2265831569900.8599(520780205339754.0000)
step:1  gvp_digit_num_1_ARI:0.4548252959809612  gvp_digit_num_2_ARI:0.367745901071553   gvp_digit_num_3_ARI:1.0

#gvp_num_layers=3  node_h_dim=(256, 32) n_iter=3
step:3    gvp_digit_num_1:362.2092(1291.9882)     gvp_digit_num_2:126.9063(416.9572)      gvp_digit_num_3:2706432814817.3550(256563538430846.5312)
step:3  gvp_digit_num_1_ARI:0.4157902176930986  gvp_digit_num_2_ARI:0.3443341093868343  gvp_digit_num_3_ARI:1.0


gvp_num_layers=3  node_h_dim=(100, 64) n_iter=5
step:5  gvp_digit_num_1:377.3179(1500.7029)     gvp_digit_num_2:138.6826(556.8882)      gvp_digit_num_3:1823838021054.7549(181622068722801.0625)
step:5  gvp_digit_num_1_ARI:0.45206483994636687 gvp_digit_num_2_ARI:0.4103133050006914  gvp_digit_num_3_ARI:1.0

gvp_num_layers=5  node_h_dim=(100, 32)     n_iter=32
step:32 gvp_digit_num_1:516.4435(1620.6733)     gvp_digit_num_2:219.6288(670.1878)      gvp_digit_num_3:2834233687903.3521(386082004837642.7500)
step:32 gvp_digit_num_1_ARI:0.4870877055576386  gvp_digit_num_2_ARI:0.43814098522821626 gvp_digit_num_3_ARI:1.0


gvp_num_layers = 10 node_h_dim=(100, 16)
step:1  gvp_digit_num_1:504.9836(1797.1726)     gvp_digit_num_2:169.0837(608.2719)      gvp_digit_num_3:1941095932379.4294(202494386284816.5312)
step:1  gvp_digit_num_1_ARI:0.42514080796916226 gvp_digit_num_2_ARI:0.3739549248898803  gvp_digit_num_3_ARI:1.0

gvp_num_layers = 10 node_h_dim=(256, 32)
step:3  gvp_digit_num_1:365.2327(1566.4202)     gvp_digit_num_2:123.4423(497.4646)      gvp_digit_num_3:1541299770846.1829(743434104869368.2500)
step:3  gvp_digit_num_1_ARI:0.5270390580245173  gvp_digit_num_2_ARI:0.37982240992211846 gvp_digit_num_3_ARI:1.0


###used as model
gvp_num_layers=3  node_h_dim=(100, 32)     n_iter=4
step:4  gvp_digit_num_1:701.8264(1844.1294)     gvp_digit_num_2:217.6042(574.8041)      gvp_digit_num_3:3003872345520.8193(433136176819947.8125)
step:4  gvp_digit_num_1_ARI:0.433659742202448   gvp_digit_num_2_ARI:0.3883964096871641  gvp_digit_num_3_ARI:1.0


#use_rotary_embeddings=True  2,100,250
step:32 gvp_digit_num_1:2257.8193(1119.7606)    gvp_digit_num_2:695.6338(384.0565)      gvp_digit_num_3:254.7545(171.0784)
step:32 gvp_digit_num_1_ARI:0.491719036218213   gvp_digit_num_2_ARI:0.30611736182801397 gvp_digit_num_3_ARI:0.3562327424818268

#use_rotary_embeddings=False trainable params:  2,100,726
step 32 gvp_digit_num_1:2306.5890(1050.5966)    gvp_digit_num_2:713.4341(363.5167)  gvp_digit_num_3:263.5430(174.4342)
step:32 gvp_digit_num_1_ARI:0.49494180678187183 gvp_digit_num_2_ARI:0.29824880528099873     gvp_digit_num_3_ARI:0.3404483211530337

#use_rotary_embeddings=True  2,100,250 use topk 100 and 1024 length
step:32 gvp_digit_num_1:2375.0031(1042.2157)    gvp_digit_num_2:729.0799(356.9017) gvp_digit_num_3:273.1033(167.5881)
step:32 gvp_digit_num_1_ARI:0.4992020941726364  gvp_digit_num_2_ARI:0.29439032999065434     gvp_digit_num_3_ARI:0.33008316451172154


step:32   gvp_digit_num_1:2025.5276(1162.2301)    gvp_digit_num_2:617.8739(381.3600)  gvp_digit_num_3:225.2348(167.4832)

step:32 gvp_digit_num_1_ARI:0.5020427874819323  gvp_digit_num_2_ARI:0.27875063204149786     gvp_digit_num_3_ARI:0.31948468286382486

#use_foldseek 2,101,270 only add scalar foldseek
step:32 gvp_digit_num_1:1474.1635(1945.4202)    gvp_digit_num_2:408.4191(563.7495)      gvp_digit_num_3:173.9592(226.3431)
step:32 gvp_digit_num_1_ARI:0.4962134775703041  gvp_digit_num_2_ARI:0.3056000038976739  gvp_digit_num_3_ARI:0.33028489981321296
#use foldseek 2,101,462 add scalar and vector foldseek
step:32 gvp_digit_num_1:1913.6563(1197.1654)    gvp_digit_num_2:554.9816(385.2621) gvp_digit_num_3:216.7075(172.4363)
step:32 gvp_digit_num_1_ARI:0.44455163929910413 gvp_digit_num_2_ARI:0.28703578351546316    gvp_digit_num_3_ARI:0.35056606332377643
#use only vector foldseek 2,100,442 
step:32 gvp_digit_num_1:2126.7617(1098.6738)    gvp_digit_num_2:649.4627(369.2571)      gvp_digit_num_3:235.1638(156.9453)
step:32 gvp_digit_num_1_ARI:0.47927438740377615 gvp_digit_num_2_ARI:0.27444272731466324 gvp_digit_num_3_ARI:0.3319957337201843
#run again
#only foldseek vector = True, rotary_mode==1 2,100,442
step:32 gvp_digit_num_1:2012.7870(1302.7356)    gvp_digit_num_2:609.5098(418.6209)        gvp_digit_num_3:220.9712(174.8269)
step:32 gvp_digit_num_1_ARI:0.4850577342614155  gvp_digit_num_2_ARI:0.2916211784284493    gvp_digit_num_3_ARI:0.3612732168382461
#only foldseek vector = True, rotary_mode==2 2,100,442
step:32 gvp_digit_num_1:2259.3853(1031.3984)    gvp_digit_num_2:682.3813(347.8175)      gvp_digit_num_3:257.6894(165.3899)
step:32 gvp_digit_num_1_ARI:0.4792999693111433  gvp_digit_num_2_ARI:0.2979689310131295  gvp_digit_num_3_ARI:0.33635273776388747
#only foldseek vector = True, rotary_mode==3 2,100,646
step:32 gvp_digit_num_1:2016.7269(1158.4834)    gvp_digit_num_2:607.8523(376.6617)      gvp_digit_num_3:223.0728(160.4129)
step:32 gvp_digit_num_1_ARI:0.49218510692855993 gvp_digit_num_2_ARI:0.2882746705231635  gvp_digit_num_3_ARI:0.32909106300903634

"""
