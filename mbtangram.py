import lightning
import scanpy as sc
import tangram as tg
import gc

def tangram_batch_allsc(adata_st_gt, adata_sc, savepath = "./data_breastbatch/", celltype = "graph_cluster_anno",  
                  epoch = 500, filename = 'human_breast', batchsize = 5000, 
                        set_seed = 0, density_prior = 'uniform', train_gene = None, spatial_label = 'scClassify'):
    lightning.seed_everything(set_seed)
    markers = train_gene
    adata_st_gt.obs[celltype] = list(adata_st_gt.obs[spatial_label])
    for item in range(len(adata_st_gt)//batchsize + 1):
        end = item*batchsize + batchsize
        if end >= len(adata_st_gt):
            end = len(adata_st_gt)

        adata_st_imp = adata_st_gt[item*batchsize:end,:]
        
        tg.pp_adatas(adata_sc, adata_st_imp, genes=markers, gene_to_lowercase = False)

        ad_map = tg.map_cells_to_space(adata_sc, adata_st_imp,
            mode="cells",
            density_prior=density_prior,
            num_epochs=epoch,
            device="cuda:0", 
            correlation = False       #     device='cpu',

        )

        gc.collect()
        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=adata_sc, gene_to_lowercase = False)
        gc.collect()
        print(epoch)
        ad_ge.write_h5ad(savepath + f"{filename}_data_allgenes_{epoch}_{item}_seed{set_seed}.h5ad")
    return True

adata_sc = sc.read("scrnaseq_data_path")
adata_st = sc.read("spatial_data_path")
info_gene = list(set(adata_sc.var_names).intersection(adata_st.var_names))
tangram_batch_allsc(adata_st, adata_sc, epoch=500, savepath='./data_seqfish/', 
                    celltype='scClassify', filename = 'seqfish', spatial_label = 'scClassify',
                   set_seed = 0,train_gene = info_gene)

#create dataset
list_adata = []
num_files = 12 # this number is depended by the batch_size and the length of dataset.
for i in range(0,num_files):
    adata_st = sc.read_h5ad(f"seqfish_data_allgenes_500_{i}_seed0.h5ad")
    list_adata.append(adata_st)  

adata_out = sc.concat(list_adata)
adata_out.write_h5ad("tangram_seqfish_allgenes.h5ad")