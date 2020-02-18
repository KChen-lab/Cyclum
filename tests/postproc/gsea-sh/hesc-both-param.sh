for ((i=0;i<8;i++))
do
~/Documents/gsea/GSEA_Linux_4.0.2/gsea-cli.sh GSEA -res /home/shaoheng/Documents/data/hESC-droplet/GSE125416/GSE125416-regular.txt \
-cls /home/shaoheng/Documents/data/hESC-droplet/GSE125416/GSE125416-regular-pseudotime-param.cls#phase${i} \
-gmx /home/shaoheng/Documents/data/hESC-droplet/GSE125416/msigdb-selected/GO-filtered.gmt \
-collapse No_Collapse \
-mode Max_probe \
-norm meandiv \
-nperm 100 \
-permute phenotype \
-rnd_type no_balance \
-scoring_scheme weighted \
-rpt_label GSE125416_phase${i} \
-metric Pearson \
-sort real \
-order descending \
-create_gcts false \
-create_svgs false \
-include_only_symbols true \
-make_sets true \
-median false \
-num 100 \
-plot_top_x 20 \
-rnd_seed 149 -save_rnd_lists false -set_max 0 -set_min 0 -zip_report false \
-out /home/shaoheng/gsea_home/output/nov01
done