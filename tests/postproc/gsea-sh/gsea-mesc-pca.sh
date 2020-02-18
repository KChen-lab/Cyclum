for ((i=1;i<6;i++))
do
~/Documents/gsea/GSEA_Linux_4.0.2/gsea-cli.sh GSEA -res /home/shaoheng/Documents/SCAE/results/mESC/mesc_scaled_matrix.txt \
-cls /home/shaoheng/Documents/SCAE/results/mESC/mesc_pc.cls#pc${i} \
-gmx /home/shaoheng/Documents/SCAE/results/mESC/mesc_gene_sets.gmt \
-collapse No_Collapse \
-mode Max_probe \
-norm meandiv \
-nperm 100 \
-permute phenotype \
-rnd_type no_balance \
-scoring_scheme weighted \
-rpt_label hESC_phase${i} \
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
-rnd_seed 2 -save_rnd_lists false -set_max 0 -set_min 0 -zip_report false \
-out /home/shaoheng/gsea_home/output/Dec23 &
done
