# I used ChatGPT for this code and the online resources of dyngen

library(tidyverse)
library(dyngen)

set.seed(1)

backbone <- backbone_bifurcating()

# the simulation is being sped up because rendering all vignettes with one core
# for pkgdown can otherwise take a very long time
model <- initialise_model(
    backbone = backbone,
    num_cells = 1000,
    num_tfs = nrow(backbone$module_info),
    num_targets = 50,
    num_hks = 50,
    verbose = FALSE,
    download_cache_dir = tools::R_user_dir("dyngen", "data"),
    simulation_params = simulation_default(
        total_time = 1000,
        census_interval = 2, 
        ssa_algorithm = ssa_etl(tau = 300/3600),
        experiment_params = simulation_type_wild_type(num_simulations = 10),
        compute_cellwise_grn=FALSE
    )
)

model <- model  %>%
    generate_tf_network() %>%
    generate_feature_network() %>% 
    generate_kinetics() %>%
    generate_gold_standard() %>%
    generate_cells() %>%
    generate_experiment() 



# Writing with anndata did not work for some reason. So try to export it as csv
write.csv(model$feature_info, file = "../data/sd_feature_info.csv", row.names = TRUE)
write.csv(as.matrix(model$backbone$expression_patterns), file = "../data/sd_expression_patterns.csv", row.names = TRUE)
write.csv(as.matrix(model$feature_network), file = "../data/sd_feature_network.csv", row.names = TRUE)
write.csv(as.matrix(model$experiment$counts_mrna), file = "../data/sd_counts_mrna.csv", row.names = TRUE)
write.csv(as.matrix(model$experiment$cell_info), file = "../data/sd_cell_info.csv", row.names = TRUE)




