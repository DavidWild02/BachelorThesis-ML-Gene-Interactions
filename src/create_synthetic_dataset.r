# I used ChatGPT for this code and the online resources of dyngen

library(tidyverse)
library(dyngen)

set.seed(1)


backbone = backbone_bifurcating()

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
        compute_cellwise_grn=TRUE
    )
)

model <- model %>%
    generate_tf_network() %>%
    generate_feature_network() %>% 
    generate_kinetics() %>%
    generate_gold_standard() %>%
    generate_cells() %>%
    generate_experiment()

write.csv(model$feature_info, file = "../data/sd_feature_info.csv", row.names = TRUE)
write.csv(as.matrix(model$gold_standard$mod_changes), file = "../data/sd_expression_patterns.csv", row.names = TRUE)
write.csv(as.matrix(model$gold_standard$network), file = "../data/sd_feature_network.csv", row.names = TRUE)
write.csv(as.matrix(model$simulations$counts), file = "../data/sd_counts.csv", row.names = TRUE)
write.csv(as.matrix(model$experiment$cell_info), file = "../data/sd_cell_info.csv", row.names = TRUE)

# ad <- as_anndata(model)
# ad$write_h5ad("./data/synthetic_dataset.h5ad")

# expression_patterns = backbone$expression_patterns

# # Create cell state adjacency matrix
# cell_states <- unique(c(expression_patterns$from, expression_patterns$to))
# state_adjacency <- matrix(0, nrow=length(cell_states), ncol=length(cell_states),
#                           dimnames = list(cell_states, cell_states))

# for (i in seq_len(nrow(expression_patterns))) {
#     from <- expression_patterns$from[i]
#     to <- expression_patterns$to[i]
#     state_adjacency[from, to] <- 1
# }
# write.csv(state_adjacency, file = "state_adjacency.csv", row.names = TRUE)


# # Extract sub networks from the GRN for each cell state transition
# extract_modules_from_progression_pattern = function(progression_pattern) {
#     stringr::str_extract_all(module_progression, "[+-][^,\\|]+")[[1]]
# }

# transition_modules <- expression_patterns %>%
#     rowwise() %>%
#     mutate(module_list = extract_modules_from_progression_pattern(module_progression)) %>%
#     select(from, to, module_list)

# for (i in seq_len(nrow(transition_modules))) {
#     from <- transition_modules$from[i]
#     to <- transition_modules$to[i]
#     module_list <- transition_modules$module_list[i]

    
# }


