#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#
from jnius import autoclass
# Import directly from private modules to avoid circular imports which are a problem in Python 3.6.
from pypgx.api._partition import PgxPartition
from pypgx.api._pgx_collection import VertexSet, VertexSequence, EdgeSequence, EdgeSet
from pypgx.api._pgx_entity import PgxVertex
from pypgx.api._pgx_graph import PgxGraph, BipartiteGraph
from pypgx.api._pgx_map import PgxMap
from pypgx.api._property import VertexProperty, EdgeProperty
from pypgx.api.filters import VertexFilter, EdgeFilter
from pypgx.api.mllib import CorruptionFunction, PermutationCorruption

pg2vec_builder = {
    "in_arguments": {
        "graphlet_id_property_name": {"type": [str], "default": None},
        "vertex_property_names": {"type": [list], "default": None},
        "min_word_frequency": {"type": [int], "default": 1},
        "batch_size": {"type": [int], "default": 128},
        "num_epochs": {"type": [int], "default": 5},
        "layer_size": {"type": [int], "default": 200},
        "learning_rate": {"type": [float], "default": 0.04},
        "min_learning_rate": {"type": [float], "default": 0.0001},
        "window_size": {"type": [int], "default": 4},
        "walk_length": {"type": [int], "default": 8},
        "walks_per_vertex": {"type": [int], "default": 5},
        "use_graphlet_size": {"type": [bool], "default": True},
        "validation_fraction": {"type": [float], "default": 0.05},
    },
    "out_arguments": {},
    "return_value": {},
}

deepwalk_builder = {
    "in_arguments": {
        "min_word_frequency": {"type": [int], "default": 1},
        "batch_size": {"type": [int], "default": 128},
        "num_epochs": {"type": [int], "default": 2},
        "layer_size": {"type": [int], "default": 200},
        "learning_rate": {"type": [float], "default": 0.025},
        "min_learning_rate": {"type": [float], "default": 0.0001},
        "window_size": {"type": [int], "default": 5},
        "walk_length": {"type": [int], "default": 5},
        "walks_per_vertex": {"type": [int], "default": 4},
        "sample_rate": {"type": [float], "default": 0.00001},
        "validation_fraction": {"type": [float], "default": 0.05},
    },
    "out_arguments": {},
    "return_value": {},
}

supervised_graphwise_builder = {
    "in_arguments": {
        "batch_size": {"type": [int], "default": 128},
        "num_epochs": {"type": [int], "default": 3},
        "learning_rate": {"type": [float], "default": 0.01},
        "layer_size": {"type": [int], "default": 128},
        "vertex_input_property_names": {"type": [list], "default": []},
        "edge_input_property_names": {"type": [list], "default": []},
        "vertex_target_property_name": {"type": [str], "default": None},
        "pred_layer_config": {"type": [list, type(None)], "default": None},
    },
    "out_arguments": {},
    "return_value": {},
}

unsupervised_graphwise_builder = {
    "in_arguments": {
        "batch_size": {"type": [int], "default": 128},
        "num_epochs": {"type": [int], "default": 3},
        "learning_rate": {"type": [float], "default": 0.001},
        "layer_size": {"type": [int], "default": 128},
        "vertex_input_property_names": {"type": [list], "default": []},
    },
    "out_arguments": {},
    "return_value": {},
}
java_permutation_corruption = autoclass(
    "oracle.pgx.config.mllib.corruption.PermutationCorruption")()
graphwise_dgi_layer_config = {
    "in_arguments": {
        "corruption_function": {"type": [CorruptionFunction],
                                "default": PermutationCorruption(java_permutation_corruption)},
        "readout_function": {"type": [str], "default": "MEAN"},
        "discriminator": {"type": [str], "default": "BILINEAR"},
    },
    "out_arguments": {},
    "return_value": {},
}

graphwise_conv_layer_config = {
    "in_arguments": {
        "weight_init_scheme": {"type": [str], "default": "XAVIER_UNIFORM"},
        "activation_fn": {"type": [str], "default": "ReLU"},
    },
    "out_arguments": {},
    "return_value": {},
}

graphwise_pred_layer_config = {
    "in_arguments": {
        "hidden_dim": {"type": [int, type(None)], "default": None},
        "weight_init_scheme": {"type": [str], "default": "XAVIER_UNIFORM"},
        "activation_fn": {"type": [str], "default": "ReLU"},
    },
    "out_arguments": {},
    "return_value": {},
}

pagerank = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
        "norm": {"type": [bool], "default": False},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "pagerank",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

pagerank_approximate = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "approx_pagerank",
            "subtype": {VertexProperty: {"dimension": 0, "type": "double"}},
        }
    },
    "return_value": {},
}

weighted_pagerank = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
        "norm": {"type": [bool], "default": False},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "weighted_pagerank",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

personalized_pagerank = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "v": {"type": [PgxVertex, VertexSet], "default": None},
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
        "norm": {"type": [bool], "default": False},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "personalized_pagerank",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

personalized_weighted_pagerank = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "v": {"type": [PgxVertex, VertexSet], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
        "norm": {"type": [bool], "default": False},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "personalized_weighted_pagerank",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

vertex_betweenness_centrality = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "bc": {
            "type": [str, VertexProperty],
            "default": "betweenness",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

approximate_vertex_betweenness_centrality = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "seeds": {"type": [int, VertexSet], "default": None},
    },
    "out_arguments": {
        "bc": {
            "type": [str, VertexProperty],
            "default": "approx_betweenness",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

closeness_centrality = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "cc": {
            "type": [str, VertexProperty],
            "default": "closeness",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

weighted_closeness_centrality = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "cc": {
            "type": [str, VertexProperty],
            "default": "weighted_closeness",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

hits = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "auth": {
            "type": [str, VertexProperty],
            "default": "authorities",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "hubs": {
            "type": [str, VertexProperty],
            "default": "hubs",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
    },
    "return_value": {},
}

eigenvector_centrality = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "tol": {"type": [float], "default": 0.001},
        "max_iter": {"type": [int], "default": 100},
        "l2_norm": {"type": [bool], "default": False},
        "in_edges": {"type": [bool], "default": False},
    },
    "out_arguments": {
        "ec": {
            "type": [str, VertexProperty],
            "default": "eigenvector",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

out_degree_centrality = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "dc": {
            "type": [str, VertexProperty],
            "default": "out_degree",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

in_degree_centrality = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "dc": {
            "type": [str, VertexProperty],
            "default": "in_degree",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

degree_centrality = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "dc": {
            "type": [str, VertexProperty],
            "default": "degree",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

adamic_adar_counting = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "aa": {
            "type": [str, EdgeProperty],
            "default": "adamic_adar",
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

communities_label_propagation = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "label": {
            "type": [str, VertexProperty],
            "default": "label_propagation",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

communities_conductance_minimization = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "label": {
            "type": [str, VertexProperty],
            "default": "conductance_minimization",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

communities_infomap = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "rank": {
            "type": [VertexProperty],
            "default": None,
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "tau": {"type": [float], "default": 0.15},
        "tol": {"type": [float], "default": 0.0001},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "label": {
            "type": [str, VertexProperty],
            "default": "infomap",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

louvain = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "max_iter": {"type": [int], "default": 100},
        "nbr_pass": {"type": [int], "default": 1},
        "tol": {"type": [float], "default": 0.0001},
    },
    "out_arguments": {
        "community": {
            "type": [str, VertexProperty],
            "default": "community",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

conductance = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "partition": {"type": [PgxPartition], "default": None},
        "partition_idx": {"type": [int], "default": 100},
    },
    "out_arguments": {},
    "return_value": {},
}

partition_conductance = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "partition": {"type": [PgxPartition], "default": None},
    },
    "out_arguments": {},
    "return_value": {},
}

partition_modularity = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "partition": {"type": [PgxPartition], "default": None},
    },
    "out_arguments": {},
    "return_value": {},
}

scc_kosaraju = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "label": {
            "type": [str, VertexProperty],
            "default": "scc_kosaraju",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

scc_tarjan = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "label": {
            "type": [str, VertexProperty],
            "default": "scc_tarjan",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

wcc = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "label": {
            "type": [str, VertexProperty],
            "default": "wcc",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

salsa = {
    "in_arguments": {
        "bipartite_graph": {"type": [BipartiteGraph], "default": None},
        "tol": {"type": [float], "default": 0.001},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "salsa",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

personalized_salsa = {
    "in_arguments": {
        "bipartite_graph": {"type": [BipartiteGraph], "default": None},
        "v": {"type": [PgxVertex, VertexSet], "default": None},
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "rank": {
            "type": [str, VertexProperty],
            "default": "personalized_salsa",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

whom_to_follow = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "v": {"type": [PgxVertex], "default": None},
        "top_k": {"type": [int], "default": 100},
        "size_circle_of_trust": {"type": [int], "default": 500},
        "tol": {"type": [float], "default": 0.001},
        "damping": {"type": [float], "default": 0.85},
        "max_iter": {"type": [int], "default": 100},
        "salsa_tol": {"type": [float], "default": 0.001},
        "salsa_max_iter": {"type": [int], "default": 100},
    },
    "out_arguments": {
        "hubs": {"type": [str, type(None), VertexSequence], "default": None},
        "auth": {"type": [str, type(None), VertexSequence], "default": None},
    },
    "return_value": {},
}

matrix_factorization_gradient_descent = {
    "in_arguments": {
        "bipartite_graph": {"type": [BipartiteGraph], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "learning_rate": {"type": [float], "default": 0.001},
        "change_per_step": {"type": [float], "default": 1.0},
        "lbd": {"type": [float], "default": 0.15},
        "max_iter": {"type": [int], "default": 100},
        "vector_length": {"type": [int], "default": 10},
    },
    "out_arguments": {
        "features": {
            "type": [str, VertexProperty],
            "default": "features",
            "subtype": {VertexProperty: {"type": "double", "dimension": "vector_length"}},
        }
    },
    "return_value": {},
}

fattest_path = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "root": {"type": [PgxVertex], "default": None},
        "capacity": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "fattest_path_distance",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "fattest_path_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "fattest_path_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_dijkstra = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "parent": {
            "type": [str, VertexProperty],
            "default": "dijkstra_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "dijkstra_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_filtered_dijkstra = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "filter_expression": {"type": [VertexFilter, EdgeFilter], "default": None},
    },
    "out_arguments": {
        "parent": {
            "type": [str, VertexProperty],
            "default": "dijkstra_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "dijkstra_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_bidirectional_dijkstra = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "parent": {
            "type": [str, VertexProperty],
            "default": "bidirectional_dijkstra_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "bidirectional_dijkstra_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_filtered_bidirectional_dijkstra = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
        "filter_expression": {"type": [VertexFilter, EdgeFilter], "default": None},
    },
    "out_arguments": {
        "parent": {
            "type": [str, VertexProperty],
            "default": "bidirectional_dijkstra_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "bidirectional_dijkstra_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_bellman_ford = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "bellman_ford_distance",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "bellman_ford_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "bellman_ford_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_bellman_ford_reversed = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "bellman_ford_distance",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "bellman_ford_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "bellman_ford_parent_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_hop_distance = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "hop_dist_distance",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "hop_dist_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "hop_dist_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

shortest_path_hop_distance_reversed = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "hop_dist_distance",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "hop_dist_parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
        "parent_edge": {
            "type": [str, VertexProperty],
            "default": "hop_dist_edge",
            "subtype": {VertexProperty: {"type": "edge", "dimension": 0}},
        },
    },
    "return_value": {},
}

count_triangles = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "sort_vertices_by_degree": {"type": [bool], "default": None},
    },
    "out_arguments": {},
    "return_value": {},
}

k_core = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "min_core": {"type": [int], "default": 0},
        "max_core": {"type": [int], "default": 2147483647},
    },
    "out_arguments": {
        "kcore": {
            "type": [str, VertexProperty],
            "default": "kcore",
            "subtype": {VertexProperty: {"type": "long", "dimension": 0}},
        }
    },
    "return_value": {},
}

diameter = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "eccentricity": {
            "type": [str, VertexProperty],
            "default": "eccentricity",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

radius = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "eccentricity": {
            "type": [str, VertexProperty],
            "default": "eccentricity",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

periphery = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {"periphery": {"type": [str, type(None), VertexSet], "default": None}},
    "return_value": {},
}

center = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {"center": {"type": [str, type(None), VertexSet], "default": None}},
    "return_value": {},
}

local_clustering_coefficient = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "lcc": {
            "type": [str, VertexProperty],
            "default": "lcc",
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

find_cycle = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [type(None), PgxVertex], "default": None},
    },
    "out_arguments": {
        "vertex_seq": {"type": [str, type(None), VertexSequence], "default": None},
        "edge_seq": {"type": [str, type(None), EdgeSequence], "default": None},
    },
    "return_value": {},
}

reachability = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "max_hops": {"type": [int], "default": None},
        "ignore_edge_direction": {"type": [bool], "default": None},
    },
    "out_arguments": {},
    "return_value": {},
}

topological_sort = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "topo_sort": {
            "type": [str, VertexProperty],
            "default": "topo_sort",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

topological_schedule = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "vs": {"type": [VertexSet], "default": None},
    },
    "out_arguments": {
        "topo_sched": {
            "type": [str, VertexProperty],
            "default": "topo_sched",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        }
    },
    "return_value": {},
}

out_degree_distribution = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "dist_map": {
            "type": [str, type(None), PgxMap],
            "default": None,
            "subtype": {PgxMap: {"key_type": "integer", "value_type": "long"}},
        }
    },
    "return_value": {},
}

in_degree_distribution = {
    "in_arguments": {"graph": {"type": [PgxGraph], "default": None}},
    "out_arguments": {
        "dist_map": {
            "type": [str, type(None), PgxMap],
            "default": None,
            "subtype": {PgxMap: {"key_type": "integer", "value_type": "long"}},
        }
    },
    "return_value": {},
}

prim = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "weight": {
            "type": [EdgeProperty],
            "default": None,
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        },
    },
    "out_arguments": {
        "mst": {
            "type": [str, EdgeProperty],
            "default": "mst",
            "subtype": {EdgeProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {},
}

filtered_bfs = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "root": {"type": [PgxVertex, VertexSet], "default": None},
        "navigator": {"type": [VertexFilter], "default": None},
        "init_with_inf": {"type": [bool], "default": True},
        "max_depth": {"type": [int], "default": 2147483647},
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "distance",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
    },
    "return_value": {},
}

filtered_dfs = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "root": {"type": [PgxVertex, VertexSet], "default": None},
        "navigator": {"type": [VertexFilter], "default": None},
        "init_with_inf": {"type": [bool], "default": True},
        "max_depth": {"type": [int], "default": 2147483647},
    },
    "out_arguments": {
        "distance": {
            "type": [str, VertexProperty],
            "default": "distance",
            "subtype": {VertexProperty: {"type": "integer", "dimension": 0}},
        },
        "parent": {
            "type": [str, VertexProperty],
            "default": "parent",
            "subtype": {VertexProperty: {"type": "vertex", "dimension": 0}},
        },
    },
    "return_value": {},
}

all_reachable_vertices_edges = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "k": {"type": [int], "default": None},
        "filter": {"type": [EdgeFilter, type(None)], "default": None}
    },
    "out_arguments": {},
    "return_value": {
        "vertices_on_path": {"type": [VertexSet]},
        "edges_on_path": {"type": [EdgeSet]},
        "f_dist": {"type": [PgxMap]},
    },
}

compute_high_degree_vertices = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "k": {"type": [int], "default": None}
    },
    "out_arguments": {
        "high_degree_vertex_mapping": {
            "type": [PgxMap, str, type(None)],
            "default": None,
            "subtype": {PgxMap: {"key_type": "integer", "value_type": "vertex"}}
        },
        "high_degree_vertices": {"type": [VertexSet, type(None), str], "default": None}
    },
    "return_value": {}
}

create_distance_index = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "high_degree_vertex_mapping": {
            "type": [PgxMap, str, type(None)],
            "default": None,
            "subtype": {PgxMap: {"key_type": "integer", "value_type": "vertex"}}
        },
        "high_degree_vertices": {"type": [VertexSet, type(None), str], "default": None}
    },
    "out_arguments": {
        "index": {"type": [VertexProperty, str, type(None)], "default": None}
    },
    "return_value": {}
}

bipartite_check = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None}
    },
    "out_arguments": {
        "is_left": {
            "type": [str, type(None), VertexProperty],
            "default": "is_left",
            "subtype": {VertexProperty: {"type": "boolean", "dimension": 0}},
        }
    },
    "return_value": {}
}

enumerate_simple_paths = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "k": {"type": [int], "default": None},
        "vertices_on_path": {"type": [VertexSet], "default": None},
        "edges_on_path": {"type": [EdgeSet], "default": None},
        "dist": {
            "type": [PgxMap],
            "default": None,
            "subtype": {PgxMap: {"key_type": "vertex", "value_type": "integer"}}
        }
    },
    "out_arguments": {},
    "return_value": {}
}

limited_shortest_path_hop_dist = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "max_hops": {"type": [int], "default": None},
        "high_degree_vertex_mapping": {
            "type": [PgxMap],
            "default": None,
            "subtype": {PgxMap: {"key_type": "integer", "value_type": "vertex"}}
        },
        "high_degree_vertices": {"type": [VertexSet], "default": None},
        "index": {"type": [VertexProperty], "default": None}
    },
    "out_arguments": {
        "path_vertices": {"type": [VertexSequence, type(None)], "default": None},
        "path_edges": {"type": [EdgeSequence, type(None)], "default": None},
    },
    "return_value": {}
}

limited_shortest_path_hop_dist_filtered = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "src": {"type": [PgxVertex], "default": None},
        "dst": {"type": [PgxVertex], "default": None},
        "max_hops": {"type": [int], "default": None},
        "high_degree_vertex_mapping": {
            "type": [PgxMap],
            "default": None,
            "subtype": {PgxMap: {"key_type": "integer", "value_type": "vertex"}}
        },
        "high_degree_vertices": {"type": [VertexSet], "default": None},
        "index": {"type": [VertexProperty], "default": None},
        "filter": {"type": [EdgeFilter], "default": None}
    },
    "out_arguments": {
        "path_vertices": {"type": [VertexSequence, type(None)], "default": None},
        "path_edges": {"type": [EdgeSequence, type(None)], "default": None},
    },
    "return_value": {}
}

random_walk_with_restart = {
    "in_arguments": {
        "graph": {"type": [PgxGraph], "default": None},
        "source": {"type": [PgxVertex], "default": None},
        "length": {"type": [int], "default": None},
        "reset_prob": {"type": [float], "default": None}
    },
    "out_arguments": {
        "visit_count": {
            "type": [PgxMap, type(None)],
            "default": None,
            "subtype": {PgxMap: {"key_type": "vertex", "value_type": "integer"}}
        }
    },
    "return_value": {}
}

matrix_factorization_recommendations = {
    "in_arguments": {
        "bipartite_graph": {"type": [PgxGraph], "default": None},
        "user": {"type": [PgxVertex], "default": None},
        "vector_length": {"type": [int], "default": None},
        "feature": {"type": [VertexProperty], "default": None}
    },
    "out_arguments": {
        "estimated_rating": {
            "type": [str, VertexProperty, type(None)],
            "default": None,
            "subtype": {VertexProperty: {"type": "double", "dimension": 0}},
        }
    },
    "return_value": {}
}
