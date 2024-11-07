import argparse
import os

#
# Global Configs
#

THREAD_POOL_WORKERS = 1

# WANDB
WANDB_PROJECT_ID = "FAME2MI"
# WANDB_ENTITY_ID = "b89befc18787e4d6d6b95ce54fcec0fafb10f589"


# GLOBAL CHECKPOINT DIR
CHECKPOINT_DIR = "/home/sun/project/FAME2Pre/checkpoint/"
# Graph Dataset - I KNOW ITS NOT OKAY TO MAKE THIS PARAMS LIKE THAT
#  BUT GRAPH DATASET CAN ONLY TAKE GLOBAL AVAILABLE ARGUMENTS.
# TODO: DESCRIBE
GRAPH_EDGE_FACTOR = 1
GRAPH_NEIGHBORING_K = 10
FAME2_DUMP_DIR =  '/home/sun/data/FAME2labelling'
FILEPATH_CLINICAL_EVENT_DF = 'data/fame2_clinical_events_2year_data.csv'
EVENT_COLUMNS = ['VOCE', 'UR_TVF', 'NUR_TVF', 'MI_TVF', 'CV_TVF']
SPLIT_PATH = "/home/sun/data/"

# Works as an argument parser and also stores all default configurations
def parse_args():
    parser = argparse.ArgumentParser()

    # Run/Task specific flags (for task specific runs that are connected with optimization look below)
    parser.add_argument("--nb_clinical_data_features", default=24, type=int, help="The number of clinical data features, if used with ClinicalDataPooling")
    parser.add_argument("--run_mode", default="simple", type=str, help="'simple' or 'hypertune'")
    parser.add_argument("--init_weights_from", default=None, type=str, help="Load model weights from checkpoint file '.ckpt'")
    parser.add_argument("--init_weights_for", default="both", type=str, help="Init weights for 'gnn', 'pooling', 'both'")
    parser.add_argument("--nb_classes", type=int, default=1, help="The number of classes to predict for our clf task")
    parser.add_argument("--retain_logs_in_ram", default=False, type=bool, help="Retain the logs in ram, so afterwards you could dump them in a pickle and create plots.")
    parser.add_argument("--task", default="forecast", type=str, help="The task to execute. Can be 'regression', 'baseline' (for CNN), 'detection', 'forecast' ")
    parser.add_argument("--loss_func", default="BCEWithLogitsLoss", type=str, help="The loss function, taken from torch.nn. Specifically pick `BCEWithLogitsLoss`, `BCELoss`, `CrossEntropyLoss`")
    parser.add_argument("--checkpoint_after_n_epochs", default=10, type=int, help="Checkpoint after a specific number of epochs")
    parser.add_argument("--standardize_global", default=False, help="Standardizes the global features")
    parser.add_argument("--save_top_k", default=2, type=int, help="Check pytorch_lightning.callbacks.ModelCheckpoint")
    parser.add_argument("--use_lesion_wide_info", default=True,  help="Uses lesion wide features like FFR and DS")
    parser.add_argument("--save_hyper_path", type=str, help="save hyperParameters")



    # Kfold specific args
    parser.add_argument("--nb_folds", default=5, type=int, help="Folds number if training_mode is kfold")
    parser.add_argument("--nb_optuna_trials", default=5, type=int, help="Number of optuna trials")
    parser.add_argument("--hypertune_metric", default="F1Score", type=str, help="Metric to optimize: F1Score, Accuracy, Precision, Recall")

    # Data location arguments
    parser.add_argument("--dataset_dir", default="/home/sun/data/graphK5/", type=str, help="Give the path that our data-loaders we use to load the data")

    # hacky way to standardize the torch.geometric dataset !!!
    parser.add_argument("--m", default=0)
    parser.add_argument("--v", default=1)
    parser.add_argument("--dont_validate", default=True,action="store_true", help="Used on kfold/hypertune to create a dataset with just train and test and no validation")

    # Optimization specs
    parser.add_argument("--nb_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.000111671409982944673, type=float, help="The initial lr.")
    parser.add_argument("--weight_init_strategy", default="Xavier Normal", type=str, help="Pick one: Xavier Uniform, Xavier Normal,Kaiming Uniform, Kaiming Normal")
    parser.add_argument("--weight_decay", default=0.001417599992947637583, type=float, help="Weight decay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--adam_beta1", default=0.7736289028321897,  type=float, help="1st beta for Adam.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="2nd beta for Adam.")
    parser.add_argument("--scheduler", default=None, type=str, help="What type of scheduler to use: constant or linear.")
    parser.add_argument("--gradient_clipping", default=3, type=float, help="Clip gradients")

    # Batches
    parser.add_argument("--use_balanced_batches", default=True,action='store_true', help="Use a sampler from pytorch geometric that keeps batches balanced: IMBALANCED_SAMPLER")
    parser.add_argument("--train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--test_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--limit_batches_during_training", default=None, type=float, help="Limit the batches in training (eg use 5 of batch size for debugging)")

    # Logging and evaluating and model save
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use weights and biases for logging model stats")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Online or offline")
    parser.add_argument("--model_watch_interval", type=int, default=5, help="When to upload model stats to wandb. Batch granularity")
    parser.add_argument("--checkpoint", default=True,action='store_true', help="Whether to checkpoint, to a folder called checkpoints (Top2 using F1score") # TODO make monitor value and K configurable
    parser.add_argument("--use_scheduler", action='store_true', help="Whether to use an ReduceLROnPlateau scheduler ")
    parser.add_argument("--log_every_steps", type=int, default=50, help="Log training progress every X steps (iters)")
    parser.add_argument("--eval_every_epochs", type=int, default=1, help="Evaluate every X epochs.")
    parser.add_argument("--eval_check_interval", type=float, default=None, help="Calls eval every X percent of the epoch to check the training.")
    parser.add_argument("--save_model", type=bool, default=False, help="Save the model or not")

    # Runtime config
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization.")
    parser.add_argument("--device", type=str, default="cuda", help="What device to use: 'cuda', 'mps','cpu'")
    # parser.add_argument("--num_workers", default=os.cpu_count(), type=int, help="Workers in dataloader.")
    parser.add_argument("--num_workers", default=1, type=int, help="Workers in dataloader.")

    parser.add_argument("--debug", default=True, type=bool, help="Debug Flag")


    # GNN dataset parameters
    parser.add_argument("--standardize_graph", default=False,action='store_true', help="Standardizes the input of the GNN")
    parser.add_argument("--skip_pixel_intensity", action='store_true', help="Skips the pixel intensity")
    parser.add_argument("--train_val_split", type=str, default=None, help="Clearly defins how to split the train dataset to train & validation for non kfold runs")

    # GNN model parameters
    # -- pooling
    parser.add_argument("--gnn_pooling_cls_name", type=str, default="ConfigurablePooling", help="One of pooling layers define din 'cardio.networks.gnn' ")
    parser.add_argument("--pool_only_lesion_points", action='store_true', help="In case we use whole artery, gnn pooling will pool only lesion points")
    parser.add_argument("--node_pooling", type=str, default='sum', help="How to pool the nodes of the gnn: sum, mean, max, all_stats")
    # -- model
    parser.add_argument("--gnn_is_siamese", default=False, help="In case we use siamese dataset use this flag")
    parser.add_argument("--gnn_cls_name", type=str, default="GIN", help="One of the pred class names in 'torch_geometric.nn.models.basic_gnn' or one of our custom gnn class names, CustomGIN")
    parser.add_argument("--gnn_node_emb_dim", type=int, default=256, help="GNN hidden dim that will be input of the GATconv")
    parser.add_argument("--gnn_global_hidden_dim", type=int, default=512, help="GNN The dimension to have after the pooling ")
    parser.add_argument("--gnn_nb_layers", type=int, default=1, help="GNN number of Transformer layers")
    parser.add_argument("--gnn_dropout", type=float, default=0.415081799988646554, help="GNN The dropout on the Attention layers of the gnn")
    parser.add_argument("--gnn_act", type=str, default="relu", help="The type of activation (one of torch activations)")
    parser.add_argument("--gnn_norm", type=str, default="BatchNorm", help="One of the normalizations, see 'torch_geometric.nn.norm.__init__.py'")
    parser.add_argument("--gnn_freeze_weights", type=bool, default=False, help="Freeze gnn weights (not pooling)")
    parser.add_argument("--gnn_ffr_pooling_factor", type=int, default=1, help="Scale the global dim by a factor multiple times (until dim is >= 16) before you concat it with ffr")
    parser.add_argument("--gnn_ffr_pooling_proj_dim", type=int, default=16, help="Stops scalling when you reach this projection dimention")

    # This is automatically set from the dataset module
    parser.add_argument("--gnn_node_features", type=int, default=100, help="Dim of features in the nodes that we will feed to GNN")

    # CNN model parameters
    parser.add_argument("--standardize_img", action='store_true', help="Standardizes the input of the GNN")
    parser.add_argument("--cnn_dropout", type=float, default=0.45081799988646554, help="The dropout of the CNN fully connected layer at the end")

    # CNN model that inputs data to GNN
    parser.add_argument("--cnn_out_channels", type=int, default=64, help="The num of features the CNN needs to create, and pass to GNN a channels")
    parser.add_argument("--cnn_kernel", type=int, default=3, help="The kernel to use for the CNN")

    # Transformer for radiomix model parameters
    parser.add_argument("--tr_nb_radiomix_features", type=int, default=46, help="The number of different radiomix features we use")
    parser.add_argument("--tr_d_model", type=int, default=512, help="The dim of the transformer embeddings")
    parser.add_argument("--tr_nhead", type=int, default=8, help="The number of heads")
    parser.add_argument("--tr_dropout", type=float, default=0.1, help="The number of heads")
    parser.add_argument("--tr_nb_layers", type=int, default=6, help="The number of layers on the encoder")
    parser.add_argument("--tr_dim_feedforward", type=int, default=2048, help="The feed forward dimension on the encode layers")

    return parser.parse_args()
