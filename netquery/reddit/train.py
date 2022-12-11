from argparse import ArgumentParser

from netquery.utils import *
from netquery.reddit.data_utils import load_graph
from netquery.model import QueryEncoderDecoder
from netquery.train_helpers import run_train

from torch import optim

parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="/dfs/scratch0/nqe-reddit/100_graph")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_iter", type=int, default=100000)
parser.add_argument("--val_every", type=int, default=1000)
parser.add_argument("--max_path_len", type=int, default=3)
parser.add_argument("--max_inter_size", type=int, default=3)
parser.add_argument("--tol", type=float, default=0.0001)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--log_dir", type=str, default="/dfs/scratch0/nqe-reddit/training_logs/")
parser.add_argument("--model_dir", type=str, default="/dfs/scratch0/nqe-reddit/trained_models/")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")
parser.add_argument("--val_prop", type=float, default=0.1)
parser.add_argument("--opt", type=str, default="adam")
args = parser.parse_args()


graph, feature_modules = load_graph(args.data_dir, args.embed_dim)
if args.cuda:
    graph.features = cudify(feature_modules)
out_dims = {mode:args.embed_dim for mode in graph.relations}

train_paths, test_paths, train_inters, test_inters = load_train_test_data(args.data_dir)
train_paths, val_paths = get_val(train_paths, args.val_prop)
train_inters, val_inters = get_val(train_inters, args.val_prop)

enc = get_encoder(args.depth, graph, out_dims, feature_modules, args.cuda)
dec = get_metapath_decoder(graph, enc.out_dims if args.depth > 0 else out_dims, args.decoder)
inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder)
    
enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
if args.cuda:
    enc_dec.cuda()

if args.opt == "sgd":
    optimizer = optim.SGD([p for p in enc_dec.parameters() if p.requires_grad], lr=args.lr, momentum=0)
elif args.opt == "adam":
    optimizer = optim.Adam([p for p in enc_dec.parameters() if p.requires_grad], lr=args.lr)
    
log_file = args.log_dir + "/{data:s}-{depth:d}-{embed_dim:d}-{lr:f}-{decoder:s}-{inter_decoder:s}.log".format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        inter_decoder=args.inter_decoder)
print(log_file)
model_file = args.model_dir + "/{data:s}-{depth:d}-{embed_dim:d}-{lr:f}-{decoder:s}-{inter_decoder:s}.log".format(
        data=args.data_dir.strip().split("/")[-1],
        depth=args.depth,
        embed_dim=args.embed_dim,
        lr=args.lr,
        decoder=args.decoder,
        inter_decoder=args.inter_decoder)
logger = setup_logging(log_file)

run_train(enc_dec, optimizer, train_paths, val_paths, test_paths, train_inters, val_inters, test_inters, logger, 
        max_iter=args.max_iter, max_path_len=args.max_path_len, max_inter_size=args.max_inter_size, val_every=args.val_every)
torch.save(enc_dec.state_dict(), model_file)
