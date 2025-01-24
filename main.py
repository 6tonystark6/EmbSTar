import os
import argparse
import warnings

warnings.filterwarnings('ignore')

from load_data import load_dataset, split_dataset, allocate_dataset, Dataset_Config
from attack_model import EmbSTar

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', default='FLICKR', choices=['FLICKR', 'COCO', 'NUS'])
parser.add_argument('--knockoff', dest='knockoff', default=True, choices=[True, False])
parser.add_argument('--dataset_path', dest='dataset_path', default='./Datasets/')
parser.add_argument('--attacked_method', dest='attacked_method', default='DPSH', choices=['DPSH', 'HashNet', 'CSQ'])
parser.add_argument('--attacked_models_path', dest='attacked_models_path', default='attacked_models/')
parser.add_argument('--knockoff_bit', dest='kb', type=int, default=32)
parser.add_argument('--knockoff_epochs', dest='ke', type=int, default=1)
parser.add_argument('--knockoff_batch_size', dest='kbz', type=int, default=24)
parser.add_argument('--knockoff_image_learning_rate', dest='klr', type=float, default=1e-4)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--gpu', dest='gpu', type=str, default='0', choices=['0', '1', '2', '3'])
parser.add_argument('--bit', dest='bit', type=int, default=32, choices=[16, 32, 48, 64])
parser.add_argument('--batch_size', dest='batch_size', type=int, default=24)
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=200)
parser.add_argument('--n_epochs_decay', type=int, default=50)
parser.add_argument('--epoch_count', type=int, default=1)
parser.add_argument('--learning_rate', dest='lr', type=float, default=1e-4)
parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step', 'plateau', 'cosine'],)
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100)
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=50)
parser.add_argument('--sample_dir', dest='sample', default='samples/')
parser.add_argument('--output_path', dest='output_path', default='outputs/')
parser.add_argument('--output_dir', dest='output_dir', default='DPSH_FLICKR_32')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

DataConfigs = Dataset_Config(args.dataset, args.dataset_path)
X, Y, L = load_dataset(DataConfigs.data_path)
X_s, Y_s, L_s = split_dataset(X, Y, L, DataConfigs.query_size, DataConfigs.training_size, DataConfigs.database_size)
image_train, _, label_train, image_database, _, label_database, image_test, _, label_test = allocate_dataset(X_s, Y_s,
                                                                                                             L_s)

model = EmbSTar(args=args, DataConfigs=DataConfigs)

if args.knockoff:
    model.train_knockoff(image_train)
    model.test_knockoff(image_test, label_test, image_database, label_database)

if args.train:
    model.train(image_train, label_train, image_database, label_database, image_test, label_test)

if args.test:
    model.test(image_database, label_database, image_test, label_test, args.dataset, args.attacked_method, args.bit)
