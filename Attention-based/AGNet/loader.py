from torch.utils.data import DataLoader

from dataprocess import *


def data_loader(args, path):
    dset = CongreG8Dataset(
        path,
        seq_len=args.seq_len,
        slidingwindow_len=args.slidingwindow_len,
        delim=args.delim)
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers
        )
    return dset, loader



def data_loader_dat(args, path):
    dset = CongreG8DatasetDat(
        path,
        seq_len=args.seq_len,
        slidingwindow_len=args.slidingwindow_len,
        delim=args.delim)
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers
        )
    return dset, loader