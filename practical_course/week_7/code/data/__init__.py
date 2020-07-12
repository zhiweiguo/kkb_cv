from data import laneline
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'laneline':
        train_set = laneline.LaneLine(args, split='train')
        val_set = laneline.LaneLine(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError