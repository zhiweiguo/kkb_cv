class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'laneline':
            return '/home/aim/WorkSpace/my_workspace/kkb_cv/practical_course/week_5/data/train_val'  # folder that contains VOCdevkit/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
