import fire

from util.dataset_util import dataset_distribute

if __name__ == '__main__':
    fire.Fire({
        "dataset_distribute": dataset_distribute,
    })
