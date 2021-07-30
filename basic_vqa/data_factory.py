import data_loader
import data_loader_v2
import config

def get_dataloader():
    dataloader = None
    if config.USE_OLD_DATALOADER:
        dataloader = data_loader.get_loader(
            input_dir=config.INPUT_DIR,
            input_vqa_train='train.npy',
            input_vqa_valid='valid.npy',
            max_qst_length=config.MAX_QST_LEN,
            max_num_ans=config.MAX_NUM_ANS,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            train_portion=config.TRAIN_PORTION)
    else:
        dataloader = data_loader_v2.get_loader(
            input_dir=config.INPUT_DIR,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            train_portion=config.TRAIN_PORTION)
    return dataloader
