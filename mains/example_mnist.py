import tensorflow as tf
from tensorflow.python import debug as tf_debug

from data_loader.data_generator_mnist import MnistImgLoader
from models.mnist_model import MnistModel
from trainers.mnist_trainer import MnistTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Jiang-Ubuntu:6004")
    #sess.run(my_fetches)

    # create your data generator
    data_loader = MnistImgLoader(config)

    # create instance of the model you want
    model = MnistModel(data_loader, config)

    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir, 
                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                            'test/loss_per_epoch','test/acc_per_epoch'])

    # create trainer and path all previous components to it
    trainer = MnistTrainer(sess, model, config, logger, data_loader)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
