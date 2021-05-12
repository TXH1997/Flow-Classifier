import src.trainer as tr
import src.utils as utils

if __name__ == '__main__':
    config = utils.read_conf('./src/experiments.conf', 'cnn')
    device = utils.get_device(0)
    trainer = tr.Trainer(device, config)
    trainer.train()
