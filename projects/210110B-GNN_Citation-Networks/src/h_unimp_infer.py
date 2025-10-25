import os
import argparse
import yaml
import paddle
import tqdm
import numpy as np

from easydict import EasyDict as edict
from dataset.data_generator_citationnetwork import CitationNetwork, DataGenerator
import models
from pgl.utils.logger import log
from utils import load_model, _create_if_not_exist


def infer(config):
    """Inference function"""
    dataset = CitationNetwork(config)
    dataset.prepare_data()

    test_iter = DataGenerator(
        dataset=dataset,
        samples=config.samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_type="test")

    model_params = dict(config.model.items())
    model_params['m2v_dim'] = config.m2v_dim
    model = getattr(models, config.model.name).GNNModel(**model_params)

    load_model(config.output_path, model)
    model.eval()

    pred_temp = []
    with paddle.no_grad():
        for batch in tqdm.tqdm(test_iter.generator()):
            graph_list, x, m2v_x, y, label_y, label_idx = batch
            x = paddle.to_tensor(x, dtype='float32')
            m2v_x = paddle.to_tensor(m2v_x, dtype='float32')
            label_y = paddle.to_tensor(label_y, dtype='int64')
            label_idx = paddle.to_tensor(label_idx, dtype='int64')

            graph_list = [(item[0].tensor(), paddle.to_tensor(item[2]))
                          for item in graph_list]
            
            out = model(graph_list, x, m2v_x, label_y, label_idx)
            pred_temp.append(out.numpy())

    predictions = np.concatenate(pred_temp, axis=0)
    
    # Save predictions
    output_dir = os.path.join(config.output_path, 'predictions')
    _create_if_not_exist(output_dir)
    
    pred_file = os.path.join(output_dir, f'{config.test_name}_predictions.npy')
    np.save(pred_file, predictions)
    
    log.info(f"Predictions saved to {pred_file}")
    log.info(f"Prediction shape: {predictions.shape}")
    
    # Also save test indices for reference
    test_idx_file = os.path.join(output_dir, f'{config.test_name}_indices.npy')
    np.save(test_idx_file, dataset.test_idx)
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='R-UniMP Inference for Citation Network')
    parser.add_argument("--conf", type=str, default="./configs/r_unimp_citationnetwork.yaml")
    args = parser.parse_args()
    
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.samples = [int(i) for i in config.samples.split('-')]

    print(config)
    infer(config)
