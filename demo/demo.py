import os
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline

if __name__ == '__main__':
    num_models = pipeline.num_models
    ensemble = pipeline.GeoGoatEnsemble(num_models)
    ensemble.load_state_dict(torch.load('./checkpoints/ensemble.pt', weights_only=True))

    pipeline.visualize('demo/upload.jpg', None, fname='demo.jpg')
