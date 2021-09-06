import os
import sys
import dill
import json
import argparse
import torch
import numpy as np
import tqdm
sys.path.append("Trajectron_plus_plus/trajectron")
from trajectronEWTA import TrajectronEWTA
from Trajectron_plus_plus.trajectron.model.model_registrar import ModelRegistrar
from Trajectron_plus_plus.trajectron.evaluation import evaluation


PARAMS = {
    'eth-ucy': (7, 12),
    'nuScenes': (1, 8)
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model full path", type=str)
    parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
    parser.add_argument("--data", help="full path to data file", type=str)
    parser.add_argument("--node_type", type=str)
    parser.add_argument("--kalman", type=str)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    if 'ewta' in model_dir and 'nuScenes' not in model_dir:
        model_registrar.load_models(ts)
    else:
        model_registrar.model_dict.clear()
        checkpoint_path = os.path.join(model_dir, 'model_registrar-%d.pt' % ts)
        checkpoint = torch.load(checkpoint_path, map_location=model_registrar.device)
        model_registrar.model_dict = checkpoint['model_dict']
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    trajectron = TrajectronEWTA(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    set_seed(0)
    args = parse_arguments()
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)
    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes
    for scene in tqdm.tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])
    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']
    prediction_parameters = PARAMS['nuScenes'] if 'nuScenes' in args.data else PARAMS['eth-ucy']
    with torch.no_grad():
        print('processing %s' % args.node_type)
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        for scene in tqdm.tqdm(scenes):
            timesteps = np.arange(scene.timesteps)
            predictions, features = eval_stg.predict(
                scene, timesteps, ph, min_history_timesteps=prediction_parameters[0],
                min_future_timesteps=prediction_parameters[1],
                selected_node_type=args.node_type)
            if features is None:
                continue
            batch_error_dict = evaluation.compute_batch_statistics(
                predictions, scene.dt, max_hl=max_hl, ph=ph,
                node_type_enum=env.NodeType, map=None,
                best_of=True,
                prune_ph_to_future=True)
            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        total_number_testing_samples = eval_fde_batch_errors.shape[0]
        print('All         (ADE/FDE): %.2f, %.2f   --- %d' % (
            eval_ade_batch_errors.mean(),
            eval_fde_batch_errors.mean(),
            total_number_testing_samples))
        if args.kalman:
            with open(args.kalman, 'rb') as f:
                kalman_errors = dill.load(f, encoding='latin1')
            assert kalman_errors.shape[0] == eval_fde_batch_errors.shape[0]
            largest_errors_indexes = np.argsort(kalman_errors)
            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            for top_index in range(1, 4):
                challenging = largest_errors_indexes[-int(
                    total_number_testing_samples * top_index / 100):]
                fde_errors_challenging = np.copy(eval_fde_batch_errors)
                ade_errors_challenging = np.copy(eval_ade_batch_errors)
                mask[challenging] = False
                fde_errors_challenging[mask] = 0
                ade_errors_challenging[mask] = 0
                print('Challenging Top %d (ADE/FDE): %.2f, %.2f   --- %d' %
                      (top_index,
                       np.sum(ade_errors_challenging) / len(challenging),
                       np.sum(fde_errors_challenging) / len(challenging),
                       len(challenging)))
