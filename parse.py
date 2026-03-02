import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from torch import nn
from torchvision import datasets
from torchvision.models import squeezenet1_1
import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=np.exceptions.VisibleDeprecationWarning)

def parse_dir_name(directory):
    params = {
        'run_id': '',
        'nodes': '',
        'bandwidth': '',
        'tdd': '',
        'rank': '',
        'network': 'wwan',
        'distribution': '',
    }

    dir_name_parts = directory.name.split('_')
    params['run_id'] = dir_name_parts[0]

    for part in dir_name_parts:
        if re.search(r"\d+MHz", part, re.IGNORECASE):
            match = re.match(r"(\d+)(MHz)", part, re.IGNORECASE)
            if match:
                params['bandwidth'] = match.group(1) + ' MHz'
        elif re.match(r"\d+N$", part):
            params['nodes'] = part
        elif re.match(r"\d+-\d+$", part):
            params['tdd'] = part
        elif re.search(r"MIMO", part, re.IGNORECASE):
            params['rank'] = '2x2'
        elif re.search(r"SISO", part, re.IGNORECASE):
            params['rank'] = '1x1'
        elif re.search(r"Dirichlet", part, re.IGNORECASE):
            params['distribution'] = 'dirichlet'
        elif re.search(r"IID", part, re.IGNORECASE):
            params['distribution'] = 'iid'
        elif re.search(r"WiFi", part, re.IGNORECASE):
            params['network'] = 'wlan'
            params['rank'] = ''
            params['tdd'] = ''
            params['congestion'] = ''
            params['bandwidth'] = ''
        elif re.search(r"wwan", part, re.IGNORECASE):
            params['network'] = 'wwan'
        elif re.search(r"Ethernet", part, re.IGNORECASE):
            params['network'] = 'lan'
            params['rank'] = ''
            params['tdd'] = ''
            params['congestion'] =  ''
            params['bandwidth'] = ''

    return params


def main():
    raw_data = Path.home() / 'Downloads' / 'FedAvg'
    output_dir = Path('data')

    # The MASTER DF FOR INDIVIDUAL
    df = pd.DataFrame(columns=['run_id', 'timestamp', 'nodes' ,'cid', 'bandwidth', 'tdd', 'rank', 'network', 'distribution'])

    # THE MASTER DF FOR AGGREGATED METRICS
    df_server = pd.DataFrame(columns=['run_id', 'timestamp', 'nodes', 'bandwidth', 'tdd', 'rank', 'network', 'distribution'])

    df_models = pd.DataFrame(columns=['run_id', 'nodes', 'bandwidth', 'tdd', 'rank', 'network', 'distribution'])

    # ML Stuff
    transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=250, shuffle=False)
    device = torch.device('mps')

    directories = list(raw_data.iterdir())
    for directory in tqdm(directories, desc='Processing experiments.'):
        if directory.name in ['Special cases', '.DS_Store']:
            continue
        experiment_params = parse_dir_name(directory)

        # Grab single file data
        exec_time_files = list(directory.glob('execution_time.txt'))
        start_time_files = list(directory.glob('start_time.txt'))

        # Grab the agg data
        agg_files = list(directory.glob('train_agg_metrics.csv'))

        if agg_files:
            agg_df = pd.read_csv(agg_files[0])
            for key in ['run_id', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'nodes']:
                agg_df[key] = experiment_params[key]
                agg_df['round'] = agg_df['server_round']


        # Grab the PHY layer metrics
        if experiment_params['network'] == 'wwan':
            common_files = list(Path(directory / 'phys_layer').glob('common.csv'))
            phy_files = Path(directory / 'phys_layer').glob('ue*.csv')

            if common_files and phy_files:
                phy_dfs = [pd.read_csv(f) for f in phy_files]
                phy_df = pd.concat(phy_dfs, ignore_index=True)
                phy_df = phy_df.drop(columns=['ueId', 'ranUeId', 'amfUeId', 'pmi'])

                common_df = pd.read_csv(common_files[0])
                common_df = common_df.drop(columns=['train_loss', 'train_time', 'eval_loss', 'eval_acc', 'eval_time'])

                seg_group = phy_df.groupby('segment')

                # Weight the BLER for each RNTI by the data sent per segement
                bler_weighted = seg_group.apply(lambda x: pd.Series({'dlBler': np.average(x['dlBler'], weights=x['dlBytes']),
                                                                     'ulBler': np.average(x['ulBler'], weights=x['ulBytes'])}))

                phy = seg_group.median(numeric_only=True).reset_index()
                phy[['dlBytes', 'ulBytes']] = seg_group[['dlBytes', 'ulBytes']].sum().values
                phy[['dlBler', 'ulBler']] = bler_weighted.values

                phy = phy.merge(common_df, on='segment', how='inner')
                phy['round'] = phy['server_round']
                phy.drop(columns=['server_round', 'timestamp'], inplace=True)

                agg_df = agg_df.merge(phy, on='round', how='inner')

        # Grab the individual data
        individual_files = list(directory.glob('individual_metrics.json'))
        if individual_files:
            data = json.load(open(individual_files[0]))
            rows = []
            for round_num, v in data.items():
                for record in v["train"]:
                    record["round"] = int(round_num)
                    rows.append(record)

            individual_df = pd.DataFrame(rows)
            for key in ['run_id', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'nodes']:
                individual_df[key] = experiment_params.get(key)

        # Get the latency files
        latency_files = list(directory.glob('latency_*_CID*.csv'))

        if latency_files:
            latency_parts = []
            for latency in latency_files:
                cid_match = re.search(r"CID(\d+)", latency.name)
                if cid_match:
                    latency_df = pd.read_csv(latency)
                    latency_df['cid'] = int(cid_match.group(1))
                    latency_df.index.name = 'round'
                    latency_df = latency_df.reset_index()
                    latency_df['round'] = latency_df['round'] + 1
                    latency_parts.append(latency_df)

            if latency_parts:
                latency_dfs = pd.concat(latency_parts, ignore_index=True)
                latency_dfs['run_id'] = experiment_params['run_id']

        individual_df = pd.merge(individual_df, latency_dfs, on=['run_id', 'cid', 'round'], how='inner') # This store all individual data for an experiment

        # Calc median of round
        agg_df = pd.merge(agg_df, latency_dfs.groupby(['run_id', 'round'])[['uplink_latency', 'downlink_latency']].mean(), on=['run_id', 'round'], how='inner')

        # Add the start_time and execution_time
        if start_time_files:
            with open(start_time_files[0], 'r') as f:
                start_time = f.readlines()[0].strip().replace('s', '') # This is in epoch time

        if exec_time_files:
            with open(exec_time_files[0], 'r') as f:
                exec_time = f.readlines()[0].strip().replace('s', '')


        individual_df['start_time'] = start_time
        individual_df['exec_time'] = exec_time
        agg_df['start_time'] = start_time
        agg_df['exec_time'] = exec_time


        # FINAL DF concat to master (Individual)
        df = pd.concat([df, individual_df]) # Concat it to the master
        df_server = pd.concat([df_server, agg_df])

        model_files = list(directory.glob('*.pt'))
        if model_files:
            model = squeezenet1_1(num_classes=10)
            model.load_state_dict(torch.load(model_files[0], weights_only=True))
            model.eval()

            model.to(device)
            correct, total, running_loss = 0, 0, 0.0
            all_predictions, all_labels, all_confidences = [], [], []
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    running_loss += criterion(outputs, labels).item()
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                    all_predictions.append(predicted.cpu())
                    all_labels.append(labels.cpu())
                    all_confidences.append(probs.max(dim=1).values.cpu())


            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            all_confidences = torch.cat(all_confidences)

            # Per class accuracy
            cm = confusion_matrix(all_labels, all_predictions)
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)


            # Expected Calibration Error
            n_bins = 10 # This is num classes
            tp = (all_predictions == all_labels).float() # True positive mask
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            n_samples = len(all_confidences)
            ece = 0.0

            for m in range(n_bins):
                bin_mask = (all_confidences > bin_boundaries[m]) & (all_confidences < bin_boundaries[m + 1])
                bin_size = bin_mask.sum().item()

                if bin_size == 0:
                    continue
                bin_acc = tp[bin_mask].mean().item()
                bin_conf = all_confidences[bin_mask].mean().item()

                ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

            # Save to DF
            model_df = pd.DataFrame([{
                'run_id': experiment_params['run_id'],
                'start_time': start_time,
                'exec_time': exec_time,
                'network': experiment_params['network'],
                'nodes': experiment_params['nodes'],
                'bandwidth': experiment_params['bandwidth'],
                'tdd': experiment_params['tdd'],
                'rank': experiment_params['rank'],
                'distribution': experiment_params['distribution'],
                'correct': correct,
                'total': total,
                'running_loss': running_loss,
                'per_class_accuracy': per_class_accuracy.tolist(),
                **{f'acc_{cls}': per_class_accuracy[i] for i, cls in enumerate(test_set.classes)},
                'accuracy': correct / total,
                'loss': running_loss / len(test_loader),
                'ece': ece,
            }])

        df_models = pd.concat([df_models, model_df])



    df.to_csv(output_dir / 'all_data.csv', index=False) # WLAN IS GONE HERE

    df_server.to_csv(output_dir / 'all_data_agg.csv', index=False)

    df_models.to_csv(output_dir / 'all_models_data.csv', index=False)


    # Filter and save the nodes data
    Path(output_dir, 'nodes').mkdir(parents=True, exist_ok=True)
    for n in [1,3,4,5,6]:
        df[df['nodes'] == f'{n}N'].to_csv(output_dir / 'nodes' / f'{n}nodes.csv', index=False)
        df_server[df_server['nodes'] == f'{n}N'].to_csv(output_dir / 'nodes' / f'{n}nodes_serverset.csv', index=False)
        df_models[df_models['nodes'] == f'{n}N'].to_csv(output_dir / 'nodes' / f'{n}nodes_modelset.csv', index=False)

    # Filter and save data for TDD
    Path(output_dir, 'tdd_split').mkdir(parents=True, exist_ok=True)
    for tdd in ['7-2', '5-4', '2-2', '2-7', '3-1']:
        df[df['tdd'] == tdd].to_csv(output_dir / 'tdd_split' / f'{tdd}_tdd.csv', index=False)
        df_server[df_server['tdd'] == tdd].to_csv(output_dir / 'tdd_split' / f'{tdd}_tdd_server_split.csv', index=False)
        df_models[df_models['tdd'] == tdd].to_csv(output_dir / 'tdd_split' / f'{tdd}_tdd_modelset.csv', index=False)

    # Filter and save data for Bandwidth
    Path(output_dir, 'bandwidth').mkdir(parents=True, exist_ok=True)
    for bw in ['20 MHz', '40 MHz', '80 MHz', '100 MHz']:
        df[df['bandwidth'] == bw].to_csv(output_dir / 'bandwidth' / f'{bw.replace(' ', '').lower()}_bandwidth.csv', index=False)
        df_server[df_server['bandwidth'] == bw].to_csv(output_dir / 'bandwidth' / f'{bw.replace(' ', '').lower()}_bandwidth_serverset.csv', index=False)
        df_models[df_models['bandwidth'] == bw].to_csv(output_dir / 'bandwidth' / f'{bw.replace(' ', '').lower()}_bandwidth_modelset.csv', index=False)

    # Filter and save data for Rank
    Path(output_dir, 'rank').mkdir(parents=True, exist_ok=True)
    for rank in ['2x2', '1x1']:
        df[df['rank'] == rank].to_csv(output_dir / 'rank' / f'{rank.lower()}_rank.csv', index=False)
        df_server[df_server['rank'] == rank].to_csv(output_dir / 'rank' / f'{rank.lower()}_rank_serverset.csv', index=False)
        df_models[df_models['rank'] == rank].to_csv(output_dir / 'rank' / f'{rank.lower()}_rank_modelset.csv', index=False)

    # Filter and save data for Distribution
    Path(output_dir, 'distribution').mkdir(parents=True, exist_ok=True)
    for dist in ['dirichlet', 'iid']:
        df[df['distribution'] == dist].to_csv(output_dir / 'distribution' / f'{dist.lower()}.csv', index=False)
        df_server[df_server['distribution'] == dist].to_csv(output_dir / 'distribution' / f'{dist.lower()}_serverset.csv', index=False)
        df_models[df_models['distribution'] == dist].to_csv(output_dir / 'distribution' / f'{dist.lower()}_modelset.csv', index=False)


if __name__ == '__main__':
    main()