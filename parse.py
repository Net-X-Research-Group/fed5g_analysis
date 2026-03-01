import json
import re
from pathlib import Path
import numpy as np
import pandas as pd


def parse_dir_name(directory):
    params = {
        'run_id': None,
        'nodes': None,
        'bandwidth': None,
        'tdd': None,
        'rank': None,
        'network': 'wwan',
        'distribution': None,
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
            params['rank'] = None
            params['tdd'] = None
            params['congestion'] = None
            params['bandwidth'] = None
        elif re.search(r"wwan", part, re.IGNORECASE):
            params['network'] = 'wwan'
        elif re.search(r"Ethernet", part, re.IGNORECASE):
            params['network'] = 'lan'
            params['rank'] = None
            params['tdd'] = None
            params['congestion'] = None
            params['bandwidth'] = None

    return params


def main():
    raw_data = Path.home() / 'Downloads' / 'FedAvg'
    output_dir = Path('data')

    # The MASTER DF FOR INDIVIDUAL
    df = pd.DataFrame(columns=['run_id', 'timestamp', 'nodes' ,'cid', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'uplink_latency', 'downlink_latency'])

    # THE MASTER DF FOR AGGREGATED METRICS
    df_server = pd.DataFrame(columns=['run_id', 'timestamp', 'nodes', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'uplink_latency', 'downlink_latency'])

    for directory in raw_data.iterdir():
        if directory.name in ['Special cases', '.DS_Store']:
            continue
        experiment_params = parse_dir_name(directory)

        # Grab single file data
        exec_time_files = list(directory.glob('execution_time.txt'))
        start_time_files = list(directory.glob('start_time.txt'))

        # Grab the agg data
        agg_files = list(directory.glob('train_agg_metrics.csv'))

        if agg_files:
            raw_agg_data = pd.read_csv(agg_files[0])
            for key in ['run_id', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'nodes']:
                raw_agg_data[key] = experiment_params[key]
                raw_agg_data['round'] = raw_agg_data['server_round']


        # Grab the PHY layer metrics
        if experiment_params['network'] == 'wwan':
            phy_files = Path(directory / 'phys_layer').glob('ue*.csv')
            phy_dfs = [pd.read_csv(f) for f in phy_files]
            phy_df = pd.concat(phy_dfs, ignore_index=True)
            phy_df = phy_df.drop(columns=['ueId', 'ranUeId', 'amfUeId', 'pmi'])

            seg_group = phy_df.groupby('segment')

            # Weight the BLER for each RNTI by the data sent per segement
            bler_weighted = seg_group.apply(lambda x: pd.Series({'dlBler': np.average(x['dlBler'], weights=x['dlBytes']),
                                                                 'ulBler': np.average(x['ulBler'], weights=x['ulBytes'])}))

            phy = seg_group.median(numeric_only=True).reset_index()
            phy[['dlBytes', 'ulBytes']] = seg_group[['dlBytes', 'ulBytes']].sum().values
            phy[['dlBler', 'ulBler']] = bler_weighted.values



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
            latency_dfs = pd.DataFrame(columns=['round', 'cid', 'uplink_latency', 'downlink_latency'])
            for latency in latency_files:
                cid_match = re.search(r"CID(\d+)", latency.name)
                if cid_match:
                    latency_df = pd.read_csv(latency)
                    latency_df['cid'] = int(cid_match.group(1))
                    latency_df.index.name = 'round'
                    latency_df = latency_df.reset_index()
                    latency_df['round'] = latency_df['round'] + 1
                    latency_dfs = pd.concat([latency_dfs, latency_df])
            latency_dfs['run_id'] = experiment_params['run_id']
            latency_dfs.reset_index(inplace=True)
        individual_df = pd.merge(individual_df, latency_dfs, on=['run_id', 'cid', 'round'], how='inner') # This store all individual data for an experiment

        # Add the start_time and execution_time
        if start_time_files:
            with open(start_time_files[0], 'r') as f:
                start_time = f.readlines()[0].strip().replace('s', '') # This is in epoch time

        if exec_time_files:
            with open(exec_time_files[0], 'r') as f:
                exec_time = f.readlines()[0].strip().replace('s', '')


        individual_df['start_time'] = start_time
        individual_df['exec_time'] = exec_time


        # FINAL DF concat to master (Individual)
        df = pd.concat([df, individual_df]) # Concat it to the master



    df.to_csv(output_dir / 'all_data.csv', index=False) # WLAN IS GONE HERE

    df_server.to_csv(output_dir / 'all_data_agg.csv', index=False)


    # Filter and save the nodes data
    Path(output_dir, 'nodes').mkdir(parents=True, exist_ok=True)
    for n in [1,3,4,5,6]:
        df[df['nodes'] == f'{n}N'].to_csv(output_dir / 'nodes' / f'{n}_nodes.csv', index=False)

    # Filter and save data for TDD
    Path(output_dir, 'tdd_split').mkdir(parents=True, exist_ok=True)
    for tdd in ['7-2', '5-4', '2-2', '2-7', '3-1']:
        df[df['tdd'] == tdd].to_csv(output_dir / 'tdd_split' / f'{tdd}_tdd.csv', index=False)

    # Filter and save data for Bandwidth
    Path(output_dir, 'bandwidth').mkdir(parents=True, exist_ok=True)
    for bw in ['20 MHz', '40 MHz', '80 MHz', '100 MHz']:
        df[df['bandwidth'] == bw].to_csv(output_dir / 'bandwidth' / f'{bw.replace(' ', '').lower()}_bandwidth.csv', index=False)

    # Filter and save data for Rank
    Path(output_dir, 'rank').mkdir(parents=True, exist_ok=True)
    for rank in ['2x2', '1x1']:
        df[df['rank'] == rank].to_csv(output_dir / 'rank' / f'{rank.lower()}_rank.csv', index=False)

    # Filter and save data for Distribution
    Path(output_dir, 'distribution').mkdir(parents=True, exist_ok=True)
    for dist in ['dirichlet', 'iid']:
        df[df['distribution'] == dist].to_csv(output_dir / 'distribution' / f'{dist.lower()}.csv', index=False)

if __name__ == '__main__':
    main()