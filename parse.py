import json
import re
from pathlib import Path

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

    # The MASTER DF
    df = pd.DataFrame(columns=['run_id', 'timestamp', 'nodes' ,'cid', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'uplink_latency', 'downlink_latency'])

    for directory in raw_data.iterdir():
        if directory.name in ['Special cases', '.DS_Store']:
            continue
        experiment_params = parse_dir_name(directory)

        # Get the latency files
        latency_files = list(directory.glob('latency_*_CID*.csv'))

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

        if latency_files:
            for latency in latency_files:
                cid_match = re.search(r"CID(\d+)", latency.name)
                if cid_match:
                    latency_df = pd.read_csv(latency)
                    latency_df['cid'] = int(cid_match.group(1))
                    latency_df.index.name = 'round'
                    latency_df = latency_df.reset_index()
                    latency_df['round'] = latency_df['round'] + 1
                    # After building individual_df, add experiment params
                    for key in ['run_id', 'bandwidth', 'tdd', 'rank', 'network', 'distribution', 'nodes']:
                        individual_df[key] = experiment_params.get(key)


                    merged = individual_df.merge(latency_df, on=['cid', 'round'], how='inner')

                    df = pd.concat([df, merged], ignore_index=True)


    df.to_csv(output_dir / 'all_data.csv', index=False)

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