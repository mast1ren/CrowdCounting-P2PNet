# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data
    elif args.dataset_file == 'DRONEBIRD':
        from crowd_datasets.DRONEBIRD.loading_data import loading_data
        return loading_data

    return None