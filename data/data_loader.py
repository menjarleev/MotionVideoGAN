
def CreateDataLoader(opt):
    from .custom_dataset_data_loader import CustomDatasetLoader
    data_loader = CustomDatasetLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader