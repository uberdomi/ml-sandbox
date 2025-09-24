from src import create_dataset, SupportedDatasets, list_supported_datasets

if __name__ == "__main__":
    # List all supported datasets
    datasets = list_supported_datasets()
    print(f"Supported datasets: {', '.join(datasets)}")
    
    datasets = (SupportedDatasets.MNIST, SupportedDatasets.FASHION_MNIST, SupportedDatasets.CIFAR10)
    # datasets = (SupportedDatasets.CIFAR10,)

    for dataset in datasets:
        print(f"\n{'='*10} Working with {dataset.name} {'='*10}\n")

        dataset_instance = create_dataset(dataset, root="demo_data", force_download=False)

        # Print dataset information and show random samples
        dataset_instance.print_info()
        print(len(dataset_instance))
        
        dataset_instance.show_random_samples(num_samples=5)
        dataset_instance.show_illustrative_samples()

    print("\n" + "="*20 + " Success! " + "="*20 + "\n")
