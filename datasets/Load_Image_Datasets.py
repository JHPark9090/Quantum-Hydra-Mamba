import random, copy
import numpy as np
import mne
import torch
import torchvision
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


seed = 2024
dataset_name = "CIFAR"



def load_mnist(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([ToTensor(), lambda x: x.view(-1)])  # Flatten MNIST images
    data_train = MNIST(root='./data', train=True, download=True, transform=transform)
    data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    input_dim = 28 * 28

    X_train = data_train.data.float() / 255.0  # Normalize pixel values to [0, 1]
    y_train = data_train.targets.clone().detach()
    X_test = data_test.data.float() / 255.0
    y_test = data_test.targets.clone().detach()
    
    # Shuffle data
    shuffle_idx = torch.randperm(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx2 = torch.randperm(n_valtest)
    X_test = X_test[shuffle_idx2]
    y_test = y_test[shuffle_idx2]

    # Limit dataset size
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_valtest]
    y_test = y_test[:n_valtest]    
    
    # Flatten images
    X_train = X_train.view(-1, 28*28)
    X_test = X_test.view(-1, 28*28)
    
    # Create TensorDatasets
    train_X = X_train.to(device)
    train_y = y_train.to(device)
    test_X = X_test.to(device)
    test_y = y_test.to(device)

    train_dataset = TensorDataset(train_X, train_y)
    valtest_dataset = TensorDataset(test_X, test_y)

    # Equally split validation and test sets
    val_size = int(0.5 * len(valtest_dataset))
    test_size = int(0.5 * len(valtest_dataset))
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim


def load_fashion(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([ToTensor(), lambda x: x.view(-1)])  # Flatten MNIST images
    data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    input_dim = 28 * 28

    X_train = data_train.data.float() / 255.0  # Normalize pixel values to [0, 1]
    y_train = data_train.targets.clone().detach()
    X_test = data_test.data.float() / 255.0
    y_test = data_test.targets.clone().detach()
    
    # Shuffle data
    shuffle_idx = torch.randperm(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx2 = torch.randperm(n_valtest)
    X_test = X_test[shuffle_idx2]
    y_test = y_test[shuffle_idx2]

    # Limit dataset size
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_valtest]
    y_test = y_test[:n_valtest]    
        
    # Flatten images
    X_train = X_train.view(-1, 28*28)
    X_test = X_test.view(-1, 28*28)
    
    # Create TensorDatasets
    train_X = X_train.to(device)
    train_y = y_train.to(device)
    test_X = X_test.to(device)
    test_y = y_test.to(device)

    train_dataset = TensorDataset(train_X, train_y)
    valtest_dataset = TensorDataset(test_X, test_y)

    # Equally split validation and test sets
    val_size = int(0.5 * len(valtest_dataset))
    test_size = int(0.5 * len(valtest_dataset))
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim


def load_cifar(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([
        ToTensor(),  # Convert to tensor
        # Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        lambda x: x.view(-1)  # Flatten CIFAR10 images (3 * 32 * 32)
    ])
    data_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    input_dim = 3 * 32 * 32

    X_train = torch.tensor(data_train.data).float() / 255.0  # Normalize pixel values to [0, 1] 
    y_train = torch.tensor(data_train.targets)  # Convert labels to tensor
    X_test = torch.tensor(data_test.data).float() / 255.0  # Normalize pixel values to [0, 1]
    y_test = torch.tensor(data_test.targets)
    
    # Shuffle data
    shuffle_idx = torch.randperm(n_train)
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    shuffle_idx2 = torch.randperm(n_valtest)
    X_test = X_test[shuffle_idx2]
    y_test = y_test[shuffle_idx2]

    # Limit dataset size
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_valtest]
    y_test = y_test[:n_valtest]    
        
    # Flatten images
    X_train = X_train.contiguous().view(X_train.size(0), -1)
    X_test = X_test.contiguous().view(X_test.size(0), -1)
    
    # Create TensorDatasets
    train_X = X_train.to(device)
    train_y = y_train.to(device)
    test_X = X_test.to(device)
    test_y = y_test.to(device)

    train_dataset = TensorDataset(train_X, train_y)
    valtest_dataset = TensorDataset(test_X, test_y)

    # Equally split validation and test sets
    val_size = int(0.5 * len(valtest_dataset))
    test_size = int(0.5 * len(valtest_dataset))
    val_dataset, test_dataset = random_split(valtest_dataset, [val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim



def load_celeba(seed, n_train, n_valtest, device, batch_size):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load dataset with transformation
    transform = Compose([
        ToTensor(),  # Convert to tensor
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
        lambda x: x.view(-1)  # Flatten CIFAR10 images (3 * 32 * 32)
    ])
    full_data = CelebA(root='./data', split='all', download=True, transform=transform)
    input_dim = 3 * 178 * 218

    # Equally split validation and test sets
    subset_size = n_train + n_valtest
    subset_indices = torch.randperm(len(full_data))[:subset_size]  # Randomly select samples
    data_subset = Subset(data_train, subset_indices)
    val_size = int(0.5 * n_valtest)
    test_size = int(0.5 * n_valtest)
    train_dataset, val_dataset, test_dataset = random_split(full_data, [n_train, val_size, test_size])

    # DataLoader parameters
    if batch_size==0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)
    
    return train_loader, val_loader, test_loader, input_dim


def load_eeg(seed, device, batch_size, sampling_freq=1.6):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Load and preprocess the PhysioNet EEG Motor Imagery data
    N_SUBJECT = 50
    IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]

    # Load data from PhysioNet (example assumes data is downloaded locally)
    physionet_paths = [
        mne.datasets.eegbci.load_data(
            subjects=subj_id,
            runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
            path="PhysioNet_EEG",
        ) for subj_id in range(1, N_SUBJECT+1)
    ]
    physionet_paths = np.concatenate(physionet_paths)

    # Ensuring that all subjects share same sampling frequency
    # TARGET_SFREQ = 160  # 160 Hz sampling rate
    TARGET_SFREQ = sampling_freq

    # Concatenate all loaded raw data
    parts = []
    for path in physionet_paths:
        raw = mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel='auto',
            verbose='WARNING',
        )
        # Resample raw data to ensure consistent sfreq
        raw.resample(TARGET_SFREQ, npad="auto")
        parts.append(raw)
        
    # Concatenate resampled raw data
    raw = mne.concatenate_raws(parts)

    # Pick EEG channels and extract events
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
    )
    events, _ = mne.events_from_annotations(raw)

    # Epoch the data
    epoched = mne.Epochs(
        raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
        proj=False, picks=eeg_channel_inds, baseline=None, preload=True
    )

    # Convert data to NumPy arrays
    X = (epoched.get_data() * 1e3).astype(np.float32)  # Convert to millivolts
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 0: left, 1: right

    # Flatten the time and channel dimensions for input to dense neural network
    X_flat = X.reshape(X.shape[0], -1)

    # First split (train, temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_flat, y, test_size=0.3, random_state=seed
    )

    # Compute standardization parameters from training set
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6

    # Standardize datasets using train statistics
    X_train = (X_train - X_mean) / X_std
    X_temp = (X_temp - X_mean) / X_std

    # Split validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )
    
    def MakeTensorDataset(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        tensordataset = TensorDataset(X_tensor, y_tensor)
        return tensordataset
    
    # Create datasets and dataloaders
    train_dataset = MakeTensorDataset(X_train, y_train)
    val_dataset = MakeTensorDataset(X_val, y_val)
    test_dataset = MakeTensorDataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim


def load_eeg_ts(seed, device, batch_size, sampling_freq=16):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # Load and preprocess the PhysioNet EEG Motor Imagery data
    N_SUBJECT = 50
    IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST = [4, 8, 12]

    # Load data from PhysioNet (example assumes data is downloaded locally)
    physionet_paths = [
        mne.datasets.eegbci.load_data(
            subjects=subj_id,
            runs=IMAGINE_OPEN_CLOSE_LEFT_RIGHT_FIST,
            path="PhysioNet_EEG",
        ) for subj_id in range(1, N_SUBJECT+1)
    ]
    physionet_paths = np.concatenate(physionet_paths)

    # Ensuring that all subjects share same sampling frequency
    # TARGET_SFREQ = 160  # 160 Hz sampling rate
    TARGET_SFREQ = sampling_freq

    # Concatenate all loaded raw data
    parts = []
    for path in physionet_paths:
        raw = mne.io.read_raw_edf(
            path,
            preload=True,
            stim_channel='auto',
            verbose='WARNING',
        )
        # Resample raw data to ensure consistent sfreq
        raw.resample(TARGET_SFREQ, npad="auto")
        parts.append(raw)
        
    # Concatenate resampled raw data
    raw = mne.concatenate_raws(parts)

    # Pick EEG channels and extract events
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
    )
    events, _ = mne.events_from_annotations(raw)

    # Epoch the data
    epoched = mne.Epochs(
        raw, events, dict(left=2, right=3), tmin=1, tmax=4.1,
        proj=False, picks=eeg_channel_inds, baseline=None, preload=True
    )

    # Convert data to NumPy arrays
    X = (epoched.get_data() * 1e3).astype(np.float32)  # Convert to millivolts
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 0: left, 1: right
    
    # Train-validation-test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    
    def MakeTensorDataset(X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        tensordataset = TensorDataset(X_tensor, y_tensor)
        return tensordataset
    
    # Create datasets and dataloaders
    train_dataset = MakeTensorDataset(X_train, y_train)
    val_dataset = MakeTensorDataset(X_val, y_val)
    test_dataset = MakeTensorDataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    
    input_dim = X_train.shape
    
    return train_loader, val_loader, test_loader, input_dim
    

def load_binary_classification(seed, n_train, n_valtest, device, batch_size, n_features):
    """
    Generate a binary classification dataset using sklearn.datasets.make_classification 
    and load it in a format compatible with the example MNIST loader.

    Args:
        seed (int): Random seed for reproducibility.
        n_train (int): Number of training samples.
        n_valtest (int): Number of validation + test samples.
        device (torch.device): PyTorch device (CPU/GPU).
        batch_size (int): Batch size for DataLoader.
        n_features (int): Number of features for each sample.

    Returns:
        train_loader, val_loader, test_loader, input_dim: DataLoaders for training, validation, and testing,
                                                         and the number of input features (input_dim).
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate dataset
    X, y = make_classification(
        n_samples=n_train + n_valtest,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=2,
        random_state=seed,
    )

    # Shuffle and split dataset into train, validation, and test sets
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-6)  # Add epsilon to prevent division by zero

    # Split into training and validation/test datasets
    X_train, X_valtest = X[:n_train], X[n_train:]
    y_train, y_valtest = y[:n_train], y[n_train:]

    # Further split validation and test datasets equally
    val_size = test_size = n_valtest // 2
    X_val, X_test = X_valtest[:val_size], X_valtest[val_size:]
    y_val, y_test = y_valtest[:val_size], y_valtest[val_size:]

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    val_dataset = TensorDataset(X_val.to(device), y_val.to(device))
    test_dataset = TensorDataset(X_test.to(device), y_test.to(device))

    # DataLoader parameters
    if batch_size == 0:
        params = {'shuffle': True}
        test_params = {'shuffle': False}
    else:
        params = {'shuffle': True, 'batch_size': batch_size}
        test_params = {'shuffle': False, 'batch_size': batch_size}

    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **test_params)
    test_loader = DataLoader(test_dataset, **test_params)

    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim


def load_data(dataset_name, batch_size, num_workers=0, n_train=50000, n_valtest=10000):
    """
    Generic data loader function that dispatches to specific loaders.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 2025

    if dataset_name == 'cifar10':
        train_loader, val_loader, test_loader, _ = load_cifar(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'coco':
        # Note: This is a placeholder for the actual COCO data loader.
        # The implementation will depend on how the COCO dataset is stored and preprocessed.
        print("Warning: COCO data loader is not fully implemented.")
        # Create dummy data loaders for now.
        dummy_tensor = torch.randn(128, 3, 224, 224)
        dummy_labels = torch.randint(0, 80, (128,))
        dummy_dataset = TensorDataset(dummy_tensor, dummy_labels)
        train_loader = DataLoader(dummy_dataset, batch_size=batch_size)
        val_loader = DataLoader(dummy_dataset, batch_size=batch_size)
        test_loader = DataLoader(dummy_dataset, batch_size=batch_size)
        return train_loader, val_loader, test_loader
    elif dataset_name == 'mnist':
        train_loader, val_loader, test_loader, _ = load_mnist(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'fashion':
        train_loader, val_loader, test_loader, _ = load_fashion(
            seed=seed,
            n_train=n_train,
            n_valtest=n_valtest,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'celeba':
        train_loader, val_loader, test_loader, _ = load_celeba(
            seed=seed,
            n_train=10000,  # Smaller subset for CelebA
            n_valtest=2000,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    elif dataset_name == 'eeg':
        train_loader, val_loader, test_loader, _ = load_eeg(
            seed=seed,
            device=device,
            batch_size=batch_size
        )
        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
