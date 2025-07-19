"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import omegaconf
import torch
from torchvision import transforms
import transformers
from lanistr.utils.parallelism_utils import is_main_process


logger = logging.getLogger(__name__)


class DataLoaderManager:
    """Manages DataLoader creation with consistent patterns."""
    
    def __init__(self, args: omegaconf.DictConfig):
        """Initialize DataLoader manager with configuration.
        
        Args:
            args: Configuration containing batch sizes, worker count, etc.
        """
        self.args = args
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = getattr(args, 'eval_batch_size', args.train_batch_size)
        self.test_batch_size = getattr(args, 'test_batch_size', args.train_batch_size)
        self.workers = args.workers
        self.distributed = args.distributed
    
    def _create_sampler(self, dataset: torch.utils.data.Dataset, 
                       shuffle: bool = True, drop_last: bool = True) -> Optional[torch.utils.data.Sampler]:
        """Create appropriate sampler based on distributed settings.
        
        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            Sampler if distributed, None otherwise
        """
        if self.distributed:
            return torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle, drop_last=drop_last
            )
        return None
    
    def _create_dataloader(self, dataset: torch.utils.data.Dataset, 
                          batch_size: int, shuffle: bool = True, 
                          drop_last: bool = True) -> torch.utils.data.DataLoader:
        """Create a DataLoader with consistent configuration.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle data (ignored if sampler is used)
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            Configured DataLoader
        """
        sampler = self._create_sampler(dataset, shuffle, drop_last)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            num_workers=self.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=drop_last,
        )
    
    def _log_dataset_info(self, phase: str, dataset: torch.utils.data.Dataset) -> None:
        """Log dataset information for a specific phase.
        
        Args:
            phase: Phase name (train, test, valid)
            dataset: Dataset to log info for
        """
        if is_main_process():
            print(f'Number of {phase:<10} examples: {len(dataset)}')
            logger.info('Number of %s examples: %d', phase, len(dataset))
    
    def generate_pretrain_loaders(self, dataset: Dict[str, torch.utils.data.Dataset]) -> Dict[str, torch.utils.data.DataLoader]:
        """Generate DataLoaders for pretraining task.
        
        Args:
            dataset: Dictionary containing train dataset
            
        Returns:
            Dictionary of DataLoaders
        """
        trainset = dataset['train']
        self._log_dataset_info('training', trainset)
        
        train_dataloader = self._create_dataloader(
            trainset, 
            self.train_batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        return {'train': train_dataloader}
    
    def generate_finetune_loaders(self, dataset: Dict[str, torch.utils.data.Dataset]) -> Dict[str, torch.utils.data.DataLoader]:
        """Generate DataLoaders for finetuning task.
        
        Args:
            dataset: Dictionary containing train, test, and valid datasets
            
        Returns:
            Dictionary of DataLoaders
        """
        trainset = dataset['train']
        testset = dataset['test']
        valset = dataset['valid']
        
        # Log dataset information
        self._log_dataset_info('training', trainset)
        self._log_dataset_info('test', testset)
        
        # Create DataLoaders
        train_dataloader = self._create_dataloader(
            trainset, 
            self.train_batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        valid_dataloader = self._create_dataloader(
            valset, 
            self.eval_batch_size, 
            shuffle=False, 
            drop_last=False
        )
        
        # Test sampler is typically None for final evaluation
        test_sampler = None if not self.distributed else self._create_sampler(testset, shuffle=False, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=True,
        )
        
        return {
            'train': train_dataloader,
            'valid': valid_dataloader,
            'test': test_dataloader,
        }
    
    def generate_loaders(self, dataset: Dict[str, torch.utils.data.Dataset]) -> Dict[str, torch.utils.data.DataLoader]:
        """Generate appropriate DataLoaders based on task type.
        
        Args:
            dataset: Dictionary containing datasets
            
        Returns:
            Dictionary of DataLoaders
        """
        if self.args.task == 'pretrain':
            return self.generate_pretrain_loaders(dataset)
        elif self.args.task == 'finetune':
            return self.generate_finetune_loaders(dataset)
        else:
            raise ValueError(f"Unknown task type: {self.args.task}")


class ImageTransformManager:
    """Manages image transformations for training and testing."""
    
    def __init__(self, args: omegaconf.DictConfig):
        """Initialize transform manager with configuration.
        
        Args:
            args: Configuration containing image parameters
        """
        self.args = args
        self.image_size = args.image_size
        self.image_crop = getattr(args, 'image_crop', args.image_size)
        self.image_encoder_name = args.image_encoder_name
    
    def get_image_processor(self) -> transformers.ViTImageProcessor:
        """Get the image processor for the specified encoder.
        
        Returns:
            ViT Image Processor
        """
        return transformers.ViTImageProcessor.from_pretrained(self.image_encoder_name)
    
    def create_train_transforms(self, image_processor: transformers.ViTImageProcessor) -> transforms.Compose:
        """Create training image transforms.
        
        Args:
            image_processor: Image processor for normalization parameters
            
        Returns:
            Composed training transforms
        """
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size, scale=(0.67, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_processor.image_mean, 
                std=image_processor.image_std
            ),
        ])
    
    def create_test_transforms(self, image_processor: transformers.ViTImageProcessor) -> transforms.Compose:
        """Create test image transforms.
        
        Args:
            image_processor: Image processor for normalization parameters
            
        Returns:
            Composed test transforms
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_crop),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_processor.image_mean, 
                std=image_processor.image_std
            ),
        ])
    
    def get_image_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get the image transforms for training and testing.
        
        Returns:
            Tuple of (train_transforms, test_transforms)
        """
        image_processor = self.get_image_processor()
        train_transforms = self.create_train_transforms(image_processor)
        test_transforms = self.create_test_transforms(image_processor)
        
        return train_transforms, test_transforms


class MaskGenerator:
    """A class to generate boolean masks for the pretraining task.

    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is
    either 0 or 1, where 1 indicates "masked".
    """

    def __init__(
        self,
        input_size: int = 192,
        mask_patch_size: int = 32,
        model_patch_size: int = 4,
        mask_ratio: float = 0.6,
    ):
        """Initialize the MaskGenerator.

        Args:
            input_size: the size of the input image
            mask_patch_size: the size of the mask patch
            model_patch_size: the size of the model patch
            mask_ratio: the ratio of the mask patch to the model patch
        """
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        self._validate_parameters()
        self._calculate_derived_parameters()

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.input_size % self.mask_patch_size != 0:
            raise ValueError('Input size must be divisible by mask patch size')
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError('Mask patch size must be divisible by model patch size')

    def _calculate_derived_parameters(self) -> None:
        """Calculate derived parameters for mask generation."""
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self) -> torch.Tensor:
        """Generate a random boolean mask.
        
        Returns:
            Boolean tensor representing the mask
        """
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return torch.tensor(mask.flatten(), dtype=torch.bool)

    def get_mask_info(self) -> Dict[str, Any]:
        """Get information about the mask configuration.
        
        Returns:
            Dictionary containing mask configuration info
        """
        return {
            'input_size': self.input_size,
            'mask_patch_size': self.mask_patch_size,
            'model_patch_size': self.model_patch_size,
            'mask_ratio': self.mask_ratio,
            'token_count': self.token_count,
            'mask_count': self.mask_count,
            'rand_size': self.rand_size,
            'scale': self.scale
        }


# Backward compatibility functions
def generate_loaders(args: omegaconf.DictConfig, dataset: torch.utils.data.Dataset) -> Dict[str, torch.utils.data.DataLoader]:
    """Generate the data loaders for the given dataset.

    Args:
        args: the arguments for the experiment
        dataset: the dataset to load

    Returns:
        A dictionary of data loaders
    """
    loader_manager = DataLoaderManager(args)
    return loader_manager.generate_loaders(dataset)


def get_image_transforms(args: omegaconf.DictConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get the image transforms for the given arguments.

    Args:
        args: the arguments for the experiment

    Returns:
        A tuple of the train and test transforms
    """
    transform_manager = ImageTransformManager(args)
    return transform_manager.get_image_transforms()
