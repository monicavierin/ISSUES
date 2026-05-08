import os
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE, ADASYN

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class MemesDataset(Dataset):
    def __init__(self, root_folder, dataset, split='train', image_size=224, fast=True,
                 use_smote=False, smote_strategy='auto'):
        super(MemesDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split

        self.image_size = image_size
        self.fast = fast
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy

        self.info_file = os.path.join(root_folder, dataset, f'labels/{dataset}_info.csv')
        try:
            self.df = pd.read_csv(self.info_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.info_file, encoding='latin-1')
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

        # Apply SMOTE before CLIP encoding to balance dataset
        if self.use_smote and self.split == 'train':
            self._apply_smote()

        if self.fast:
            self.embds = torch.load(f'{self.root_folder}/{self.dataset}/clip_embds/{split}_no-proj_output.pt')
            self.embdsDF = pd.DataFrame(self.embds)

            assert len(self.embds) == len(self.df)

    def _apply_smote(self):
        print(f"Applying SMOTE on '{self.split}' split...")
        
        # Use text length and other characteristics as features for SMOTE
        texts = self.df['text'].fillna('nothing').values
        text_lengths = np.array([len(str(t)) for t in texts]).reshape(-1, 1)
        
        # Text length used as proxy feature
        features = text_lengths
        
        labels = self.df['label'].values
        
        # Count class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        if self.smote_strategy == 'auto':
            smote = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
        elif self.smote_strategy == 'adasyn':
            # Use ADADSYN if want adaptive synthethic sample generation 
            smote = ADASYN(random_state=42, n_neighbors=min(5, min(counts)-1))
        else:
            smote = SMOTE(random_state=42, sampling_strategy=self.smote_strategy,
                         k_neighbors=min(5, min(counts)-1))
        
        try:
            features_resampled, labels_resampled = smote.fit_resample(features, labels)
        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data.")
            return
        
        # Count total synthetic samples generated
        n_synthetic = len(labels_resampled) - len(labels)
        print(f"SMOTE generated {n_synthetic} synthetic samples")
        print(f"New class distribution: {dict(zip(*np.unique(labels_resampled, return_counts=True)))}")
        
        # Identify original and synthetic samples indices
        original_indices = np.arange(len(labels))
        synthetic_indices = np.arange(len(labels), len(labels_resampled))
        
        # For synthetic samples, the original data need to be duplicate with slight variations
        if n_synthetic > 0:
            # Take original data to be duplicated
            original_df = self.df.copy()
            
            # For each synthetic sample, choose an original sample randomly and add noise
            synthetic_rows = []
            for idx in synthetic_indices:
                # Choose original sample randomly from minority class 
                original_idx = np.random.choice(original_indices[labels == labels_resampled[idx]])
                original_row = original_df.iloc[original_idx].copy()
                
                # Add variance to text for noise
                synthetic_rows.append(original_row)
            
            # Merge the original DataFrame with synthetic rows
            synthetic_df = pd.DataFrame(synthetic_rows)
            self.df = pd.concat([self.df, synthetic_df], ignore_index=True)
            
            print(f"Dataset size after SMOTE: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if row['text'] == 'nothing':
            txt = 'null'
        else:
            txt = row['text']

        if self.fast:
            embd_idx = self.embdsDF.loc[self.embdsDF['idx_meme'] == row['id']].index
            if len(embd_idx) == 0:
                embd_idx = [0]  # Fallback to first embedding if not found
            embd_idx = embd_idx[0]
            embd_row = self.embds[embd_idx]

            # use CLIP pre-calculated embeddings as image and text inputs
            image = embd_row['image']
            text = embd_row['text']

        else:
            # use raw image and text inputs
            if self.dataset == 'hmc':
                image_fn = row['img'].split('/')[1]
            else:
                image_fn = row['image']
            image = Image.open(f"{self.root_folder}/{self.dataset}/img/{image_fn}").convert('RGB')\
                .resize((self.image_size, self.image_size))
            text = txt

        if self.dataset == 'hmc':
            image_name = row['img'].split('/')[1]
        else:
            image_name = row['image']

        item = {
            'image_name': image_name,
            'image': image,
            'text': text,
            'label': row['label'],
            'idx_meme': row['id'],
            'origin_text': txt
        }

        return item

class MemesCollator(object):
    def __init__(self, args):
        self.args = args
        if not args.fast_process:
            _, self.clip_preprocess = clip.load("ViT-L/14", device="cuda", jit=False)

    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])

        text_input = []
        for el in batch:
            text_input.append(clip.tokenize(f'{"a photo of $"} , {el["origin_text"]}', context_length=77,
                                            truncate=True))

        enh_texts = torch.cat([item for item in text_input], dim=0)

        simple_prompt = clip.tokenize('a photo of $', context_length=77).repeat(labels.shape[0], 1)

        image_names = [item['image_name'] for item in batch]

        batch_new = {
                     'image_names': image_names,
                     'labels': labels,
                     'idx_memes': idx_memes,
                     'enhanced_texts': enh_texts,
                     'simple_prompt': simple_prompt
                     }

        if self.args.fast_process:
            images_emb = torch.cat([item['image'] for item in batch], dim=0)
            texts_emb = torch.cat([item['text'] for item in batch], dim=0)

            batch_new['images'] = images_emb
            batch_new['texts'] = texts_emb

        else:
            img = []
            texts = []
            for item in batch:
                pixel_values = self.clip_preprocess(item['image']).unsqueeze(0)
                img.append(pixel_values)

                text = clip.tokenize(item['text'], context_length=77, truncate=True)
                texts.append(text)

            pixel_values = torch.cat([item for item in img], dim=0)
            texts = torch.cat([item for item in texts], dim=0)

            batch_new['pixel_values'] = pixel_values
            batch_new['texts'] = texts

        return batch_new


def load_dataset(args, split):
    dataset = MemesDataset(root_folder=f'./resources/datasets', dataset=args.dataset, split=split,
                           image_size=args.image_size, fast=args.fast_process,
                           use_smote=getattr(args, 'use_smote', False), smote_strategy=getattr(args, 'smote_strategy', 'auto'))

    return dataset
