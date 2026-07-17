import os
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE, ADASYN

# Force synchronous CUDA kernel launches to make debugging easier and trace errors more clearly.
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Dataset wrapper for meme samples. It can load either precomputed CLIP embeddings or raw images/text.
class MemesDataset(Dataset):
    def __init__(self, root_folder, dataset, split='train', image_size=224, fast=True,
                 use_smote=False, smote_strategy='auto', clip_model_name='ViT-L/14'):
        super(MemesDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset = dataset
        self.split = split

        self.image_size = image_size
        self.fast = fast
        self.use_smote = use_smote
        self.smote_strategy = smote_strategy
        self.clip_model_name = clip_model_name
        self.info_file = os.path.join(root_folder, dataset, f'labels/{dataset}_info.csv')
        
        # Read the metadata table using UTF-8 first and fall back to Latin-1 if needed.
        try:
            self.df = pd.read_csv(self.info_file, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.info_file, encoding='latin-1')
        self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)
        float_cols = self.df.select_dtypes(float).columns
        self.df[float_cols] = self.df[float_cols].fillna(-1).astype('Int64')

        if self.fast:
            # In fast mode, load precomputed CLIP embeddings instead of processing raw images.
            # These embeddings are also used by SMOTE to rebalance the training set in feature space.
            emb_suffix = "_mclip" if self.clip_model_name == "m-CLIP" else ""
            emb_filename = f"{split}_no-proj_output{emb_suffix}.pt"
            emb_path = f"{self.root_folder}/{self.dataset}/clip_embds/{emb_filename}"

            if not os.path.exists(emb_path):
                raise FileNotFoundError(
                    f"Embedding file not found: {emb_path}\n"
                    f"Run createClipEmbedding.py --dataset {self.dataset} --clip_model {self.clip_model_name} first."
                )

            print(f"[dataset] Loading embeddings: {emb_filename}")
            self.embds = torch.load(emb_path)
            self.embdsDF = pd.DataFrame(self.embds)

            assert len(self.embds) == len(self.df), (
                f"Embeddings ({len(self.embds)}) and dataframe ({len(self.df)}) size mismatch"
            )

            # Apply SMOTE only on the training split when requested.
            self.original_len = len(self.df)
            if self.use_smote and self.split == 'train':
                self._apply_smote_on_embeddings()

    # Oversample minority samples by generating synthetic CLIP embeddings in the feature space.
    def _apply_smote_on_embeddings(self):
        print(f"Applying SMOTE on CLIP embeddings for '{self.split}' split...")

        labels = self.df['label'].values
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")

        # 1. Use concatenated image-text CLIP embeddings as the SMOTE feature vector.
        # The final feature matrix is shaped as (N, 1536) for the standard CLIP backbone.
        n_samples = len(self.df)
        image_embs = []
        text_embs  = []

        for i in range(n_samples):
            row     = self.df.iloc[i]
            # Search for the embedding corresponding to the current row's idx_meme
            matches = self.embdsDF.loc[self.embdsDF['idx_meme'] == row['id']]
            idx     = matches.index[0] if len(matches) > 0 else 0
            embd    = self.embds[idx]
            image_embs.append(embd['image'].squeeze(0).float().numpy())
            text_embs.append(embd['text'].squeeze(0).float().numpy())

        image_matrix = np.stack(image_embs)   # (N, 768)
        text_matrix  = np.stack(text_embs)    # (N, 768)
        feature_matrix = np.concatenate([image_matrix, text_matrix], axis=1)  # (N, 1536)

        # 2. Choose the nearest-neighbor count for SMOTE based on the minority class size.
        k = min(5, int(min(counts)) - 1)
        if k < 1:
            print("SMOTE skipped: not enough minority samples (need at least 2).")
            return

        try:
            if self.smote_strategy == 'adasyn':
                sampler = ADASYN(random_state=42, n_neighbors=k)
            elif self.smote_strategy == 'auto':
                sampler = SMOTE(random_state=42, k_neighbors=k)
            else:
                sampler = SMOTE(random_state=42, sampling_strategy=self.smote_strategy,
                                k_neighbors=k)

            features_resampled, labels_resampled = sampler.fit_resample(feature_matrix, labels)
        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data.")
            return

        n_synthetic = len(labels_resampled) - n_samples
        print(f"SMOTE generated {n_synthetic} synthetic samples")
        print(f"New class distribution: {dict(zip(*np.unique(labels_resampled, return_counts=True)))}")

        if n_synthetic == 0:
            return

        # 3. Split the synthetic feature vector back into image and text embedding parts.
        # Then append the synthetic samples to the dataset so they can be used during training.
        img_txt_dim = 1024 if self.clip_model_name == 'm-CLIP' else 768
        synth_features = features_resampled[n_samples:]
        synth_img_embs = synth_features[:, :img_txt_dim]
        synth_txt_embs = synth_features[:, img_txt_dim:]
        synth_labels   = labels_resampled[n_samples:]

        # 4. Add synthetic entries into both the embedding cache and metadata dataframe.
        # A donor sample from the same class is used to copy metadata such as text and image-related fields.
        original_indices = np.arange(n_samples)
        synthetic_embd_entries = []
        synthetic_df_rows      = []

        for i in range(n_synthetic):
            lbl = synth_labels[i]

            # Choose a random donor from the original samples of the same class to copy metadata
            donor_idx = int(np.random.choice(original_indices[labels == lbl]))
            donor_row = self.df.iloc[donor_idx].copy()

            # Negative ID for synthetic samples to avoid clashes with original IDs
            synthetic_id = -(i + 1)
            donor_row['id'] = synthetic_id

            # Make new embedding entry with tensors of the SMOTE-interpolated embeddings
            synthetic_embd_entries.append({
                'idx_meme': synthetic_id,
                'image': torch.tensor(synth_img_embs[i], dtype=torch.float32).unsqueeze(0),
                'text':  torch.tensor(synth_txt_embs[i], dtype=torch.float32).unsqueeze(0),
            })
            synthetic_df_rows.append(donor_row)

        # Append synthetic rows to the main dataset structure.
        self.embds   = self.embds + synthetic_embd_entries
        synthetic_df = pd.DataFrame(synthetic_df_rows)
        self.df      = pd.concat([self.df, synthetic_df], ignore_index=True)

        # Rebuild embdsDF so that lookups in __getitem__ remain true
        self.embdsDF = pd.DataFrame(self.embds)

        print(f"Dataset size after SMOTE: {len(self.df)}")
        assert len(self.embds) == len(self.df), "Post-SMOTE size mismatch between embds and df!"


    # Return the number of samples in the current split.
    def __len__(self):
        return len(self.df)

    # Return one sample as a dictionary containing image/text inputs, label, and metadata.
    def __getitem__(self, idx):
        # Take dataframe row with index
        row = self.df.iloc[idx]
        label = row['label']

        # Determine image naming for both original and synthetic SMOTE entries.
        if row['id'] >= 0:  # Original data have positive ID
            if self.dataset == 'hmc':
                image_name = row['img'].split('/')[1]
            else:
                image_name = row['image']
        else:
            image_name = f"smote_synthetic_{abs(row['id'])}" # Synthetics data SMOTE have negative ID

        txt = 'null' if row['text'] == 'nothing' else row['text']

        # In fast mode, return precomputed embeddings directly.
        # Otherwise, load raw images and text from the dataset folder.
        if self.fast:
            # Search for the embedding corresponding to the current row's idx_meme
            matches = self.embdsDF.loc[self.embdsDF['idx_meme'] == row['id']].index
            if len(matches) == 0:
                # Fallback make sure models don't break if idx_meme not found in embdsDF
                embd_row = self.embds[0]
                print(f"Warning: idx_meme {row['id']} not found in embdsDF, using index 0")
            else:
                embd_row = self.embds[matches[0]]

            # Use CLIP pre-calculated embeddings as image and text inputs
            image = embd_row['image']
            text = embd_row['text']

        else:
            # Use raw image and text inputs
            if self.dataset == 'hmc':
                image_fn = row['img'].split('/')[1]
            else:
                image_fn = row['image']
            image = Image.open(f"{self.root_folder}/{self.dataset}/img/{image_fn}").convert('RGB')\
                .resize((self.image_size, self.image_size))
            text = txt

        item = {
            'image_name': image_name,
            'image': image,
            'text': text,
            'label': row['label'],
            'idx_meme': row['id'],
            'origin_text': txt
        }

        return item

# Collate a batch into tensors suitable for the model input pipeline.
class MemesCollator(object):
    def __init__(self, args):
        self.args = args
        self.clip_model_name = getattr(args, 'clip_model', 'ViT-L/14')

        # In slow mode, load the CLIP image processor and optional tokenizer for text encoding.
        if not args.fast_process:
            _, self.clip_preprocess = clip.load("ViT-L/14", device="cuda", jit=False)

            # For m-CLIP slow path: load HF tokenizer for text encoding
            # Image preprocessing stays identical — same ViT-L/14 image encoder
            if self.clip_model_name == 'm-CLIP':
                try:
                    from transformers import AutoTokenizer
                    self.mclip_tokenizer = AutoTokenizer.from_pretrained(
                        "M-CLIP/XLM-Roberta-Large-Vit-L-14"
                    )
                    print("[collator] m-CLIP tokenizer loaded for slow-path text encoding.")
                except ImportError:
                    raise ImportError(
                        "transformers is required for --clip_model m-CLIP slow path.\n"
                        "Install with: pip install transformers"
                    )
            else:
                self.mclip_tokenizer = None

    # Convert a list of dataset samples into one batched dictionary that the model can consume.
    def __call__(self, batch):
        labels = torch.LongTensor([item['label'] for item in batch])
        idx_memes = torch.LongTensor([item['idx_meme'] for item in batch])

        # Build enhanced text prompts by combining a simple visual prompt with the original meme text.
        text_input = []
        for el in batch:
            text_input.append(clip.tokenize(f'{"a photo of $"} , {el["origin_text"]}', context_length=77,
                                            truncate=True))

        enh_texts = torch.cat([item for item in text_input], dim=0)

        # Repeat a simple base prompt for each sample so the model can compare original and enhanced text inputs.
        simple_prompt = clip.tokenize('a photo of $', context_length=77).repeat(labels.shape[0], 1)

        image_names = [item['image_name'] for item in batch]

        batch_new = {
                     'image_names': image_names,
                     'labels': labels,
                     'idx_memes': idx_memes,
                     'enhanced_texts': enh_texts,
                     'simple_prompt': simple_prompt
                     }

        # Fast path: the batch already contains CLIP image/text embeddings.
        if self.args.fast_process:
            images_emb = torch.cat([item['image'] for item in batch], dim=0)
            texts_emb = torch.cat([item['text'] for item in batch], dim=0)

            batch_new['images'] = images_emb
            batch_new['texts'] = texts_emb

        # Slow path: preprocess raw images and tokenize text on the fly.
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

# Small factory function that instantiates the dataset object for a given split.
def load_dataset(args, split):
    dataset = MemesDataset(root_folder=f'./resources/datasets', dataset=args.dataset, split=split, image_size=args.image_size, 
                           fast=args.fast_process, use_smote=getattr(args, 'use_smote', False), 
                           smote_strategy=getattr(args, 'smote_strategy', 'auto'), clip_model_name=getattr(args, 'clip_model', 'ViT-L/14'))

    return dataset