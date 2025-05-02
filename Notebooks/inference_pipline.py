from src.lightning_modules import LitModel
from tqdm import tqdm
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
import subprocess
import os
def load_model(checkpoint, device):
    """ Loads a model from a checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint.

    Returns
    -------
    model : lightning_modules.LitModel
        The model.
    """    
    model = LitModel.load_from_checkpoint(checkpoint).eval()
    return model.to(device)

def get_inference(x, device, model=None, checkpoint_path=None):
    ''' Only run inference with a given model
    
    Parameters
    
    x: torch.tensor.
        Shape should be (batch_size, revisits, bands, 50, 50)
    device: torch device
        cpu or gpu
    model: Model to run for inference. This OR checkpoint path.
    
    checkpoint_path: Path to load model from. Has precendence over model.
    '''
    if (not model) and (not checkpoint_path):
        print("Include model object or path to checkpoint")
    elif checkpoint_path:
        model = load_model(checkpoint_path, device)
    y_hat = model(x)
    return y_hat

def load_tif(path, device='cpu'):
    '''
    Load tiff from path into tensor
    '''
    with rasterio.open(path) as w:
        img = w.read()
    return torch.tensor(img).to(device)

def load_multi_tif(paths, device='cpu'):
    revisits = []
    for path in paths:
        with rasterio.open(path) as w:
            revisits.append(w.read())
    revisits = np.array(revisits)
    return torch.tensor(revisits).to(device)

def load_multi_tif_partial(paths, start_x, start_y, end_x, end_y, device="cpu"):
    window = rasterio.windows.Window(start_x,start_y,end_x-start_x,end_y-start_y)
    revisits = []
    for path in paths:
        with rasterio.open(path) as w:
            revisits.append(w.read(window=window))
    revisits = np.array(revisits)
    return torch.tensor(revisits).to(device)

def normalize(img, dist=None):
    '''
    img is the image to normalize <br>
    dist is a tuple (mu,std), or none if using per-chip norm
    '''
    
    
    norm = img.clone()
    # Set nan values to 0
    norm[img.isnan()] = 0
    norm = norm[:,:12]

    # Normalize along dims
    if not norm:
        mean = norm.mean(dim=(0,2,3), keepdim=True)
        std = norm.std(dim=(0,2,3), keepdim=True)
        
        b_mean = norm.mean(dim=(2,3), keepdim=True)
        b_std = norm.std(dim=(2,3), keepdim=True)
        norm = (norm - b_mean) / (b_std + 1e-8)
            
    else:
        norm = (norm - dist[0]) / (dist[1] + 1e-8)
        mean = dist[0]
        std = dist[1]
    
    return norm, mean, std

def chip_from_tensor(image=torch.Tensor, overlap=0, chip_size=26, dist=None, device="cpu"):
    ''' Split tensor into chips with associated indices
    
    Parameters <br>
    image: Tensor containing whole image <br>
    overlap: Overlap in original image. <br>
    
    Returns:<br>
    chip objects, original indices, (mean,std), canvas_shape
    '''
    
    height = image.shape[2]
    width = image.shape[3]
    channels = image.shape[1]
    revisits = image.shape[0]
    gap = chip_size - overlap
    
    img = torch.zeros((revisits,
                       channels,
                       height + gap * (height % gap != 0) - height%gap,
                       width + gap * (width % gap != 0) - width%gap))
    img[:,:,:height,:width] = image[:,:,:,:]
    
    img, mu, std = normalize(img, dist)

    indices = []
    chips = []
    for row in range(0,height-chip_size+1,gap):
        for col in range(0, width-chip_size+1, gap):
            indices.append((row,col))
            chip = img[:,:,row:row+chip_size, col:col+chip_size]
            chips.append(chip)
    
    return chips, indices, (mu, std), img.shape

def loader_from_chips(chips, indices, batch_size, device='cpu'):
    data = InferenceDataset(chips, indices)
    loader = DataLoader(data, batch_size, shuffle=False)
    return loader

def infer_from_loader(loader, canvas_size, overlap, model, std, mu, chip_size=26):
    ###
    height = (canvas_size[2] // chip_size) * 156
    width = (canvas_size[3] // chip_size) * 156
    channels = canvas_size[1]
    gap = chip_size - overlap
    
    preds = []
    r = []
    c = []
    for chips, idxs in loader:
        preds.append(model(chips))
        r.append(idxs[0])
        c.append(idxs[1])
    
    preds = torch.concat(preds, dim=0)
    r = (torch.concat(r) // gap) * (156 // (chip_size // gap))
    c = (torch.concat(c) // gap) * (156 // (chip_size // gap))

    canvas = torch.zeros(3, height, width)
    overlaps = torch.zeros(3, height, width)
    for i in range(len(preds)):
        canvas[:,r[i]:r[i] + 156, c[i]:c[i]+156] += ((preds[i,0] * (std[0,1:4] + 1e-8)) + mu[0,1:4])
        overlaps[:,r[i]:r[i] + 156, c[i]:c[i]+156] += 1
        
    return canvas / overlaps

def do_inference_from_path(revist_paths,
                           model,
                           batch_size=8,
                           chip_size=26,
                           overlap=0,
                           partial=[],
                           device="cpu",
                           verbose=True,
                           dist=None,
                           exclude8=False):
    if len(partial) == 4:
        if verbose: print("Getting partial Tiff")
        img = load_multi_tif_partial(revist_paths,
                                     partial[0],
                                     partial[1],
                                     partial[2],
                                     partial[3],
                                     device = device)
    else:
        if verbose: print("Getting full Tiff")
        img = load_multi_tif(revist_paths, device=device)
    
    if exclude8:
        if verbose: print("Excluding 8b")
        img = img[:,[0,1,2,3,4,5,6,7,9,10,11,12]]
    if verbose: print(f"Original shape: {img.shape}")
    gc.collect()
    chips, indices, (mu, std), shape = chip_from_tensor(img,
                                                        overlap=overlap,
                                                        chip_size=chip_size,
                                                        dist=dist,
                                                        device=device)
    if verbose: print(f"Chip Shape: {chips[0].shape}\nChip Count: {len(chips)}")
    loader = loader_from_chips(chips, indices, batch_size)
    gc.collect()
    predicted = infer_from_loader(loader, shape, overlap, model, std, mu, chip_size=chip_size)
    return predicted

def full_inference_to_chips(revist_paths,
                           model_path,
                           chip_save_path,
                           batch_size=8,
                           chip_size=26,
                           overlap=0,
                           device="cuda",
                           chip_norm="global",
                           verbose=True):
    
    # Start by initializing global mean and std
    with rasterio.open(revist_paths[0]) as r:
        meta = r.profile
    
    mus = []
    stds = []
    for path in revist_paths:
        tif = load_tif(path, device=device).to(torch.float32)
        mus.append(tif.mean(dim=(1,2),keepdim=True))
        stds.append(tif.std(dim=(1,2),keepdim=True))
        del tif
        gc.collect()
        torch.cuda.empty_cache()
    mu = torch.stack(mus)
    std = torch.stack(std)
    
    width = meta["width"]
    height = meta["height"]

    meta["height"] = 1248
    meta["width"] = 1248
    meta["count"] = 3
    transform = rasterio.Affine(meta["transform"][0] * (chip_size/156),
                    meta["transform"][1],
                    meta["transform"][2],
                    meta["transform"][3],
                    meta["transform"][4] * (chip_size/156),
                    meta["transform"][5])
    meta["transform"] = transform
    
    if chip_norm == "global":
        dist = (mu,std)
    else:
        dist = None
    
    model = load_model(model_path, device=device)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        for x in tqdm(range(0,width-208,208)):
            for y in tqdm(range(0,height-208,208)):
                partial = [x,y,x+208,y+208]
                infer = do_inference_from_path(revist_paths,
                                                model,
                                                batch_size=batch_size,
                                                chip_size=chip_size,
                                                overlap=overlap,
                                                partial=partial,
                                                dist=dist,
                                                verbose=False)
                gc.collect()
                torch.cuda.empty_cache()
                infer /= 10
                
                temp_path = f"{chip_save_path}_x{x}_y{y}_temp.tif"
                with rasterio.open(temp_path, 'w', **meta) as w:
                    w.write(infer.round().to(torch.int32).cpu().detach().numpy())

                final_path = f"{chip_save_path}x{x}_y{y}.tif"
                try:
                    subprocess.run([
                        "rio", "cogeo", "create",
                        temp_path, final_path,
                        "--co", "BIGTIFF=IF_SAFER",
                        "--allow-intermediate-compression",  # Reduces temp file size
                        "--no-in-memory",  # Force disk-based processing
                        "--threads", "2",  # Limit parallel threads
                        "--overview-level", "5"
                    ], check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error creating COG for x={x}, y={y}:")
                    print(f"Return code: {e.returncode}")
                    print(f"Command: {e.cmd}")
                    print(f"Stdout: {e.stdout}")
                    print(f"Stderr: {e.stderr}")
                    raise

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                del infer
                gc.collect()
                torch.cuda.empty_cache()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return mu,std
    

class InferenceDataset(Dataset):
    def __init__(self, chips, indices):
        super().__init__()
        
        self.chips = chips
        self.indices = indices
    
    def __getitem__(self, index):
        return self.chips[index], self.indices[index]
    
    def __len__(self):
        return len(self.chips)
    
    
#### Stuff from SRC to avoid annoying imports ####
# Code from worldstrat

