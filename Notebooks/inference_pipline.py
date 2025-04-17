from src.lightning_modules import LitModel
import torch
from src.datasources import S2_ALL_12BANDS
from src.modules import BicubicUpscaledBaseline
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

def load_tif(path):
    '''
    Load tiff from path into tensor
    '''
    with rasterio.open(path) as w:
        img = w.read()
    return torch.tensor(img)

def load_multi_tif(paths):
    revisits = []
    for path in paths:
        with rasterio.open(path) as w:
            revisits.append(w.read())
    revisits = np.array(revisits)
    return torch.tensor(revisits)

def load_multi_tif_partial(paths, start_x, start_y, end_x, end_y):
    window = rasterio.windows.Window(start_x,start_y,end_x-start_x,end_y-start_y)
    revisits = []
    for path in paths:
        with rasterio.open(path) as w:
            revisits.append(w.read(window=window))
    revisits = np.array(revisits)
    return torch.tensor(revisits)

def normalize(img):
    norm = img.clone()
    # Set nan values to 0
    norm[img.isnan()] = 0
    
    # Normalize along dims
    mean = norm.mean(dim=(0,2,3), keepdim=True)
    std = norm.std(dim=(0,2,3), keepdim=True)
    norm = (norm - mean) / (std + 1e-8)
    
    norm = norm[:,:12]
    
    return norm, mean, std

def chip_from_tensor(image=torch.Tensor, overlap=0, chip_size=26):
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
    
    img, mu, std = normalize(img)

    indices = []
    chips = []
    for row in range(0,height-chip_size+1,gap):
        for col in range(0, width-chip_size+1, gap):
            indices.append((row,col))
            chip = img[:,:,row:row+chip_size, col:col+chip_size]
            chips.append(chip)
    
    return chips, indices, (mu, std), img.shape

def loader_from_chips(chips, indices, batch_size):
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
        canvas[:,r[i]:r[i] + 156, c[i]:c[i]+156] += ((preds[i,0] * (std[:,:3] + 1e-8)) + mu[:,:3])[0]
        overlaps[:,r[i]:r[i] + 156, c[i]:c[i]+156] += 1
        
    return canvas / overlaps

def do_inference_from_path(revist_paths,
                           model_checkpoint,
                           batch_size=8,
                           chip_size=26,
                           overlap=0,
                           partial=[],
                           device="cpu",
                           verbose=True,
                           exclude8=False):
    if len(partial) == 4:
        if verbose: print("Getting partial Tiff")
        img = load_multi_tif_partial(revist_paths,
                                     partial[0],
                                     partial[1],
                                     partial[2],
                                     partial[3])
    else:
        if verbose: print("Getting full Tiff")
        img = load_multi_tif(revist_paths)
    
    if exclude8:
        if verbose: print("Excluding 8b")
        img = img[:,[0,1,2,3,4,5,6,7,9,10,11,12]]
    if verbose: print(f"Original shape: {img.shape}")
    chips, indices, (mu, std), shape = chip_from_tensor(img,
                                                        overlap=overlap,
                                                        chip_size=chip_size,)
    if verbose: print(f"Chip Shape: {chips[0].shape}\nChip Count: {len(chips)}")
    loader = loader_from_chips(chips, indices, batch_size)
    model = load_model(model_checkpoint,device)
    predicted = infer_from_loader(loader, shape, overlap, model, std, mu, chip_size=chip_size)
    return predicted
    
class InferenceDataset(Dataset):
    def __init__(self, chips, indices):
        super().__init__()
        
        self.chips = chips
        self.indices = indices
    
    def __getitem__(self, index):
        return self.chips[index], self.indices[index]
    
    def __len__(self):
        return len(self.chips)