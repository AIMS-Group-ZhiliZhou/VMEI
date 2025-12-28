import numpy as np
import torch
from model import Glow
import matplotlib.pyplot as plt
import cv2
from PIL import Image 
from torchvision import transforms
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import os
import vector_chao_map 
import time
import random
import DC_AE     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

def split_z(z_seq, n_channel, input_size, n_block): 

    z_sample = []

    if len(z_seq.shape) == 1:
        z_seq = z_seq.reshape(((1,) + z_seq.shape))
    n_sample = len(z_seq)
    z_seq = z_seq

    z_shapes = calc_z_shapes(n_channel, input_size, n_block)
    start = 0
    for single_shape in z_shapes:
        length = 1
        for _ in single_shape:
            length *= _
        temp = z_seq[:, start:start + length]
        temp = temp.reshape((n_sample,) + single_shape)
        z_new = torch.from_numpy(temp)
        z_sample.append(z_new.to(device))
        start += length
    return z_sample

def x2img(x, bit=16): 
    if bit == 16:
        img = (np.clip(x + 0.5, 0, 1) * 65535).astype(np.uint16)
        return img
    elif bit == 8:
        img = (np.clip(x + 0.5, 0, 1) * 255).astype(np.uint8)
        return img
    else:
        print("Undefine bit")
        return x
def concat_z(z_list):   
    z_seq = None
    for i, z in enumerate(z_list):
        temp = z.cpu().data.numpy()
        temp = temp.reshape(temp.shape[0], -1)
        if i == 0:
            z_seq = temp
        else:
            z_seq = np.concatenate((z_seq, temp), axis=1)
    return z_seq


if __name__ == '__main__':

    image_size = 128   # The size of the generated image
    input_size = 128   # The size of the input image
    image_channel = 3  # The channel of the generated image
    n_flow = 32        # The number of flows in each block
    n_block = 4        # The number of blocks
    image_path = "img/cat.png"  # The path of the input image
    encrypted_key=[3.9999,0.666,5000,5000]  # Key for encryption
    decrypted_key=[3.9999,0.666,5000,5000]  # Key for decryption

    # Glow model load
    glow = Glow(in_channel=image_channel, n_flow=n_flow, n_block=n_block, affine=False, conv_lu=True)
    glow1 = glow.to(device)
    glow1.load_state_dict(torch.load(f"model_weight/Glow_0400001.pt",map_location=torch.device(device)))
    
    # DCAE model load
    DCAE = DC_AE.DCAE().to(device)
    DCAE.eval()
    DCAE.load_state_dict(torch.load(f"model_weight/DCAE.pth",map_location=torch.device(device)))

    # Image preprocessing
    image = Image.open(image_path).convert('RGB').resize((input_size, input_size))
    image_np = np.array(image)  
    image_tensor = torch.from_numpy(image_np).permute(2 ,0, 1).float() / 255.0  
    image_batch = image_tensor.unsqueeze(0).to(device)  

    # Image encoding
    with torch.no_grad():
        encoded = DCAE.encoder(image_batch) 
    encoded_np = encoded.squeeze(0).cpu().numpy().astype(np.float32)
    num = encoded_np.size   
    z1= encoded_np
    z1.resize(1,num)
    totalnum = image_size  * image_size  * image_channel    
    resnum =totalnum - num

    # Generate random noise
    seed=random.seed(time.time()) 
    z2 = np.random.RandomState(seed).randn(1, resnum)  
    z2=z2.astype(np.float32)
    z2=z2*0.7

    # Concatenate encoded data and noise
    tz = np.concatenate((z1, z2), axis=1)    

    # Encryption
    tempz =vector_chao_map.encryted(tz,encrypted_key[0],encrypted_key[1],encrypted_key[2],encrypted_key[3]) 
    
    #generate the VMEI
    z_input = split_z(tempz, image_channel, image_size, n_block)   
    float_img = glow1.reverse(z_input).cpu().data   
    plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) 
    stego_image = x2img(plot_img)    
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
    plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint16))
    plt.show()   

    img_path="img/VMEI.png"
    cv2.imwrite(img_path, stego_image)
    
    img_path="img/VMEI.png"
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img= img.astype(np.float32) / 65535.0 -0.5
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    float_img = torch.from_numpy(img)
    float_img = float_img.permute(2, 0, 1) 
    float_img = float_img.unsqueeze(0)
    float_img = float_img.to(dtype=torch.float32)

    _, _, ext_z_input = glow1.forward(float_img.to(device)) 
    ext_z = concat_z(ext_z_input)
    ext_z = ext_z[0]

    re_partz = ext_z[0:totalnum]

    re_partz =vector_chao_map.decryted(re_partz ,decrypted_key[0],decrypted_key[1],decrypted_key[2],decrypted_key[3])
    re_partz = re_partz[0:num]
    re_partz=re_partz
    re = re_partz.reshape((1, 128, 16, 16))
    
    re=torch.from_numpy(re).float().to(device)
    with torch.no_grad():
        recon_img = DCAE.decoder(re)
    
    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop((input_size,input_size)),
        transforms.ToTensor()
    ])
    

    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = recon_img  .squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_uint8 = (img_np * 255).astype(np.uint8)
    recon_uint8 = (recon_np * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    recon_bgr = cv2.cvtColor(recon_uint8, cv2.COLOR_RGB2BGR)

    psnr = peak_signal_noise_ratio(img_np, recon_np, data_range=1.0)
    ssim = structural_similarity(img_np, recon_np, data_range=1.0, channel_axis=2)
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Origin\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(recon_np)
    plt.axis('off')
    plt.show()

