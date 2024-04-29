import numpy as np
from skimage import io, color, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import binom, norm
from scipy.optimize import minimize

from image_py_scripts import gaussian_noise

# Function to implement image inpainting
def image_inpainting(M, M_Omega, rak, maxiter, model):
    # Placeholder function for image inpainting
    # Replace with your image inpainting implementation
    # Call trained model and get reconstructed M_Omega
    M_omega_reconstructed = model(M_Omega)
    PSNR = psnr(M, M_omega_reconstructed)
    SSIM = ssim(M, M_omega_reconstructed)
    return PSNR, SSIM

# Parameters
r, c, rak = 400, 500, 10
dB = 5
per = 0.5
models = ['L0-BCD', 'Lp-reg', 'Lp-ADMM', 'ORMC', 'M-Estimation']
PSNRs = np.zeros((len(models), 8))
SSIMs = np.zeros((len(models), 8))

# Loop through models and images
for m in range(len(models)):
    for i in range(1, 9):
        # Read Image and convert into black and white and reshape into 150 x 300
        # Replace the path with your own
        image_path = f'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/M-estimation-RMC/M-Estimation/Image_Inpainting_Dataset/{i}.jpg'
        M = io.imread(image_path)
        M = color.rgb2gray(img_as_float(M))
        M = np.resize(M, (r, c))

        # Add GMM noise
        array_Omega = binom.rvs(1, per, size=(r, c))
        M_Omega = M * array_Omega
        omega = np.where(array_Omega == 1)
        noise = gaussian_noise(M_Omega[omega], 1, dB)
        Noise = np.zeros_like(M_Omega)
        Noise[omega] = noise
        M_Omega = M_Omega + Noise
        maxiter = 50

        PSNR, SSIM = image_inpainting(M, M_Omega, rak, maxiter, models[m])
        print(f'Model: {models[m]}, PSNR of Image {i} = {PSNR}, SSIM of Image {i} = {SSIM}')

        PSNRs[m, i-1] = PSNR
        SSIMs[m, i-1] = SSIM
