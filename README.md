# Lighter U-Net
Lighter U-Net is an adaptation of the U-Netmodel but with less number of channels and encoder/decoder blocks. The model has 1.925M parameters.

# GU-Net
GU-net is an adaptation of lighter U-Net, to further decrease the number of parameters in the model by utilising ghost modules.

# RGU-Net
Implements residual module to GU-Net architecture to decrease conversion time.

# ARGU-Net
Asymetrical encoder-decoder architecture to produce more arcuate predicted whilst further decreasing the number of parameters in the model.

Python 3.9.18
PyTorch 2.1.0
