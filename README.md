# ML_Final_Project

Group Members: Sri Guru Ananth Gadicherla, Kavya Yangala

# Noise2Self

## Environment Requirements
- jupyter
- python == 3.7.2
- tensorflow >=1.10 & <=1.15
- scipy
- skimage
- tifffile

#### To be specific, the following code is used to build the model.
```
from models import Noise2Self
model = Noise2Self(model_dir, model_name, dimension, in_channels)
```
where ``model_dir`` and ``model_name`` will specify the path to your checkpoint files, ``dimension`` refers to the dimension of image *(2 or 3)* and ``in_channels`` refers to the number of channels of input images.

#### The following codes are for **prediction**.

- For prediction of image,
  
  model.predict(img[, im_mean, im_std])
  
  where ``img`` is the noisy image for prediction, ``im_mean`` and ``im_std`` are the mean and standard deviation. If ``im_mean`` and ``im_std`` are not specified, it will use ``img.mean()`` and ``img.std()`` by default.


Given the noisy images ``images``, the masked noisy images ``masked_images`` and masking map ``mask`` with masked locations being 1 and other 0,

net = YourNetwork()
# The two net() below should share their weights
out_raw = net(images)
out_masked = net(masked_images) 

l_rec = reduce_mean((out_raw - images)^2)
l_inv = reduce_sum((out_raw - out_masked)^2 * mask) / reduce_sum(mask)
loss = l_rec + 2 * sqrt(l_inv)



