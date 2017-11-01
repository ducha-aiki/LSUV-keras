# Layer-sequential unit-variance (LSUV) initialization for Keras

This is sample code for LSUV and initializations, implemented in python script within Keras framework.

Usage:

    from lsuv_init import LSUVinit
    ...
    batch_size = 32
    model = LSUVinit(model, train_imgs[:batch_size,:,:,:])

LSUV initialization is described in:

Mishkin, D. and Matas, J.,(2015). All you need is a good init. ICLR 2016 [arXiv:1511.06422](http://arxiv.org/abs/1511.06422).

Original Caffe implementation  [https://github.com/ducha-aiki/LSUVinit](https://github.com/ducha-aiki/LSUVinit)

Torch re-implementation [https://github.com/yobibyte/torch-lsuv](https://github.com/yobibyte/torch-lsuv)

PyTorch implementation [https://github.com/ducha-aiki/LSUV-pytorch](https://github.com/ducha-aiki/LSUV-pytorch)

**New!** Thinc re-implementation [LSUV-thinc](https://github.com/explosion/thinc/blob/e653dd3dfe91f8572e2001c8943dbd9b9401768b/thinc/neural/_lsuv.py)
