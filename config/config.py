__all__ = ['config']

config = {

    "latent_channels": 110,  # 220
    "hyperlatent_filters": 220,  # 330
    "n_downsampling_layers": 4,
    "encoder": {
        "channel_list": [60, 120, 240, 480, 960],
        'pixel_unshuffle': False,
    },

    "g_model": {
        "type": 'hific',  # 'hific'/'wdsr_b'
        "num_blocks": 7,  # default 16
        "num_residual_units": 24,  # default 32
        # "norm": 'spectral_norm',

    },

    "d_model": {
        "disc_feature": 32,  # default 64
        "disc_skip_connection": True,  # default True
        "disc_output_channels": 1,
    },

    "likelihood_type": 'gaussian',
    "mixture_components": 3,
    "use_latent_mixture_model": False,

    "optim": {
        "type": "Adam",
        "lr": 1e-4,
        "betas": (0.9, 0.999),
    },

    "scheduler": {
        "type": "MultiStepLR",
        "milestones": [0.7, 0.85],
        "gamma": 0.2,
    },

    # "scheduler": {
    #     "type": "CosineAnnealingRestartLR",
    #     "restart_weights": (1, 0.8, 0.6, 0.4, 0.2),
    # },

    "losses": {
        "pixel": {
            "type": "PixelWiseLoss",
            "criterion": 'mse',
            "loss_weight": 0.075 * 2**(-5),
            "scalar": 255,
        },
        # lambda_B = 2 ** (-4)
        # lambda_A_map = dict(low=2 ** 1, med=2 ** 0, high=2 ** (-1))
        # lambda_A = lambda_A_map[regime]
        # target_rate_map = dict(low=0.14, med=0.3, high=0.45)
        # target_rate = target_rate_map[regime]
        "bpp_rate": {
            "type": "BitPerPixelLoss",
            "lambda_a": 2**(-1),
            "lambda_b": 2**(-4),
            "lambda_schedule": dict(vals=[2., 1.], steps=[0.1]),
            "target_bpp": 0.45,
            "target_schedule": dict(vals=[0.8, 1.], steps=[0.1]),
        },
        # "perceptual": {
        #     "type": "PerceptualLoss",
        #     "layer_weights": {'34': 1.0},
        #     "vgg_type": 'vgg19',
        #     "norm_img": False,
        #     "criterion": 'l1',
        #     "pretrained": 'torchvision://vgg19',
        #     "perceptual_weight": 10,
        # },
        # "perceptual": {
        #     "type": "PerceptualLoss",
        #     "layer_weights": {'2': 0.1, '7': 0.1, '16': 1, '25': 1, '34': 1, },
        #     "vgg_type": 'vgg19',
        #     "norm_img": False,
        #     "criterion": 'l1',
        #     "pretrained": 'torchvision://vgg19',
        #     "perceptual_weight": 1
        # },
        "perceptual": {
            "type": "LPIPSLoss",
            "loss_weight": 1,
            "lpips_loss_arch": "alex",
            "normalize": True,
        },

        "adversarial": {
            "type": "GANLoss",
            "gan_type": 'vanilla',
            "loss_weight": 0.15
        },
    }
}
