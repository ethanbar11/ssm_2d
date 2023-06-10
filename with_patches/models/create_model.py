from .vit import ViT
from .cait import CaiT
from .pit import PiT
from .swin import SwinTransformer
from .t2t import T2T_ViT
from models.vision_mega import VisionMEGA
from models.convit import ConvitVisionTransformer

def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=args.embed_dim,
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=args.embed_dim // 12,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA, args=args)

    elif args.model == "convit":
        patch_size = 4 if img_size == 32 else 8
        model = ConvitVisionTransformer(img_size=img_size,patch_size=patch_size, num_classes=n_classes, depth=9,
                                        embed_dim=args.embed_dim,mlp_ratio=2,num_heads=12,drop_rate=args.sd,args=args)

    elif args.model == 'cait':
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, stochastic_depth=args.sd,
                     is_LSA=args.is_LSA, is_SPT=args.is_SPT,args=args)

    elif args.model == 'pit':
        patch_size = 2 if img_size == 32 else 4
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]

        model = PiT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=args.channel,
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)

    elif args.model == 't2t':
        model = T2T_ViT(img_size=img_size, num_classes=n_classes, drop_path_rate=args.sd, is_SPT=args.is_SPT,
                        is_LSA=args.is_LSA)

    elif args.model == 'swin':
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        # num_heads = [1,1,1]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4

        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=args.sd,
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads,
                                embed_dim=args.embed_dim,
                                num_classes=n_classes,
                                is_SPT=args.is_SPT, is_LSA=args.is_LSA, args=args)
    elif args.model == 'mega':
        patch_size = 4 if img_size == 32 else 8
        model = VisionMEGA(img_size=img_size, patch_size=patch_size, num_classes=n_classes, depth=9,
                           embed_dim=args.embed_dim, hidden_dim=args.embed_dim,
                           ffn_hidden_dim=args.embed_dim, zdim=args.embed_dim, ndim=args.ndim, args=args)



    return model
