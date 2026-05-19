import timm

model_name = 'convnext_small.dinov3_lvd1689m'

model = timm.create_model(model_name, pretrained=True)

print(model)
