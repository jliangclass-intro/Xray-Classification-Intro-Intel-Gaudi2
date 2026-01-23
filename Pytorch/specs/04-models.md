# Specification: Models

## Backbones (timm)
- `resnet50`
- `convnext_base`
- `swin_base_patch4_window7_224`

## Head replacement
- Use `timm.create_model(..., num_classes=...)` to set the classifier head.

## Pretrained weights
- Use timm pretrained weights when `pretrained: true`.
