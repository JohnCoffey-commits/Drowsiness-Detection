# Stage 14 Recovered Checkpoint Verification

## Result

`CHECKPOINT_VERIFICATION_PASSED`

The recovered Stage 7 mouth/yawn checkpoint was loaded locally and verified against torchvision ResNet18 with a two-class classifier head.

## Verification Evidence

```text
checkpoint exists: True
checkpoint size: 44790859
payload type: <class 'dict'>
top-level keys: ['model', 'model_state_dict', 'history', 'best_epoch', 'class_to_index', 'image_size', 'batch_size', 'pretrained_used']
state_dict source: payload['model_state_dict']
sample keys: ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked', 'layer1.1.conv1.weight', 'layer1.1.bn1.weight']
fc.weight shape: (2, 512)
fc.bias shape: (2,)
missing keys: []
unexpected keys: []
logits shape: (1, 2)
probs shape: (1, 2)
dummy probs: [[0.8352651596069336, 0.16473488509655]]
p_yawn index valid: True
CHECKPOINT_VERIFICATION_PASSED

```

## Confirmed Runtime Semantics

- Checkpoint path: `checkpoints/resnet18_best.pt`
- Payload key used: `model_state_dict`
- Architecture: torchvision ResNet18 with `model.fc = nn.Linear(model.fc.in_features, 2)`
- Classifier head shape: `fc.weight = (2, 512)`, `fc.bias = (2,)`
- Dummy output shape: `(1, 2)`
- Label mapping from checkpoint metadata: expected `{"no_yawn": 0, "yawn": 1}`
- `p_yawn` class index: `1`

No training was run. No checkpoint was modified after copying.
