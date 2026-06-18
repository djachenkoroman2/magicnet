# dgcnn.yaml

Файл [dgcnn.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/dgcnn/dgcnn.yaml) загружается вместе с [default.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/default.yaml) через `cfg.load(..., recursive=True)`.

Что важно в версии для `K3DXYZRGB`:

- `dataset.common.NAME: K3DXYZRGB`
- `feature_keys: pos,x,heights`
- `encoder_args.in_channels: 7`

То есть DGCNN получает на вход `xyz + rgb + height = 7` каналов на точку. Геометрия сцены по-прежнему идёт и отдельно через `pos`, а RGB добавляется как полноценные признаки.

Остальная логика совпадает с конфигом для `k3d_xyz`:

- `batch_size: 1`
- `step_per_update: 2`
- `dataset.train.voxel_max: 6000`
- backbone `DGCNN` с `k=20` и `embed_dim=1024`

Быстрый запуск:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/dgcnn/dgcnn.yaml
```
