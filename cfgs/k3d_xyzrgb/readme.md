# default.yaml

Папка [cfgs/k3d_xyzrgb](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb) повторяет структуру [cfgs/k3d_xyz](/home/researcher/dev/magicnet/cfgs/k3d_xyz), но настроена для датасета `K3DXYZRGB`.

Главные отличия от `k3d_xyz`:

- `dataset.common.NAME: K3DXYZRGB`
- `dataset.common.data_root: data/k3d_xyzrgb`
- базовый вход модели: `feature_keys: pos,x,heights`
- итоговая размерность входа: `xyz + rgb + height = 7` каналов
- в transforms добавлены RGB-аугментации и `ChromaticNormalize`

Структура сохранена той же:

- [default.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/default.yaml)
- [dgcnn/dgcnn.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/dgcnn/dgcnn.yaml)
- [pointnet++/pointnet++.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/pointnet++/pointnet++.yaml)
- [pointnext/pointnext-s.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/pointnext/pointnext-s.yaml)
- [pointnext/pointnext-b.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/pointnext/pointnext-b.yaml)
- [pointnext/pointnext-l.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/pointnext/pointnext-l.yaml)
- [pointtransformer/pointtransformer.yaml](/home/researcher/dev/magicnet/cfgs/k3d_xyzrgb/pointtransformer/pointtransformer.yaml)

Быстрые команды запуска:

```bash
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/pointnet++/pointnet++.yaml
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/pointnext/pointnext-s.yaml
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/dgcnn/dgcnn.yaml
python examples/segmentation/main.py --cfg cfgs/k3d_xyzrgb/pointtransformer/pointtransformer.yaml
```

Если нужна максимально близкая к старому `k3d_xyz` логика сравнения, начни с:

- `pointnet++/pointnet++.yaml`
- `pointnext/pointnext-s.yaml`

Обе конфигурации сохраняют ту же общую архитектуру, но добавляют RGB к исходному геометрическому входу.
