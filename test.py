import mmcv
print(hasattr(mmcv, 'Config'))  # 检查根模块中是否有 Config
print(hasattr(mmcv.utils.config, 'Config'))  # 检查 utils.config 模块中是否有 Config