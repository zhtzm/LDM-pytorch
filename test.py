from utils import parse_yaml_config

if __name__ == "__main__":
    try:
        config, _, _, _ = parse_yaml_config("cfg/ldm_s256_nc.yaml")
        for key, value in config.items():
            print(key, value)
        print(config)  # 打印解析后的配置数据
    except Exception as e:
        print(e)
