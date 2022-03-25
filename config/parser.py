import yaml
import os
import json


class Parser:
    def __init__(self):
        pass

    def save(self, profile_type):
        pass

    def load(self, name=None, profile_type=None):
        pass

    @staticmethod
    def parse(file_path: str, profile_type=None) -> dict:
        assert os.path.isfile(file_path)
        if profile_type is None:
            profile_type = os.path.splitext(file_path)[-1]
        if profile_type == ".yml" or profile_type == ".yaml":
            return Parser.parse_yaml(file_path)
        if profile_type == ".json":
            return Parser.parse_json(file_path)
        raise Exception(print("Unable to determine profile type.You can try setting it manually!"))

    @staticmethod
    def merge_data(data_1, data_2):
        if isinstance(data_1, dict) and isinstance(data_2, dict):
            new_dict = {}
            d2_keys = list(data_2.keys())
            for d1k in data_1.keys():
                if d1k in d2_keys:
                    d2_keys.remove(d1k)
                    new_dict[d1k] = Parser.merge_data(data_1.get(d1k), data_2.get(d1k))
                else:
                    new_dict[d1k] = data_1.get(d1k)
            for d2k in d2_keys:
                new_dict[d2k] = data_2.get(d2k)
            return new_dict
        else:
            if data_2 is None:
                return data_1
            else:
                return data_2

    @staticmethod
    def parse_yaml(file_path: str) -> dict:
        with open(file_path, "r", encoding="UTF-8") as f:
            data_iter = yaml.unsafe_load_all(f)
            main_config = next(data_iter)
            active_config = main_config.get("project").get("active_config")
            if not active_config is None:
                for data in data_iter:
                    if data.get("project").get("config_name") == active_config:
                        main_config = Parser.merge_data(main_config, data)
                        break
        return main_config

    @staticmethod
    def parse_json(file_path: str) -> dict:
        pass
