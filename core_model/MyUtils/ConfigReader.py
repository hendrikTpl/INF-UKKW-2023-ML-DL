# /**
#  * @author Hendrik
#  * @email hendrik.gian@gmail.com
#  * @create date 2023-03-16 13:42:16
#  * @modify date 2023-03-16 13:42:16
#  * @desc [description]
#  */

import argparse
import yaml

class ConfigReader():
    def __init__(self, config_file, args=None):
        self.config_file = config_file
        self.args = args or {}
        self.config_data = None

    def read_config(self):
        with open(self.config_file, 'r') as f:
            self.config_data = yaml.load(f, Loader=yaml.FullLoader)

        # Override any settings with command-line arguments
        if self.args:
            for key, value in self.args.items():
                keys = key.split('.')
                config = self.config_data
                for k in keys[:-1]:
                    config = config[k]
                config[keys[-1]] = value

        return self.config_data

## test case
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/base_config.yaml', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config_reader = ConfigReader(args.config, args=vars(args))
    config_data = config_reader.read_config()
    print(config_data['BATCH_SIZE'])
    print(config_data)
