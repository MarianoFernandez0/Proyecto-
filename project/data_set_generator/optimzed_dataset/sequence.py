from track import Track, Spermatozoa, ConfigData
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    args = parser.parse_args()

    config_data = ConfigData(args.config_file)

