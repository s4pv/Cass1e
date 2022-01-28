class Helper:
    def load_config(file):
        import yaml
        try:
            with open(file) as file:
                return yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError as fe:
            exit(f'Could not find {file}')
        except Exception as e:
            exit(f'Encountered exception...\n {e}')

