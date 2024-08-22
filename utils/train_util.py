import yaml




def parse(yaml_path):
    with open(yaml_path, "r") as f:
        ayml = yaml.load(f.read(), Loader = yaml.Loader)
        
    return ayml
