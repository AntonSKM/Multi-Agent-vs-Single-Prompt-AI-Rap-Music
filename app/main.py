import json
from app.experiment_setup import generate_initial_packages, run_experiment


def main():
    # load config file
    with open("config.json", "r", encoding="utf-8") as file: config = json.load(file)

    # generate initial packages and then run experiment with them
    print("### experiment started ###")
    packages = generate_initial_packages(config)
    run_experiment(config, packages)



if __name__ == "__main__":
    main()
