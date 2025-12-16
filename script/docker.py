from rocq_ml_toolbox.docker.opam_docker import OpamDocker, OpamConfig
from src.rocq_ml_toolbox.inference.client import PetClient

try:
    opam_config = OpamConfig.from_yaml('docker/config/rocq-test.yaml')
    opam_docker = None
    opam_docker = OpamDocker(opam_config)
    opam_docker.start_inference_server(port=5000)

    client = PetClient("http://127.0.0.1:5000")
    print("READY")
    print(client.get_session())

    print(opam_docker.list_opam_folder())
    for package_name in opam_config.packages:
        print(package_name)
        print(opam_docker.extract_files(package_name, {'rocq-metarocq': 'MetaRocq'}))
finally:
    if opam_docker is not None:
        opam_docker.close()