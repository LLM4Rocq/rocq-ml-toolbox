from src.rocq_ml_toolbox.docker.rocq_docker import RocqDocker, OpamConfig
from src.rocq_ml_toolbox.inference.client import PetClient

try:
    opam_config = OpamConfig.from_yaml('docker/config/rocq-test.yaml')
    rocq_docker = RocqDocker(opam_config)
    rocq_docker.start_inference_server(port=5000)

    client = PetClient("http://127.0.0.1:5000")
    print("READY")
    print(client.get_session())
finally:
    if rocq_docker is not None:
        rocq_docker.close()