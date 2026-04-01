from tqdm import tqdm
import time
import json
import os
import concurrent.futures

from src.rocq_ml_toolbox.docker.opam_docker import OpamDocker, OpamConfig
from src.rocq_ml_toolbox.parser.rocq_parser import RocqParser, Source
from src.rocq_ml_toolbox.inference.client import PytanqueExtended

opam_docker = None

def do_task(source: Source, url: str='127.0.0.1', port: int=5000, attempt:int=3):
    client = PytanqueExtended(url, port)
    client.connect()
    parser = RocqParser(client)

    source_path_normalize = source.path.replace('/', '.')
    export_path = f"script/export/{source_path_normalize}.jsonl"
    cache = set()
    if os.path.exists(export_path):
        with open(export_path, 'r') as file:
            for line in file:
                content = json.loads(line)
                fqn = content['element']['data']['fqn']
                cache.add(fqn)
    try:
        with open(export_path, 'a+') as file:
            for element, steps_raw in parser.extract_proofs_raw(source):
                if element.data['fqn'] in cache:
                    continue
                for att in range(1, attempt+1):
                    try:
                        proof = parser.extract_full_proof(source, element, steps_raw, timeout=120)
                        file.write(json.dumps(proof.to_json()) + '\n')
                        break
                    except Exception as e:
                        if att == attempt:
                            raise Exception(f"issue at {source.path} on {element.name}")
    except:
        raise Exception(source.path)

try:
    # opam_config = OpamConfig.from_yaml('docker/config/rocq-test.yaml')
    # opam_docker = OpamDocker(opam_config)
    # opam_docker.start_inference_server(port=5000)

    
    print("READY")
    # source_path = '/home/rocq/.opam/4.14.2+flambda/lib/coq/user-contrib/Stdlib/Arith/PeanoNat.v'
    # source_path = '/home/rocq/.opam/4.14.2+flambda/lib/coq/user-contrib/mathcomp/algebra/fraction.v'

    # source = opam_docker.get_source(source_path)
    # proofs = parser.extract_proofs(source)
    # for element, proof_steps in proofs:
    #     theorem = parser.compute_theorem(element, proof_steps)
    #     print(theorem)
    #     exit()
    # with open('export.jsonl', 'w') as file:
    #     for entry in ast:
    #         file.write(json.dumps(entry) +"\n")
    # exit()
    # with open('test.v', 'r') as file:
    #     content = file.read()
    
    # source = opam_docker.upload_source(content)
    # local_content = opam_docker.read_file(source.path)
    # filepath = '/home/rocq/.opam/4.14.2+flambda/lib/coq/user-contrib/Stdlib/Structures/OrderedTypeEx.v'
    # # source = opam_docker.get_source(filepath)
    # ast = parser.extract_ast(source, retry=1)
    # with open('test.jsonl', 'w') as file:
    #     for entry in ast:
    #         file.write(json.dumps(entry) + '\n')
    # exit()
    # exit()
    # state = client.get_state_at_pos(source.path, 37, 0)
    # for folder_name in opam_docker.list_opam_folder():
    # sources = opam_docker.extract_source_files_from_folder('Stdlib')
    # source = opam_docker.upload_source(content)

    lib_path = '/home/theo/.opam/mc_dev/lib/coq/user-contrib/mathcomp/'
    to_do = []
    for dirpath, _, filenames in os.walk(lib_path):
        for filename in filenames:
            if not filename.endswith('.v'):
                continue

            filepath = os.path.join(dirpath, filename)
            source = Source.from_local_path(filepath)
            to_do.append(source)
    
    # source = Source.from_local_path('/home/theo/.opam/mc_dev/lib/coq/user-contrib/mathcomp/algebra/ssralg.v')
    # do_task(source)
    # exit()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(do_task, source) for source in to_do]

        for f in tqdm(concurrent.futures.as_completed(futures)):
            f.result()
                # print( proof.element.name)
                # for step in proof.steps:
                #     print(step.step)
                #     for k, dep in enumerate(step.dependencies):
                #         print(f"{k}. {dep.name}")
                #         print(dep.range)
                # print(proof)
        # parser.add_logical_path(source)
        # print(source.path)
        # print(source.logical_path)
        # toc = parser.extract_toc(source, retry=1)
        # proofs = parser.extract_proofs(source)
        # print("Extraction Begin")
        # for element, proof_steps in proofs:
        #     print(element)
        #     theorem = parser.execute_proof(element, proof_steps)
        #     print(theorem)
        # print("End extraction")
finally:
    if opam_docker is not None:
        opam_docker.close()