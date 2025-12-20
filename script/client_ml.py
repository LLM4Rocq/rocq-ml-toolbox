import os
import json
from tqdm import tqdm

from src.rocq_ml_toolbox.inference.client import PetClient

client = PetClient('http://127.0.0.1:5000')
client.connect()
filepath = "/home/theo/.opam/mc_dev/lib/coq/user-contrib/Stdlib/Structures/OrderedTypeEx.v"
fleche_document = client.get_document(filepath)

with open('export.jsonl', 'w') as file:
    # need to remove the last element (EOF)
    for ranged_span in tqdm(fleche_document.spans[:-1]):
        text = ranged_span.span
        r = ranged_span.range

        ast = client.ast_at_pos(filepath, r.start.line, r.start.character)
        file.write(json.dumps((r.start.line, ast)) + '\n')