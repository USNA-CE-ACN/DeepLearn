import json
import os

def generate_json_file(input_file):
    os.system("flatc -t --strict-json --defaults-json schema.fbs -- " + input_file)

def parse_json_file(input_json):
    json_ops = {}
    with open(input_json) as f:
        model_json = json.load(f)
        sg_ops = model_json["subgraphs"][0]["operators"]
        for sg_op in sg_ops:
            # Look up json based on output number to find the rest
            json_ops[sg_op["outputs"][0]] = sg_op

    return json_ops
