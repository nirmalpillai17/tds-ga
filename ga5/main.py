import pathlib
import json

data_path = pathlib.Path().absolute() / "data"

# QN-5

"""
curl -X POST http://aiproxy.sanand.workers.dev/openai/v1/embeddings
        -H "Content-Type: application/json"
        -H "Authorization: Bearer $AIPROXY_TOKEN"
        -d '{"model": "text-embedding-3-small", "input": ["Original"]}'

"""

ARBITRARY = 0.020717579993515034

with open(data_path / "embedding_original.json") as f:
    values = json.load(f)

final = list(filter(lambda x: x > ARBITRARY,
                        values["data"][0]["embedding"]))
print(f"Values: {final}\nCount: {len(final)}")


# QN-6

"""
curl -X POST http://aiproxy.sanand.workers.dev/openai/v1/embeddings
        -H "Content-Type: application/json"
        -H "Authorization: Bearer $AIPROXY_TOKEN"
        -d '{"model": "text-embedding-3-small", \
             "input": ["Economic", "Essential"]}' | \
                                    tee economic_essential.json
"""

import numpy as np
from numpy.linalg import norm

with open(data_path / "economic_essential.json") as f:
    values = json.load(f)

economic_values = np.array(values["data"][0]["embedding"])
essential_values= np.array(values["data"][1]["embedding"])

cosine = np.dot(economic_values, essential_values) / \
                (norm(economic_values) * norm(essential_values))

print("Cosine Similarity:", cosine)


# QN-3, QN-4

"""
with open(data_path / "qn3_data.txt") as f:
    lines = f.readlines()

for line in lines:
    line.split(":")
"""

# Ok, I took a look at the data and decided programming 
# this ain't worth it. I'll just ask ChatGPT -\_(^^)_/-

# Guess what?! It worked surprisingly well!


