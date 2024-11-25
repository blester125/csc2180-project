#!/usr/bin/env python3

import json

import matplotlib.pyplot as plt
from scaling import Config

with open("results.json") as f:
    results = json.load(f)
    results = {Config.from_string(k): v for k, v in results.items()}


classes = {
    k: v for k, v in results.items() if k.num_features == 2 and k.num_samples == 50
}
classes = dict(sorted(classes.items(), key=lambda c: c[0].num_classes))
print(classes)

plt.plot([c.num_classes for c in classes], classes.values())
plt.show()
