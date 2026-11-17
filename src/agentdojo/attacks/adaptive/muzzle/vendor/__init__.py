"""MUZZLE's iterative part, copied from SuperRed's port (`RoldSI/superred-modules`, `optimizers/muzzle`).

Every file here is byte-identical to upstream except the import lines that point at SuperRed's core
(`superred.core.llm`, `superred.core.types.llm`) or at upstream's own package name (`muzzle_optimizer.`);
`UPSTREAM.json` pins each file's upstream hash and `tests/test_attacks/test_muzzle.py` checks it.
Upstream's `muzzle_optimizer/__init__.py` is not copied, because it imports the dropped phases.
"""
