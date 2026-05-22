# Run Evaluation
# 1. Import traces if not alrady imported - use mlflow run id
# 2. Parse traces - clean up if columns are empty
# 3. Run evaluation code on parsed traces
# 3.a deterministic scorers first
# 3.b LLM judge if requested by config
#
# Details of the run are defined in the run config file
# Report is generated separately once the results are saved in csv

 