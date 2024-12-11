#!/bin/bash

# List of different model names
model_names=(
    $1
)

# Create a directory to store output files
output_dir="output_tables"
mkdir -p $output_dir

# tasks="mmlu,mmlu_continuation,mmlu_generative,mmlu_full_choice"
# tasks="winogrande,arc_challenge,hellaswag,mmlu,gsm8k,truthfulqa_mc2"
# tasks="winogrande_ru,arc_challenge_ru,hellaswag_ru,mmlu_ru,gsm8k_ru,truthfulqa_mc2_ru"
# tasks="mmlu,mmlu_continuation,mmlu_ru,mmlu_ru_continuation,mmlu_ru_mera,mmlu_ru_mera_continuation"
# tasks="hellaswag_ru,hellaswag"
tasks="mmlu,mmlu_continuation --num_fewshot 5"
# tasks="mmlu_generative --num_fewshot 5"


# Iterate over the list and run the command
for model_name in "${model_names[@]}"; do
  model_args="pretrained=$model_name"
  command="accelerate launch -m lm_eval --model hf --tasks $tasks --batch_size 4 --model_args $model_args"
  output_file="$output_dir/output_${model_name//\//_}.txt"  # Replace "/" with "_"
  echo "Running command: $command"
  $command > $output_file  # Redirect output to file
done

# Print the contents of the output files
for output_file in $output_dir/*.txt; do
  echo -e "\nContents of $output_file:"
  cat $output_file
done

# Clean-up
rm -r $output_dir
