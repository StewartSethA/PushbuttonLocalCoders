#!/usr/bin/env bash
set -euo pipefail

# Regression for backend-row parsing: an empty extra_env field must not collapse
# and shift the human-readable note into the env command.
sep=$'\x1f'
row="id${sep}repo:quant${sep}1${sep}0${sep}q4_0${sep}q4_0${sep}512${sep}256${sep}on${sep}qwen-fixed${sep}${sep}full-context 16 GB profile"
IFS=$'\x1f' read -r id hf gpus multi kvk kvv batch ubatch fa template envpairs note <<< "$row"

[[ "$id" == id ]]
[[ "$hf" == repo:quant ]]
[[ "$template" == qwen-fixed ]]
[[ -z "$envpairs" ]]
[[ "$note" == "full-context 16 GB profile" ]]
