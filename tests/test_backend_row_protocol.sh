#!/usr/bin/env bash
set -euo pipefail

# Regression for backend-row parsing: an empty extra_env field must not collapse
# and shift the human-readable note into the env command.
sep=$'\x1f'
capacity='{"context":8192,"slots":3,"output_tokens":1024,"client_context":8192,"compact_trigger":5000,"input_tokens":6912,"safety_tokens":256,"min_tps":null,"admission_limit":2}'
row="id${sep}repo:quant${sep}1${sep}0${sep}q4_0${sep}q4_0${sep}512${sep}256${sep}on${sep}qwen-fixed${sep}${sep}full-context 16 GB profile${sep}8192${sep}3${sep}${capacity}"
IFS=$'\x1f' read -r id hf gpus multi kvk kvv batch ubatch fa template envpairs note context slots decoded_capacity <<< "$row"

[[ "$id" == id ]]
[[ "$hf" == repo:quant ]]
[[ "$template" == qwen-fixed ]]
[[ -z "$envpairs" ]]
[[ "$note" == "full-context 16 GB profile" ]]
[[ "$context" == 8192 && "$slots" == 3 && "$decoded_capacity" == "$capacity" ]]

# Independent workers also have an optional, empty lease before capacity fields.
row="1${sep}${row%${sep}8192${sep}3${sep}${capacity}}${sep}${sep}8192${sep}3${sep}${capacity}"
IFS=$'\x1f' read -r worker id hf gpus multi kvk kvv batch ubatch fa template envpairs note vramcap context slots decoded_capacity <<< "$row"
[[ "$worker" == 1 && -z "$vramcap" && -z "$envpairs" ]]
[[ "$context" == 8192 && "$slots" == 3 && "$decoded_capacity" == "$capacity" ]]
