python3 scripts/deep_search.py \
    --input_path ./outputs/policy_results_08-01_02-54-47.json \
    --output_dir ./outputs/solve_budget_forcing_results \
    --use_aihubmix \
    --aihubmix_api_url "https://openrouter.ai/api/v1/chat/completions" \
    --aihubmix_api_keys "" \
    --price_config_path "./model_config/openrouter_price.json" \
    --use_google_pro \
    --google_pro_api_key  \
    --use_custom_api \
    --aux_model_name "moonshotai/kimi-k2" \
    --max_tokens 16384 \
    --model_name "qwen/qwen3-235b-a22b-2507" \
    --top_k 10 \
    --concurrent_limit 100 \
    --max_search_calls 15 \
    --mode "solve_budget_forcing" \


