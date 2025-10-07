python3 scripts/deep_search.py \
    --input_path ./data/BrowseCompEN-Sample100/all_data_random100_sample1.json \
    --output_dir ./outputs/BrowseCompEN-Sample100/qwen3-235b-a22b-2507_main_kimi-k2_aux/max_search_calls_15/parallel_sample \
    --use_aihubmix \
    --aihubmix_api_url "https://openrouter.ai/api/v1/chat/completions" \
    --aihubmix_api_keys "" \
    --price_config_path "./model_config/openrouter_price.json" \
    --use_google_pro \
    --google_pro_api_key "" \
    --use_custom_api \
    --aux_model_name "moonshotai/kimi-k2" \
    --max_tokens 16384 \
    --model_name "qwen/qwen3-235b-a22b-2507" \
    --top_k 10 \
    --concurrent_limit 100 \
    --max_search_calls 15 \
    --mode "solve" \


